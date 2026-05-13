from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from src.realtime_engine import RealtimeFraudEngine
from src.simulation.attack_simulator import FraudAttackSimulator


def _counts_tp_tn_fp_fn(cm: np.ndarray) -> dict[str, int]:
    tn = int(cm[0, 0])
    fp = int(cm[0, 1])
    fn = int(cm[1, 0])
    tp = int(cm[1, 1])
    return {"TN": tn, "FP": fp, "FN": fn, "TP": tp}


def _print_evaluation_block(
    title: str,
    y_true: list[int],
    y_pred: list[int],
    *,
    latencies_ms: list[float] | None = None,
) -> None:
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)

    print(f"\n{'=' * 20} {title} {'=' * 20}")
    print(
        "Ground truth — legitimate (0):",
        int((y_true_arr == 0).sum()),
        "| fraud (1):",
        int((y_true_arr == 1).sum()),
    )
    print(
        "Predicted   — legitimate (0):",
        int((y_pred_arr == 0).sum()),
        "| fraud (1):",
        int((y_pred_arr == 1).sum()),
    )

    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1])
    c = _counts_tp_tn_fp_fn(cm)
    print("\nConfusion matrix [rows = actual, cols = predicted] (0=legit, 1=fraud):")
    print(cm)
    print(f"TP={c['TP']}  TN={c['TN']}  FP={c['FP']}  FN={c['FN']}")

    print("\nClassification report (fraud = positive class):")
    print(
        classification_report(
            y_true_arr,
            y_pred_arr,
            labels=[0, 1],
            target_names=["legitimate", "fraud"],
            digits=4,
            zero_division=0,
        )
    )

    if latencies_ms:
        arr = np.asarray(latencies_ms, dtype=float)
        print(
            f"\nRealtime latency — mean: {arr.mean():.2f} ms | "
            f"p95: {np.percentile(arr, 95):.2f} ms | max: {arr.max():.2f} ms"
        )


def run_realtime_flood_with_engine(
    *,
    data_path: Path,
    block_threshold: float = 0.3,
    alert_threshold: float = 0.12,
    flood_repeats: int = 12,
    max_base_sample: int = 280,
    enable_engine_logging: bool = False,
) -> dict:
    """
    Flood legitimate + fraud transactions through ``FraudAttackSimulator``, score each row
    with ``RealtimeFraudEngine``, and collect labels for evaluation.

    Predicted transaction type: ``1`` = fraud if fraud_probability >= ``block_threshold``
    (same rule as the batch simulation decision engine).
    """
    df = pd.read_csv(data_path)

    sample_n = min(max_base_sample, len(df))
    fraud_df = df[df["isFraud"] == 1]
    legit_df = df[df["isFraud"] == 0]
    fraud_take = min(len(fraud_df), max(35, sample_n // 6))
    legit_take = min(len(legit_df), max(sample_n - fraud_take, 1))
    sample_df = pd.concat(
        [
            fraud_df.sample(fraud_take, random_state=42),
            legit_df.sample(legit_take, random_state=42),
        ],
        ignore_index=True,
    ).sample(frac=1.0, random_state=42)

    y_sample = sample_df["isFraud"]
    X_sample = sample_df.drop(columns=["isFraud", "isFlaggedFraud"])

    simulator = FraudAttackSimulator(X_sample, y_sample)
    suite = simulator.build_attack_suite(flood_repeats=flood_repeats)

    engine = RealtimeFraudEngine(
        block_threshold=block_threshold,
        alert_threshold=alert_threshold,
        enable_logging=enable_engine_logging,
        skip_explanations=True,
    )

    y_true_all: list[int] = []
    y_pred_all: list[int] = []
    latencies: list[float] = []
    per_scenario: dict[str, dict[str, list]] = {}

    for scenario_name, X_attack, y_attack in suite:
        y_true_s: list[int] = []
        y_pred_s: list[int] = []
        for i in range(len(X_attack)):
            tid = f"{scenario_name}-{i}"
            tx = X_attack.iloc[i].to_dict()
            tx["transaction_id"] = tid
            out = engine.process_transaction(tx, transaction_id=tid)
            yt = int(y_attack.iloc[i])
            yp = engine.predicted_fraud_label(out.fraud_probability)

            y_true_all.append(yt)
            y_pred_all.append(yp)
            latencies.append(out.latency_ms)

            y_true_s.append(yt)
            y_pred_s.append(yp)

        per_scenario[scenario_name] = {"y_true": y_true_s, "y_pred": y_pred_s}

    return {
        "y_true": y_true_all,
        "y_pred": y_pred_all,
        "latencies_ms": latencies,
        "per_scenario": per_scenario,
        "threshold": block_threshold,
        "total_transactions": len(y_true_all),
    }


def main() -> None:
    root = Path(__file__).resolve().parent
    data_path = root / "data" / "processed" / "processed_paysim.csv"

    BLOCK_T = 0.3
    ALERT_T = 0.12
    FLOOD_REPEATS = 12

    print(
        "\n>>> Flooding the realtime pipeline (attack simulator → RealtimeFraudEngine → "
        "binary fraud type @ block threshold). Hold logs quiet for speed.\n"
    )

    summary = run_realtime_flood_with_engine(
        data_path=data_path,
        block_threshold=BLOCK_T,
        alert_threshold=ALERT_T,
        flood_repeats=FLOOD_REPEATS,
        enable_engine_logging=False,
    )

    print(
        f"Threshold for predicted fraud (transaction type): P(fraud) >= {summary['threshold']}"
    )
    print(f"Total streamed transactions: {summary['total_transactions']}")

    for scenario, blocks in summary["per_scenario"].items():
        _print_evaluation_block(
            f"Scenario: {scenario}",
            blocks["y_true"],
            blocks["y_pred"],
        )

    _print_evaluation_block(
        "OVERALL (all scenarios)",
        summary["y_true"],
        summary["y_pred"],
        latencies_ms=summary["latencies_ms"],
    )


if __name__ == "__main__":
    main()
