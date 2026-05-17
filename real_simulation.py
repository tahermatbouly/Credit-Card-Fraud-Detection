"""
real_simulation.py

Purpose:
    1) Load the saved best fraud model
    2) Load processed PaySim data
    3) Build 8 simulation scenarios
    4) Score every transaction with P(fraud)
    5) Convert probability to ALLOW / REVIEW / BLOCK
    6) Print per-scenario and total metrics
    7) Save detailed simulation results to CSV

Run from the project root after training:
    python real_simulation.py

Required files:
    data/processed/processed_paysim.csv
    output/models/best_model.pkl
    output/models/feature_names.json

Outputs:
    output/simulation/simulation_summary.csv
    output/simulation/simulation_detailed_results.csv
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


DATA_PATH = Path("data/processed/processed_paysim.csv")
MODEL_PATH = Path("output/models/best_model.pkl")
FEATURE_NAMES_PATH = Path("output/models/feature_names.json")
OUTPUT_DIR = Path("output/simulation")

TARGET_COL = "isFraud"
DROP_COLS = ["isFraud", "isFlaggedFraud"]

RANDOM_STATE = 42
BASE_SAMPLE_SIZE = 500
BLOCK_THRESHOLD = 0.30
ALERT_THRESHOLD = 0.12
FLOOD_REPEATS = 12

SCENARIO_DESCRIPTIONS: dict[str, str] = {
    "01_normal_baseline": (
        "Baseline traffic using the stratified sample without feature manipulation."
    ),
    "02_random_noise_flood": (
        "Gaussian multiplicative noise on numeric features (one-hot type columns excluded)."
    ),
    "03_micro_transaction_probe": (
        "Amount scaled to 2% of original — probes micro-payment abuse patterns."
    ),
    "04_high_value_flood": (
        "Transaction amounts multiplied by 12× with derived amount_log and is_high_amount updated."
    ),
    "05_rapid_fire_flood": (
        "The base batch repeated many times to simulate transaction flooding / bot traffic."
    ),
    "06_transfer_cashout_attack": (
        "Forces TRANSFER (or CASH_OUT) type and marks high_risk_transaction."
    ),
    "07_balance_draining_attack": (
        "Amount set near sender balance (95% drain) with balance fields reconciled."
    ),
    "08_mixed_obfuscation_attack": (
        "Combined high value, risky type, balance draining, and light noise — hardest scenario."
    ),
}


# -----------------------------------------------------------------------------
# Loading
# -----------------------------------------------------------------------------

def load_feature_names() -> list[str]:
    if not FEATURE_NAMES_PATH.exists():
        raise FileNotFoundError(
            f"Missing feature names file: {FEATURE_NAMES_PATH}\n"
            "Run train_choose_simulating_model.py first."
        )
    with open(FEATURE_NAMES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing model file: {MODEL_PATH}\n"
            "Run train_choose_simulating_model.py first."
        )
    return joblib.load(MODEL_PATH)


def load_data(data_path: Path = DATA_PATH) -> tuple[pd.DataFrame, pd.Series]:
    if not data_path.exists():
        raise FileNotFoundError(
            f"Could not find dataset at: {data_path}\n"
            "Make sure you are running from the project root."
        )

    df = pd.read_csv(data_path)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Dataset must contain target column: {TARGET_COL}")

    X = df.drop(columns=DROP_COLS, errors="ignore")
    X = X.select_dtypes(include=[np.number]).copy()
    y = df[TARGET_COL].astype(int)
    return X, y


def align_features(X: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """Make sure simulation data has the exact same features used in training."""
    X_aligned = X.copy()

    for col in feature_names:
        if col not in X_aligned.columns:
            X_aligned[col] = 0.0

    X_aligned = X_aligned[feature_names]
    return X_aligned


def stratified_sample(X: pd.DataFrame, y: pd.Series, sample_size: int = BASE_SAMPLE_SIZE) -> tuple[pd.DataFrame, pd.Series]:
    """Take a small sample but keep enough fraud rows for meaningful evaluation."""
    df = X.copy()
    df[TARGET_COL] = y.values

    fraud_df = df[df[TARGET_COL] == 1]
    legit_df = df[df[TARGET_COL] == 0]

    if len(df) <= sample_size:
        sampled = df.sample(frac=1.0, random_state=RANDOM_STATE)
    else:
        # Keep fraud visible in the simulation sample.
        fraud_n = min(len(fraud_df), max(25, sample_size // 6))
        legit_n = sample_size - fraud_n

        sampled_fraud = fraud_df.sample(n=fraud_n, random_state=RANDOM_STATE, replace=False)
        sampled_legit = legit_df.sample(n=legit_n, random_state=RANDOM_STATE, replace=False)
        sampled = pd.concat([sampled_fraud, sampled_legit], ignore_index=True)
        sampled = sampled.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

    y_sample = sampled[TARGET_COL].astype(int)
    X_sample = sampled.drop(columns=[TARGET_COL])
    return X_sample.reset_index(drop=True), y_sample.reset_index(drop=True)


# -----------------------------------------------------------------------------
# Decision engine
# -----------------------------------------------------------------------------

def make_decision(probability: float, alert_threshold: float = ALERT_THRESHOLD, block_threshold: float = BLOCK_THRESHOLD) -> str:
    """Convert fraud probability to operational decision."""
    if probability >= block_threshold:
        return "BLOCK"
    if probability >= alert_threshold:
        return "REVIEW"
    return "ALLOW"


def binary_prediction(probability: float, block_threshold: float = BLOCK_THRESHOLD) -> int:
    """For metrics: BLOCK threshold means predicted fraud."""
    return int(probability >= block_threshold)


# -----------------------------------------------------------------------------
# Attack simulator: 8 scenarios
# -----------------------------------------------------------------------------

class FraudAttackSimulator:
    def __init__(self, X: pd.DataFrame, y: pd.Series):
        if len(X) != len(y):
            raise ValueError("X and y must have the same length.")
        self.X = X.reset_index(drop=True).copy()
        self.y = y.reset_index(drop=True).copy()
        self.rng = np.random.default_rng(RANDOM_STATE)

    def normal_baseline(self) -> tuple[pd.DataFrame, pd.Series]:
        """1) No attack: original traffic."""
        return self.X.copy(), self.y.copy()

    def random_noise_flood(self, noise_level: float = 0.12) -> tuple[pd.DataFrame, pd.Series]:
        """2) Add small random noise to numeric features."""
        X_attack = self.X.copy()
        numeric_cols = X_attack.select_dtypes(include=[np.number]).columns

        one_hot_cols = [c for c in numeric_cols if c.startswith("type_")]
        numeric_cols = [c for c in numeric_cols if c not in one_hot_cols]

        rng = np.random.default_rng(RANDOM_STATE)
        for col in numeric_cols:
            noise = 1 + rng.normal(0, noise_level, size=len(X_attack))
            X_attack[col] = X_attack[col] * noise

        return X_attack, self.y.copy()

    def micro_transaction_probe(self, factor: float = 0.02) -> tuple[pd.DataFrame, pd.Series]:
        """3) Very small probing transactions."""
        X_attack = self.X.copy()
        if "amount" in X_attack.columns:
            X_attack["amount"] = X_attack["amount"] * factor
        if "amount_log" in X_attack.columns and "amount" in X_attack.columns:
            X_attack["amount_log"] = np.log1p(X_attack["amount"].clip(lower=0))
        return X_attack, self.y.copy()

    def high_value_flood(self, multiplier: float = 12.0) -> tuple[pd.DataFrame, pd.Series]:
        """4) Inflate transaction amount."""
        X_attack = self.X.copy()
        if "amount" in X_attack.columns:
            X_attack["amount"] = X_attack["amount"] * multiplier
        if "amount_log" in X_attack.columns and "amount" in X_attack.columns:
            X_attack["amount_log"] = np.log1p(X_attack["amount"].clip(lower=0))
        if "is_high_amount" in X_attack.columns and "amount" in X_attack.columns:
            threshold = X_attack["amount"].quantile(0.95)
            X_attack["is_high_amount"] = (X_attack["amount"] >= threshold).astype(int)
        return X_attack, self.y.copy()

    def rapid_fire_flood(self, repeats: int = FLOOD_REPEATS) -> tuple[pd.DataFrame, pd.Series]:
        """5) Repeat the same batch many times to simulate transaction flooding."""
        X_attack = pd.concat([self.X.copy()] * repeats, ignore_index=True)
        y_attack = pd.concat([self.y.copy()] * repeats, ignore_index=True)
        return X_attack, y_attack

    def transfer_cashout_attack(self, target_type: str = "TRANSFER") -> tuple[pd.DataFrame, pd.Series]:
        """6) Force high-risk transaction type: TRANSFER or CASH_OUT."""
        X_attack = self.X.copy()
        type_cols = [col for col in X_attack.columns if col.startswith("type_")]

        for col in type_cols:
            X_attack[col] = 0

        target_col = f"type_{target_type}"
        if target_col in X_attack.columns:
            X_attack[target_col] = 1
        elif "type_TRANSFER" in X_attack.columns:
            X_attack["type_TRANSFER"] = 1
        elif "type_CASH_OUT" in X_attack.columns:
            X_attack["type_CASH_OUT"] = 1

        if "high_risk_transaction" in X_attack.columns:
            X_attack["high_risk_transaction"] = 1

        return X_attack, self.y.copy()

    def balance_draining_attack(self, drain_ratio: float = 0.95) -> tuple[pd.DataFrame, pd.Series]:
        """7) Amount becomes close to sender's original balance."""
        X_attack = self.X.copy()

        if "oldbalanceOrg" in X_attack.columns and "amount" in X_attack.columns:
            X_attack["amount"] = X_attack["oldbalanceOrg"].clip(lower=0) * drain_ratio

        if "newbalanceOrig" in X_attack.columns and "oldbalanceOrg" in X_attack.columns and "amount" in X_attack.columns:
            X_attack["newbalanceOrig"] = (X_attack["oldbalanceOrg"] - X_attack["amount"]).clip(lower=0)

        if "balance_diff_orig" in X_attack.columns and "oldbalanceOrg" in X_attack.columns and "newbalanceOrig" in X_attack.columns:
            X_attack["balance_diff_orig"] = X_attack["oldbalanceOrg"] - X_attack["newbalanceOrig"]

        if "amount_log" in X_attack.columns and "amount" in X_attack.columns:
            X_attack["amount_log"] = np.log1p(X_attack["amount"].clip(lower=0))

        if "is_high_amount" in X_attack.columns and "amount" in X_attack.columns:
            threshold = X_attack["amount"].quantile(0.95)
            X_attack["is_high_amount"] = (X_attack["amount"] >= threshold).astype(int)

        return X_attack, self.y.copy()

    def mixed_obfuscation_attack(self) -> tuple[pd.DataFrame, pd.Series]:
        """8) Hardest scenario: combine high value + risky type + balance draining + noise."""
        X_attack = self.X.copy()

        # A) Inflate amount.
        if "amount" in X_attack.columns:
            X_attack["amount"] = X_attack["amount"] * 8

        # B) Force risky transaction type.
        type_cols = [col for col in X_attack.columns if col.startswith("type_")]
        for col in type_cols:
            X_attack[col] = 0
        if "type_TRANSFER" in X_attack.columns:
            X_attack["type_TRANSFER"] = 1
        elif "type_CASH_OUT" in X_attack.columns:
            X_attack["type_CASH_OUT"] = 1

        if "high_risk_transaction" in X_attack.columns:
            X_attack["high_risk_transaction"] = 1

        # C) Make sender balance look heavily drained.
        if "oldbalanceOrg" in X_attack.columns and "newbalanceOrig" in X_attack.columns and "amount" in X_attack.columns:
            X_attack["newbalanceOrig"] = (X_attack["oldbalanceOrg"] - X_attack["amount"]).clip(lower=0)

        if "balance_diff_orig" in X_attack.columns and "oldbalanceOrg" in X_attack.columns and "newbalanceOrig" in X_attack.columns:
            X_attack["balance_diff_orig"] = X_attack["oldbalanceOrg"] - X_attack["newbalanceOrig"]

        if "amount_log" in X_attack.columns and "amount" in X_attack.columns:
            X_attack["amount_log"] = np.log1p(X_attack["amount"].clip(lower=0))

        if "is_high_amount" in X_attack.columns and "amount" in X_attack.columns:
            threshold = X_attack["amount"].quantile(0.95)
            X_attack["is_high_amount"] = (X_attack["amount"] >= threshold).astype(int)

        # D) Add small noise to non-one-hot numeric columns.
        numeric_cols = X_attack.select_dtypes(include=[np.number]).columns
        one_hot_cols = [c for c in numeric_cols if c.startswith("type_")]
        protected_cols = set(one_hot_cols + ["high_risk_transaction", "is_high_amount"])
        numeric_cols = [c for c in numeric_cols if c not in protected_cols]

        rng = np.random.default_rng(RANDOM_STATE)
        for col in numeric_cols:
            noise = 1 + rng.normal(0, 0.05, size=len(X_attack))
            X_attack[col] = X_attack[col] * noise

        return X_attack, self.y.copy()

    def build_attack_suite(
        self,
        flood_repeats: int = FLOOD_REPEATS,
    ) -> Dict[str, tuple[pd.DataFrame, pd.Series]]:
        """Return all 8 scenarios ordered from easiest to hardest."""
        return {
            "01_normal_baseline": self.normal_baseline(),
            "02_random_noise_flood": self.random_noise_flood(noise_level=0.12),
            "03_micro_transaction_probe": self.micro_transaction_probe(factor=0.02),
            "04_high_value_flood": self.high_value_flood(multiplier=12.0),
            "05_rapid_fire_flood": self.rapid_fire_flood(repeats=flood_repeats),
            "06_transfer_cashout_attack": self.transfer_cashout_attack(target_type="TRANSFER"),
            "07_balance_draining_attack": self.balance_draining_attack(drain_ratio=0.95),
            "08_mixed_obfuscation_attack": self.mixed_obfuscation_attack(),
        }


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------

def score_scenario(
    model,
    X_attack: pd.DataFrame,
    y_attack: pd.Series,
    scenario_name: str,
    block_threshold: float = BLOCK_THRESHOLD,
    alert_threshold: float = ALERT_THRESHOLD,
) -> tuple[Dict[str, Any], pd.DataFrame]:
    """Score one scenario and return metrics + detailed rows."""
    start = time.perf_counter()
    probs = model.predict_proba(X_attack)[:, 1]
    elapsed_ms = (time.perf_counter() - start) * 1000

    decisions = [make_decision(p, alert_threshold, block_threshold) for p in probs]
    preds = np.array([binary_prediction(p, block_threshold) for p in probs], dtype=int)

    cm = confusion_matrix(y_attack, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    try:
        auc = roc_auc_score(y_attack, probs)
    except ValueError:
        auc = np.nan

    metrics = {
        "scenario": scenario_name,
        "transactions": int(len(X_attack)),
        "roc_auc": float(auc) if not np.isnan(auc) else np.nan,
        "accuracy": float(accuracy_score(y_attack, preds)),
        "precision": float(precision_score(y_attack, preds, zero_division=0)),
        "recall": float(recall_score(y_attack, preds, zero_division=0)),
        "f1": float(f1_score(y_attack, preds, zero_division=0)),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "allow_count": int(sum(d == "ALLOW" for d in decisions)),
        "review_count": int(sum(d == "REVIEW" for d in decisions)),
        "block_count": int(sum(d == "BLOCK" for d in decisions)),
        "mean_probability": float(np.mean(probs)),
        "max_probability": float(np.max(probs)),
        "scoring_time_ms": float(elapsed_ms),
        "avg_time_per_tx_ms": float(elapsed_ms / max(len(X_attack), 1)),
    }

    detailed = pd.DataFrame(
        {
            "scenario": scenario_name,
            "true_fraud": y_attack.values,
            "pred_fraud": preds,
            "fraud_probability": probs,
            "decision": decisions,
        }
    )

    return metrics, detailed


def print_scenario_report(metrics: Dict[str, Any], y_true: pd.Series, y_pred: np.ndarray) -> None:
    print("\n" + "=" * 80)
    print(metrics["scenario"])
    print("=" * 80)
    print(f"Transactions : {metrics['transactions']}")
    print(f"ROC-AUC      : {metrics['roc_auc']:.6f}" if not np.isnan(metrics["roc_auc"]) else "ROC-AUC      : N/A")
    print(f"Accuracy     : {metrics['accuracy']:.6f}")
    print(f"Precision    : {metrics['precision']:.6f}")
    print(f"Recall       : {metrics['recall']:.6f}")
    print(f"F1           : {metrics['f1']:.6f}")
    print(f"Decisions    : ALLOW={metrics['allow_count']} | REVIEW={metrics['review_count']} | BLOCK={metrics['block_count']}")
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(np.array([[metrics["TN"], metrics["FP"]], [metrics["FN"], metrics["TP"]]]))
    print("Classification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["legitimate", "fraud"],
            zero_division=0,
        )
    )


# -----------------------------------------------------------------------------
# Dashboard / API entry point
# -----------------------------------------------------------------------------

def _paths_for_root(project_root: Path) -> tuple[Path, Path, Path, Path]:
    return (
        project_root / DATA_PATH,
        project_root / MODEL_PATH,
        project_root / FEATURE_NAMES_PATH,
        project_root / OUTPUT_DIR,
    )


def load_feature_names_from(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing feature names file: {path}\n"
            "Run train_choose_simulating_model.py first."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_real_simulation(
    *,
    project_root: Path | None = None,
    block_threshold: float = BLOCK_THRESHOLD,
    alert_threshold: float = ALERT_THRESHOLD,
    flood_repeats: int = FLOOD_REPEATS,
    base_sample_size: int = BASE_SAMPLE_SIZE,
    save_outputs: bool = False,
) -> dict[str, Any]:
    """
    Run all 8 attack scenarios and return a summary dict for CLI or Dash.

    Keys: threshold, alert_threshold, overall, per_attack, flat_results,
          summary_df, detailed_df
    """
    if alert_threshold > block_threshold:
        raise ValueError("alert_threshold must be <= block_threshold")

    root = project_root or Path(__file__).resolve().parent
    data_path, model_path, feature_path, output_dir = _paths_for_root(root)

    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file: {model_path}")

    model = joblib.load(model_path)
    feature_names = load_feature_names_from(feature_path)

    X, y = load_data(data_path)
    X = align_features(X, feature_names)
    X_sample, y_sample = stratified_sample(X, y, sample_size=base_sample_size)

    simulator = FraudAttackSimulator(X_sample, y_sample)
    suite = simulator.build_attack_suite(flood_repeats=flood_repeats)

    all_metrics: list[dict[str, Any]] = []
    all_details: list[pd.DataFrame] = []
    all_y_true: list[int] = []
    all_y_pred: list[int] = []
    per_attack: list[dict[str, Any]] = []
    flat_results: list[dict[str, Any]] = []

    for scenario_name, (X_attack, y_attack) in suite.items():
        X_attack = align_features(X_attack, feature_names)
        metrics, detailed = score_scenario(
            model,
            X_attack,
            y_attack,
            scenario_name=scenario_name,
            block_threshold=block_threshold,
            alert_threshold=alert_threshold,
        )

        cm = np.array(
            [[metrics["TN"], metrics["FP"]], [metrics["FN"], metrics["TP"]]],
            dtype=int,
        )
        description = SCENARIO_DESCRIPTIONS.get(
            scenario_name,
            f"Scenario: {scenario_name}",
        )

        per_attack.append(
            {
                "attack_name": scenario_name,
                "attack_description": description,
                "confusion_matrix": cm,
                "counts": {
                    "TN": metrics["TN"],
                    "FP": metrics["FP"],
                    "FN": metrics["FN"],
                    "TP": metrics["TP"],
                },
                "metrics": metrics,
            }
        )

        for _, row in detailed.iterrows():
            prob = float(row["fraud_probability"])
            decision = str(row["decision"])
            flat_results.append(
                {
                    "attack_type": scenario_name,
                    "probability": prob,
                    "decision": decision,
                    "true_fraud": int(row["true_fraud"]),
                    "pred_fraud": int(row["pred_fraud"]),
                    "explanation": {
                        "attack_analysis": description,
                        "prediction_summary": (
                            f"P(fraud)={prob:.4f} → {decision} "
                            f"(block≥{block_threshold}, review≥{alert_threshold})"
                        ),
                        "truth_analysis": None,
                        "feature_evidence": [],
                    },
                }
            )

        all_metrics.append(metrics)
        all_details.append(detailed)
        all_y_true.extend(y_attack.astype(int).tolist())
        all_y_pred.extend(detailed["pred_fraud"].astype(int).tolist())

    overall_cm = confusion_matrix(all_y_true, all_y_pred, labels=[0, 1])
    tn, fp, fn, tp = overall_cm.ravel()

    latencies = [m["avg_time_per_tx_ms"] for m in all_metrics]
    weights = [m["transactions"] for m in all_metrics]
    total_w = max(sum(weights), 1)
    weighted_avg_latency = sum(l * w for l, w in zip(latencies, weights)) / total_w

    summary_df = pd.DataFrame(all_metrics)
    detailed_df = pd.concat(all_details, ignore_index=True)

    if save_outputs:
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(output_dir / "simulation_summary.csv", index=False)
        detailed_df.to_csv(output_dir / "simulation_detailed_results.csv", index=False)

    return {
        "threshold": block_threshold,
        "alert_threshold": alert_threshold,
        "flood_repeats": flood_repeats,
        "base_sample_size": len(X_sample),
        "scenario_count": len(suite),
        "per_attack": per_attack,
        "flat_results": flat_results,
        "overall": {
            "confusion_matrix": overall_cm,
            "counts": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
            "total_transactions": len(all_y_true),
            "avg_latency_ms": float(weighted_avg_latency),
        },
        "summary_df": summary_df,
        "detailed_df": detailed_df,
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    print("Loading trained model and running 8-scenario simulation...")
    result = run_real_simulation(
        block_threshold=BLOCK_THRESHOLD,
        alert_threshold=ALERT_THRESHOLD,
        flood_repeats=FLOOD_REPEATS,
        base_sample_size=BASE_SAMPLE_SIZE,
        save_outputs=True,
    )

    print(f"Base sample size: {result['base_sample_size']}")
    print("\nRunning 8 simulation scenarios...")
    print(
        f"BLOCK_THRESHOLD={result['threshold']}, "
        f"ALERT_THRESHOLD={result['alert_threshold']}"
    )

    for attack in result["per_attack"]:
        m = attack["metrics"]
        y_true = result["detailed_df"].loc[
            result["detailed_df"]["scenario"] == attack["attack_name"], "true_fraud"
        ]
        y_pred = result["detailed_df"].loc[
            result["detailed_df"]["scenario"] == attack["attack_name"], "pred_fraud"
        ]
        print_scenario_report(m, y_true, y_pred.values)

    print("\n" + "#" * 80)
    print("TOTAL SIMULATION RESULTS")
    print("#" * 80)

    overall = result["overall"]
    cm = overall["confusion_matrix"]
    c = overall["counts"]
    print(f"Total transactions: {overall['total_transactions']}")
    print("Overall confusion matrix [[TN, FP], [FN, TP]]:")
    print(cm)
    print(f"TN={c['TN']}, FP={c['FP']}, FN={c['FN']}, TP={c['TP']}")
    print("Overall classification report:")
    print(
        classification_report(
            result["detailed_df"]["true_fraud"],
            result["detailed_df"]["pred_fraud"],
            labels=[0, 1],
            target_names=["legitimate", "fraud"],
            zero_division=0,
        )
    )
    print(
        f"Weighted avg scoring latency: {overall['avg_latency_ms']:.2f} ms / transaction"
    )
    print("Saved simulation outputs:")
    print(f"- {OUTPUT_DIR / 'simulation_summary.csv'}")
    print(f"- {OUTPUT_DIR / 'simulation_detailed_results.csv'}")


if __name__ == "__main__":
    main()
