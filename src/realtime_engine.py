"""
Real-time fraud scoring pipeline:

    Incoming transaction → model prediction → decision engine → explainer → log / block / alert

Use from project root so paths like ``output/models/best_model.pkl`` resolve correctly.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

import joblib
import pandas as pd

from src.descision.decision_engine import FraudDecisionEngine
from src.explain.explainer import FraudExplainer


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _align_features(row_df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(0.0, index=row_df.index, columns=feature_names)
    for col in feature_names:
        if col in row_df.columns:
            out[col] = row_df[col]
    return out


@dataclass
class RealtimeOutcome:
    """Single-transaction result from the real-time engine."""

    transaction_id: str | None
    fraud_probability: float
    decision: str  # ALLOW | REVIEW | BLOCK
    action: str  # LOG | ALERT | BLOCK
    explanation: dict[str, Any] | None
    latency_ms: float
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "transaction_id": self.transaction_id,
            "fraud_probability": self.fraud_probability,
            "decision": self.decision,
            "action": self.action,
            "latency_ms": self.latency_ms,
            "explanation": self.explanation,
            **self.raw,
        }


class RealtimeFraudEngine:
    """
    Streams transactions through: predict → decide → explain (when needed) → log / alert / block.
    """

    def __init__(
        self,
        *,
        block_threshold: float = 0.3,
        alert_threshold: float = 0.12,
        model_path: Path | None = None,
        meta_path: Path | None = None,
        scaler_path: Path | None = None,
        alert_hook: Callable[[RealtimeOutcome], None] | None = None,
        block_hook: Callable[[RealtimeOutcome], None] | None = None,
        enable_logging: bool = True,
        skip_explanations: bool = False,
    ):
        root = _project_root()
        mp = model_path or root / "output" / "models" / "best_model.pkl"
        meta_p = meta_path or root / "output" / "models" / "best_model_meta.json"
        sc_p = scaler_path or root / "output" / "models" / "best_model_scaler.pkl"

        self.model = joblib.load(mp)
        self.scaler: Any | None = None
        if meta_p.exists():
            meta = json.loads(meta_p.read_text(encoding="utf-8"))
            if meta.get("has_scaler") and sc_p.exists():
                self.scaler = joblib.load(sc_p)

        if hasattr(self.model, "feature_names_in_"):
            self.feature_names = [str(c) for c in self.model.feature_names_in_]
        else:
            ref = root / "data" / "processed" / "processed_paysim.csv"
            if not ref.exists():
                raise AttributeError(
                    "Model has no feature_names_in_ and processed_paysim.csv not found "
                    f"at {ref}; cannot infer feature order."
                )
            df_head = pd.read_csv(ref, nrows=1)
            drop_cols = {"isFraud", "isFlaggedFraud"}
            self.feature_names = [c for c in df_head.columns if c not in drop_cols]

        self.block_threshold = float(block_threshold)
        self.alert_threshold = float(alert_threshold)
        if self.alert_threshold > self.block_threshold:
            raise ValueError("alert_threshold must be <= block_threshold")

        self.decision_engine = FraudDecisionEngine(threshold=self.block_threshold)
        self.explainer = FraudExplainer()
        self.alert_hook = alert_hook
        self.block_hook = block_hook
        self.enable_logging = enable_logging
        self.skip_explanations = skip_explanations

        self._log = logging.getLogger("realtime_fraud")

    def _predict_proba_row(self, X: pd.DataFrame) -> float:
        Xa = _align_features(X, self.feature_names)
        if self.scaler is not None:
            vec = self.scaler.transform(Xa.values)
            return float(self.model.predict_proba(vec)[0, 1])
        return float(self.model.predict_proba(Xa)[0, 1])

    def _tier(self, prob: float) -> tuple[str, str]:
        """
        Map probability to operational decision + action channel.

        - ALLOW + LOG: routine traffic
        - REVIEW + ALERT: suspicious; escalate / notify monitoring
        - BLOCK + BLOCK: reject transaction and alert
        """
        if prob >= self.block_threshold:
            return "BLOCK", "BLOCK"
        if prob >= self.alert_threshold:
            return "REVIEW", "ALERT"
        return "ALLOW", "LOG"

    def process_transaction(
        self,
        transaction: dict[str, Any] | pd.Series,
        *,
        transaction_id: str | None = None,
    ) -> RealtimeOutcome:
        t0 = time.perf_counter()

        if isinstance(transaction, pd.Series):
            row = transaction.to_frame().T
            tid = transaction_id or transaction.get("transaction_id")  # type: ignore[union-attr]
        else:
            tid = transaction_id or transaction.get("transaction_id")
            row = pd.DataFrame([transaction])

        prob = self._predict_proba_row(row)
        decision, action = self._tier(prob)

        explanation: dict[str, Any] | None = None
        if not self.skip_explanations:
            explanation = self.explainer.explain_fraud_prediction(
                self.model,
                self.feature_names,
                row.iloc[0].reindex(self.feature_names).fillna(0).values,
                decision=decision,
                fraud_probability=prob,
                block_threshold=self.block_threshold,
                alert_threshold=self.alert_threshold,
                attack_type=None,
                true_fraud_label=None,
                pred_fraud_label=None,
                include_feature_evidence=(decision != "ALLOW"),
            )

        latency_ms = (time.perf_counter() - t0) * 1000.0

        outcome = RealtimeOutcome(
            transaction_id=str(tid) if tid is not None else None,
            fraud_probability=prob,
            decision=decision,
            action=action,
            explanation=explanation,
            latency_ms=latency_ms,
            raw={},
        )

        self._dispatch_logs(outcome)
        return outcome

    def predicted_fraud_label(self, fraud_probability: float) -> int:
        """Binary fraud type from the same rule as the decision engine (1=fraud, 0=legitimate)."""
        return self.decision_engine.fraud_prediction_label(fraud_probability)

    def _dispatch_logs(self, outcome: RealtimeOutcome) -> None:
        if self.enable_logging:
            if outcome.action == "LOG":
                self._log.info(
                    "[ALLOW] id=%s p=%.4f %.2fms",
                    outcome.transaction_id,
                    outcome.fraud_probability,
                    outcome.latency_ms,
                )
            elif outcome.action == "ALERT":
                self._log.warning(
                    "[ALERT/REVIEW] id=%s p=%.4f %.2fms",
                    outcome.transaction_id,
                    outcome.fraud_probability,
                    outcome.latency_ms,
                )
            else:
                self._log.error(
                    "[BLOCK] id=%s p=%.4f %.2fms explain=%s",
                    outcome.transaction_id,
                    outcome.fraud_probability,
                    outcome.latency_ms,
                    outcome.explanation,
                )

        if outcome.action == "ALERT":
            if self.alert_hook:
                self.alert_hook(outcome)
        elif outcome.action == "BLOCK":
            if self.block_hook:
                self.block_hook(outcome)
            if self.alert_hook:
                self.alert_hook(outcome)

    def process_stream(self, transactions: Iterable[dict[str, Any] | pd.Series]) -> Iterator[RealtimeOutcome]:
        """Push an iterable of transactions through the pipeline (sync generator)."""
        for i, tx in enumerate(transactions):
            tid = None
            if isinstance(tx, dict):
                tid = tx.get("transaction_id")
            yield self.process_transaction(tx, transaction_id=tid or f"tx-{i}")


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _demo_stream_from_processed_csv(n: int = 30) -> list[dict[str, Any]]:
    root = _project_root()
    path = root / "data" / "processed" / "processed_paysim.csv"
    df = pd.read_csv(path)
    sample = df.drop(columns=["isFraud", "isFlaggedFraud"]).sample(min(n, len(df)), random_state=7)
    rows: list[dict[str, Any]] = []
    for idx, row in sample.iterrows():
        d = row.to_dict()
        d["transaction_id"] = f"paysim-{idx}"
        rows.append(d)
    return rows


def main() -> None:
    setup_logging()
    engine = RealtimeFraudEngine(block_threshold=0.3, alert_threshold=0.12)

    print("Real-time demo: draining a batch iterator through the engine...\n")
    for outcome in engine.process_stream(_demo_stream_from_processed_csv(n=25)):
        print(
            f"{outcome.transaction_id}: p={outcome.fraud_probability:.4f} "
            f"decision={outcome.decision} action={outcome.action} ({outcome.latency_ms:.2f} ms)"
        )


if __name__ == "__main__":
    main()
