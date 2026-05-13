from __future__ import annotations


class FraudDecisionEngine:
    """Map fraud probabilities to operational decisions and binary fraud labels."""

    def __init__(self, threshold: float = 0.3):
        self.threshold = float(threshold)

    def decide(self, probability: float) -> str:
        if probability >= self.threshold:
            return "BLOCK"
        return "ALLOW"

    def fraud_prediction_label(self, probability: float) -> int:
        """1 = predicted fraud (BLOCK), 0 = predicted legitimate (ALLOW)."""
        return 1 if probability >= self.threshold else 0
