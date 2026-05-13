from __future__ import annotations

from typing import Any

import numpy as np


class FraudExplainer:

    def explain_tree_model(
        self,
        model: Any,
        feature_names: Any,
        sample: np.ndarray,
    ) -> list[dict[str, Any]]:
        names = list(feature_names)

        if hasattr(model, "feature_importances_"):
            importances = np.asarray(model.feature_importances_, dtype=float)
        elif hasattr(model, "coef_"):
            importances = np.asarray(np.abs(model.coef_)).ravel()
            if importances.size != len(names):
                importances = np.pad(
                    importances,
                    (0, max(0, len(names) - importances.size)),
                    constant_values=0.0,
                )[: len(names)]
        else:
            return [{"feature": "model", "importance": 0.0, "note": "no feature weights"}]

        top_indices = np.argsort(importances)[::-1][: min(5, len(importances))]
        explanation: list[dict[str, Any]] = []
        sample_flat = np.asarray(sample).ravel()

        for i in top_indices:
            explanation.append(
                {
                    "feature": names[int(i)] if int(i) < len(names) else str(i),
                    "importance": float(importances[int(i)]),
                    "value": float(sample_flat[int(i)]) if int(i) < sample_flat.size else None,
                }
            )

        return explanation
