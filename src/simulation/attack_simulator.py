from __future__ import annotations

import numpy as np
import pandas as pd


class FraudAttackSimulator:
    """Generate flooded attack streams while preserving row-aligned fraud labels."""

    def __init__(self, X: pd.DataFrame, y_true: pd.Series):
        self.X = X.copy().reset_index(drop=True)
        self.y_true = y_true.copy().reset_index(drop=True)
        if len(self.X) != len(self.y_true):
            raise ValueError("X and y_true must have the same length.")

    def high_value_attack(self, multiplier: float = 10.0) -> tuple[pd.DataFrame, pd.Series]:
        attack = self.X.copy()
        attack["amount"] = attack["amount"] * multiplier
        return attack, self.y_true.copy()

    def rapid_fire_attack(self, repeats: int = 50) -> tuple[pd.DataFrame, pd.Series]:
        """Repeat the base batch many times to simulate a transaction flood."""
        repeats = max(1, int(repeats))
        attack = pd.concat([self.X] * repeats, ignore_index=True)
        y_rep = pd.concat([self.y_true] * repeats, ignore_index=True)
        return attack, y_rep

    def random_noise_attack(self, noise_level: float = 0.1) -> tuple[pd.DataFrame, pd.Series]:
        attack = self.X.copy()
        rng = np.random.default_rng(42)
        cols = attack.select_dtypes(include=np.number).columns
        for col in cols:
            noise = rng.normal(0.0, noise_level, size=len(attack))
            attack[col] = attack[col] * (1 + noise)
        return attack, self.y_true.copy()

    def build_attack_suite(
        self,
        *,
        flood_repeats: int = 40,
        noise_level: float = 0.12,
        high_value_multiplier: float = 12.0,
    ) -> list[tuple[str, pd.DataFrame, pd.Series]]:
        """Ordered list of (scenario_name, features, y_true) including heavy floods."""
        suite: list[tuple[str, pd.DataFrame, pd.Series]] = []

        suite.append(("normal_baseline", self.X.copy(), self.y_true.copy()))

        hv_x, hv_y = self.high_value_attack(multiplier=high_value_multiplier)
        suite.append(("high_value_flood", hv_x, hv_y))

        rf_x, rf_y = self.rapid_fire_attack(repeats=flood_repeats)
        suite.append(("rapid_fire_flood", rf_x, rf_y))

        nn_x, nn_y = self.random_noise_attack(noise_level=noise_level)
        suite.append(("random_noise_flood", nn_x, nn_y))

        return suite
