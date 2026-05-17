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

    def build_attack_suite(self) -> Dict[str, tuple[pd.DataFrame, pd.Series]]:
        """Return all 8 scenarios ordered from easiest to hardest."""
        return {
            "01_normal_baseline": self.normal_baseline(),
            "02_random_noise_flood": self.random_noise_flood(noise_level=0.12),
            "03_micro_transaction_probe": self.micro_transaction_probe(factor=0.02),
            "04_high_value_flood": self.high_value_flood(multiplier=12.0),
            "05_rapid_fire_flood": self.rapid_fire_flood(repeats=FLOOD_REPEATS),
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
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    print("Loading trained model...")
    model = load_model()
    feature_names = load_feature_names()

    print("Loading data...")
    X, y = load_data(DATA_PATH)
    X = align_features(X, feature_names)

    print("Creating stratified base sample...")
    X_sample, y_sample = stratified_sample(X, y, sample_size=BASE_SAMPLE_SIZE)
    print(f"Base sample size: {len(X_sample)}")
    print("Base class distribution:")
    print(y_sample.value_counts().rename({0: "legitimate", 1: "fraud"}))

    simulator = FraudAttackSimulator(X_sample, y_sample)
    suite = simulator.build_attack_suite()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_metrics = []
    all_details = []
    all_y_true = []
    all_y_pred = []

    print("\nRunning 8 simulation scenarios...")
    print(f"BLOCK_THRESHOLD={BLOCK_THRESHOLD}, ALERT_THRESHOLD={ALERT_THRESHOLD}")

    for scenario_name, (X_attack, y_attack) in suite.items():
        X_attack = align_features(X_attack, feature_names)
        metrics, detailed = score_scenario(
            model,
            X_attack,
            y_attack,
            scenario_name=scenario_name,
            block_threshold=BLOCK_THRESHOLD,
            alert_threshold=ALERT_THRESHOLD,
        )

        y_pred = detailed["pred_fraud"].values
        print_scenario_report(metrics, y_attack, y_pred)

        all_metrics.append(metrics)
        all_details.append(detailed)
        all_y_true.extend(y_attack.values.tolist())
        all_y_pred.extend(y_pred.tolist())

    summary_df = pd.DataFrame(all_metrics)
    detailed_df = pd.concat(all_details, ignore_index=True)

    summary_path = OUTPUT_DIR / "simulation_summary.csv"
    detailed_path = OUTPUT_DIR / "simulation_detailed_results.csv"

    summary_df.to_csv(summary_path, index=False)
    detailed_df.to_csv(detailed_path, index=False)

    print("\n" + "#" * 80)
    print("TOTAL SIMULATION RESULTS")
    print("#" * 80)

    total_cm = confusion_matrix(all_y_true, all_y_pred, labels=[0, 1])
    tn, fp, fn, tp = total_cm.ravel()

    print(f"Total transactions: {len(all_y_true)}")
    print("Overall confusion matrix [[TN, FP], [FN, TP]]:")
    print(total_cm)
    print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print("Overall classification report:")
    print(
        classification_report(
            all_y_true,
            all_y_pred,
            labels=[0, 1],
            target_names=["legitimate", "fraud"],
            zero_division=0,
        )
    )

    print("Saved simulation outputs:")
    print(f"- {summary_path}")
    print(f"- {detailed_path}")


if __name__ == "__main__":
    main()
