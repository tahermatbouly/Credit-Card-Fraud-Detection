"""
Real-Time Fraud Detection Dashboard with Attack Picker

What changed:
- You choose ONE attack scenario from the GUI.
- The dashboard processes transactions row-by-row like real-time streaming.
- Decisions are ALLOW / REVIEW / BLOCK.
- You can also choose ALL scenarios if you want a full run.

Run:
    python dash_app.py

Put this file in the project root and rename it to dash_app.py
if you want it to replace the old dashboard.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from dash import Dash, Input, Output, State, callback, dcc, html
from dash.dash_table import DataTable

from fraud_probability_audit import audit_predictions, save_audit_logs


# =========================================================
# Paths
# =========================================================


def _project_root() -> Path:
    return Path(__file__).resolve().parent


# =========================================================
# Attack descriptions
# =========================================================


ATTACK_OPTIONS = [
    {"label": "01 - Normal Baseline", "value": "01_normal_baseline"},
    {"label": "02 - Random Noise Flood", "value": "02_random_noise_flood"},
    {"label": "03 - Micro Transaction Probe", "value": "03_micro_transaction_probe"},
    {"label": "04 - High Value Flood", "value": "04_high_value_flood"},
    {"label": "05 - Rapid Fire Flood", "value": "05_rapid_fire_flood"},
    {"label": "06 - Transfer / Cash-out Attack", "value": "06_transfer_cashout_attack"},
    {"label": "07 - Balance Draining Attack", "value": "07_balance_draining_attack"},
    {"label": "08 - Mixed Obfuscation Attack", "value": "08_mixed_obfuscation_attack"},
    {"label": "Run ALL 8 Scenarios", "value": "all"},
]


ATTACK_DESCRIPTIONS: Dict[str, str] = {
    "01_normal_baseline": (
        "Normal Baseline: no attack is applied. Transactions are streamed to the model "
        "exactly as they appear in the sampled data. This is the reference case."
    ),
    "02_random_noise_flood": (
        "Random Noise Flood: small random changes are added to numeric features. "
        "This tests if the model is stable when transaction data is slightly manipulated or noisy."
    ),
    "03_micro_transaction_probe": (
        "Micro Transaction Probe: transaction amounts are reduced to very small values. "
        "This simulates an attacker testing if an account/card/payment channel is active before a bigger attack."
    ),
    "04_high_value_flood": (
        "High Value Flood: transaction amounts are multiplied by a large number. "
        "This simulates sudden high-value payments or transfers."
    ),
    "05_rapid_fire_flood": (
        "Rapid Fire Flood: the same batch is repeated many times. "
        "This simulates automated high-volume transaction traffic."
    ),
    "06_transfer_cashout_attack": (
        "Transfer / Cash-out Attack: transaction type columns are forced to a high-risk type "
        "such as TRANSFER or CASH_OUT."
    ),
    "07_balance_draining_attack": (
        "Balance Draining Attack: the transaction amount becomes close to the original sender balance, "
        "leaving the account almost empty. This simulates account takeover or draining behavior."
    ),
    "08_mixed_obfuscation_attack": (
        "Mixed Obfuscation Attack: combines multiple suspicious behaviors together: high amount, "
        "risky transaction type, sender balance manipulation, and small random noise. "
        "This is the hardest scenario."
    ),
}


# =========================================================
# Load Model + Metadata
# =========================================================


def _load_model_bundle(root: Path):
    model_path = root / "output" / "models" / "best_model.pkl"
    meta_path = root / "output" / "models" / "best_model_meta.json"
    scaler_path = root / "output" / "models" / "best_model_scaler.pkl"
    feature_names_path = root / "output" / "models" / "feature_names.json"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            "Run training first: python train_choose_simulating_model.py"
        )

    model = joblib.load(model_path)
    scaler = None

    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("has_scaler") and scaler_path.exists():
            scaler = joblib.load(scaler_path)

    if hasattr(model, "feature_names_in_"):
        feature_names = [str(c) for c in model.feature_names_in_]
    elif feature_names_path.exists():
        feature_names = json.loads(feature_names_path.read_text(encoding="utf-8"))
    else:
        data_path = _find_data_path(root)
        head = pd.read_csv(data_path, nrows=1)
        feature_names = [
            c for c in head.columns if c not in {"isFraud", "isFlaggedFraud"}
        ]

    return model, scaler, feature_names


# =========================================================
# Data Loading + Sampling
# =========================================================


def _find_data_path(root: Path) -> Path:
    processed_path = root / "data" / "processed" / "processed_paysim.csv"
    root_paysim_path = root / "paysim.csv"

    if processed_path.exists():
        return processed_path
    if root_paysim_path.exists():
        return root_paysim_path

    raise FileNotFoundError(
        "Could not find dataset. Expected one of:\n"
        f"- {processed_path}\n"
        f"- {root_paysim_path}"
    )


def _prepare_features(df: pd.DataFrame, feature_names: List[str]) -> Tuple[pd.DataFrame, pd.Series]:
    if "isFraud" not in df.columns:
        raise ValueError("Dataset must contain isFraud column.")

    y = df["isFraud"].astype(int).reset_index(drop=True)

    X_raw = df.drop(columns=["isFraud", "isFlaggedFraud"], errors="ignore").copy()

    # If raw PaySim is used and `type` is still categorical, one-hot encode it.
    if "type" in X_raw.columns:
        type_dummies = pd.get_dummies(X_raw["type"], prefix="type")
        X_raw = pd.concat([X_raw.drop(columns=["type"]), type_dummies], axis=1)

    # Drop non-numeric columns such as nameOrig/nameDest if present.
    for col in list(X_raw.columns):
        if not pd.api.types.is_numeric_dtype(X_raw[col]):
            X_raw = X_raw.drop(columns=[col])

    # Align exactly to training feature order.
    X = X_raw.reindex(columns=feature_names, fill_value=0.0)
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    return X.reset_index(drop=True), y


def _sample_base_transactions(
    data_path: Path,
    feature_names: List[str],
    *,
    max_base_sample: int = 300,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(data_path)
    sample_n = min(max_base_sample, len(df))

    fraud_df = df[df["isFraud"] == 1]
    legit_df = df[df["isFraud"] == 0]

    if len(fraud_df) > 0 and len(legit_df) > 0:
        fraud_take = min(len(fraud_df), max(20, sample_n // 6))
        legit_take = min(len(legit_df), max(sample_n - fraud_take, 1))

        sample_df = pd.concat(
            [
                fraud_df.sample(fraud_take, random_state=random_state),
                legit_df.sample(legit_take, random_state=random_state),
            ],
            ignore_index=True,
        ).sample(frac=1.0, random_state=random_state)
    else:
        sample_df = df.sample(sample_n, random_state=random_state)

    return _prepare_features(sample_df, feature_names)


# =========================================================
# Attack Builder
# =========================================================


def _apply_attack(
    attack_type: str,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    flood_repeats: int,
    noise_level: float = 0.12,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.Series]:
    X_attack = X.copy().reset_index(drop=True)
    y_attack = y.copy().reset_index(drop=True)
    rng = np.random.default_rng(random_state)

    if attack_type == "01_normal_baseline":
        return X_attack, y_attack

    if attack_type == "02_random_noise_flood":
        numeric_cols = X_attack.select_dtypes(include=[np.number]).columns
        type_cols = [c for c in numeric_cols if c.startswith("type_")]
        for col in numeric_cols:
            if col not in type_cols:
                noise = 1 + rng.normal(0, noise_level, size=len(X_attack))
                X_attack[col] = X_attack[col] * noise
        return X_attack, y_attack

    if attack_type == "03_micro_transaction_probe":
        if "amount" in X_attack.columns:
            X_attack["amount"] = X_attack["amount"] * 0.02
        if "amount_log" in X_attack.columns:
            X_attack["amount_log"] = np.log1p(np.maximum(X_attack.get("amount", 0), 0))
        return X_attack, y_attack

    if attack_type == "04_high_value_flood":
        if "amount" in X_attack.columns:
            X_attack["amount"] = X_attack["amount"] * 12
        if "is_high_amount" in X_attack.columns:
            X_attack["is_high_amount"] = 1
        if "amount_log" in X_attack.columns:
            X_attack["amount_log"] = np.log1p(np.maximum(X_attack.get("amount", 0), 0))
        return X_attack, y_attack

    if attack_type == "05_rapid_fire_flood":
        repeats = max(int(flood_repeats), 1)
        X_attack = pd.concat([X_attack] * repeats, ignore_index=True)
        y_attack = pd.concat([y_attack] * repeats, ignore_index=True)
        return X_attack, y_attack

    if attack_type == "06_transfer_cashout_attack":
        type_cols = [c for c in X_attack.columns if c.startswith("type_")]
        for col in type_cols:
            X_attack[col] = 0

        if "type_TRANSFER" in X_attack.columns:
            X_attack["type_TRANSFER"] = 1
        elif "type_CASH_OUT" in X_attack.columns:
            X_attack["type_CASH_OUT"] = 1

        if "high_risk_transaction" in X_attack.columns:
            X_attack["high_risk_transaction"] = 1

        return X_attack, y_attack

    if attack_type == "07_balance_draining_attack":
        if "oldbalanceOrg" in X_attack.columns and "amount" in X_attack.columns:
            X_attack["amount"] = X_attack["oldbalanceOrg"] * 0.95
        if "oldbalanceOrg" in X_attack.columns and "newbalanceOrig" in X_attack.columns:
            X_attack["newbalanceOrig"] = X_attack["oldbalanceOrg"] - X_attack.get("amount", 0)
            X_attack["newbalanceOrig"] = X_attack["newbalanceOrig"].clip(lower=0)
        if "balance_diff_orig" in X_attack.columns:
            X_attack["balance_diff_orig"] = X_attack.get("oldbalanceOrg", 0) - X_attack.get("newbalanceOrig", 0)
        if "amount_log" in X_attack.columns:
            X_attack["amount_log"] = np.log1p(np.maximum(X_attack.get("amount", 0), 0))
        return X_attack, y_attack

    if attack_type == "08_mixed_obfuscation_attack":
        if "amount" in X_attack.columns:
            X_attack["amount"] = X_attack["amount"] * 8

        type_cols = [c for c in X_attack.columns if c.startswith("type_")]
        for col in type_cols:
            X_attack[col] = 0
        if "type_TRANSFER" in X_attack.columns:
            X_attack["type_TRANSFER"] = 1
        elif "type_CASH_OUT" in X_attack.columns:
            X_attack["type_CASH_OUT"] = 1

        if "high_risk_transaction" in X_attack.columns:
            X_attack["high_risk_transaction"] = 1
        if "is_high_amount" in X_attack.columns:
            X_attack["is_high_amount"] = 1

        if "oldbalanceOrg" in X_attack.columns and "newbalanceOrig" in X_attack.columns:
            X_attack["newbalanceOrig"] = X_attack["oldbalanceOrg"] - X_attack.get("amount", 0)
            X_attack["newbalanceOrig"] = X_attack["newbalanceOrig"].clip(lower=0)
        if "balance_diff_orig" in X_attack.columns:
            X_attack["balance_diff_orig"] = X_attack.get("oldbalanceOrg", 0) - X_attack.get("newbalanceOrig", 0)
        if "amount_log" in X_attack.columns:
            X_attack["amount_log"] = np.log1p(np.maximum(X_attack.get("amount", 0), 0))

        numeric_cols = X_attack.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if not col.startswith("type_"):
                noise = 1 + rng.normal(0, 0.05, size=len(X_attack))
                X_attack[col] = X_attack[col] * noise

        return X_attack, y_attack

    raise ValueError(f"Unknown attack type: {attack_type}")


# =========================================================
# Real-time Scoring
# =========================================================


def _predict_probability(model, scaler, row_df: pd.DataFrame) -> float:
    if scaler is not None:
        arr = scaler.transform(row_df.values)
        return float(model.predict_proba(arr)[:, 1][0])
    return float(model.predict_proba(row_df)[:, 1][0])


def _decision(probability: float, *, alert_threshold: float, block_threshold: float) -> str:
    if probability >= block_threshold:
        return "BLOCK"
    if probability >= alert_threshold:
        return "REVIEW"
    return "ALLOW"


def _confusion_counts(y_true: List[int], y_pred: List[int]) -> Tuple[np.ndarray, Dict[str, int]]:
    cm = np.zeros((2, 2), dtype=int)
    for actual, pred in zip(y_true, y_pred):
        actual_i = int(actual)
        pred_i = int(pred)
        if actual_i in (0, 1) and pred_i in (0, 1):
            cm[actual_i, pred_i] += 1

    counts = {
        "TN": int(cm[0, 0]),
        "FP": int(cm[0, 1]),
        "FN": int(cm[1, 0]),
        "TP": int(cm[1, 1]),
    }
    return cm, counts


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def _run_realtime_stream(
    model,
    scaler,
    feature_names: List[str],
    X_attack: pd.DataFrame,
    y_attack: pd.Series,
    *,
    attack_type: str,
    alert_threshold: float,
    block_threshold: float,
    max_rows_to_process: int = 5000,
) -> Dict:
    X_attack = X_attack.reindex(columns=feature_names, fill_value=0.0)
    X_attack = X_attack.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y_attack = y_attack.reset_index(drop=True)

    if len(X_attack) > max_rows_to_process:
        X_attack = X_attack.iloc[:max_rows_to_process].reset_index(drop=True)
        y_attack = y_attack.iloc[:max_rows_to_process].reset_index(drop=True)

    rows = []
    y_true_all: List[int] = []
    y_pred_all: List[int] = []
    probabilities: List[float] = []
    latencies: List[float] = []

    for i in range(len(X_attack)):
        row_df = X_attack.iloc[[i]]
        actual = int(y_attack.iloc[i])

        started = time.perf_counter()
        probability = _predict_probability(model, scaler, row_df)
        latency_ms = (time.perf_counter() - started) * 1000

        decision = _decision(
            probability,
            alert_threshold=alert_threshold,
            block_threshold=block_threshold,
        )
        pred = 1 if probability >= block_threshold else 0

        y_true_all.append(actual)
        y_pred_all.append(pred)
        probabilities.append(probability)
        latencies.append(latency_ms)

        rows.append(
            {
                "tx_id": f"{attack_type}-TX-{i}",
                "attack": attack_type,
                "probability": probability,
                "decision": decision,
                "action": "BLOCK" if decision == "BLOCK" else ("ALERT" if decision == "REVIEW" else "LOG"),
                "true_fraud": actual,
                "pred_fraud": pred,
                "latency_ms": latency_ms,
                "amount": float(row_df["amount"].iloc[0]) if "amount" in row_df.columns else None,
            }
        )

    cm, counts = _confusion_counts(y_true_all, y_pred_all)

    total = len(y_true_all)
    accuracy = _safe_div(counts["TP"] + counts["TN"], total)
    precision = _safe_div(counts["TP"], counts["TP"] + counts["FP"])
    recall = _safe_div(counts["TP"], counts["TP"] + counts["FN"])
    f1 = _safe_div(2 * precision * recall, precision + recall)

    decision_counts = pd.Series([r["decision"] for r in rows]).value_counts().to_dict()

    return {
        "attack_name": attack_type,
        "attack_description": ATTACK_DESCRIPTIONS.get(attack_type, attack_type),
        "rows": rows,
        "y_true": y_true_all,
        "y_pred": y_pred_all,
        "probabilities": probabilities,
        "latencies_ms": latencies,
        "confusion_matrix": cm,
        "counts": counts,
        "decision_counts": {
            "ALLOW": int(decision_counts.get("ALLOW", 0)),
            "REVIEW": int(decision_counts.get("REVIEW", 0)),
            "BLOCK": int(decision_counts.get("BLOCK", 0)),
        },
        "metrics": {
            "total_transactions": total,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "avg_probability": float(np.mean(probabilities)) if probabilities else 0.0,
            "max_probability": float(np.max(probabilities)) if probabilities else 0.0,
            "avg_latency_ms": float(np.mean(latencies)) if latencies else 0.0,
            "p95_latency_ms": float(np.percentile(latencies, 95)) if latencies else 0.0,
            "max_latency_ms": float(np.max(latencies)) if latencies else 0.0,
        },
    }


# =========================================================
# Figures
# =========================================================


def _confusion_figure(cm: np.ndarray):
    z = cm.astype(int).tolist()
    labels = ["Legitimate", "Fraud"]

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=labels,
            y=labels,
            colorscale="Blues",
            text=z,
            texttemplate="%{text}",
            hovertemplate=(
                "Actual %{y}<br>Predicted %{x}<br>Count %{z}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        height=400,
    )
    return fig


def _empty_figure(title: str = "No simulation yet"):
    fig = go.Figure()
    fig.update_layout(title=title, height=400)
    return fig


# =========================================================
# Dash App
# =========================================================


def create_app():
    app = Dash(__name__)
    app.title = "Real-Time Fraud Attack Simulator"

    app.layout = html.Div(
        style={
            "fontFamily": "Arial",
            "padding": "20px",
            "maxWidth": "1650px",
            "margin": "0 auto",
        },
        children=[
            dcc.Store(id="audit-store", data=[]),
            html.H1("🚨 Real-Time Fraud Attack Simulator"),
            html.P(
                "Choose the attack type, then stream transactions row-by-row through the model. "
                "Each transaction gets ALLOW / REVIEW / BLOCK."
            ),
            html.Div(
                style={
                    "display": "grid",
                    "gridTemplateColumns": "2fr 1fr 1fr 1fr 1fr",
                    "gap": "15px",
                    "marginBottom": "20px",
                    "alignItems": "end",
                },
                children=[
                    html.Div(
                        children=[
                            html.Label("Choose Attack Type"),
                            dcc.Dropdown(
                                id="attack-dropdown",
                                options=ATTACK_OPTIONS,
                                value="01_normal_baseline",
                                clearable=False,
                            ),
                        ]
                    ),
                    html.Div(
                        children=[
                            html.Label("Alert Threshold"),
                            dcc.Input(
                                id="alert-threshold-input",
                                type="number",
                                value=0.12,
                                min=0.0,
                                max=0.99,
                                step=0.01,
                                style={"width": "100%"},
                            ),
                        ]
                    ),
                    html.Div(
                        children=[
                            html.Label("Block Threshold"),
                            dcc.Input(
                                id="block-threshold-input",
                                type="number",
                                value=0.3,
                                min=0.01,
                                max=0.99,
                                step=0.01,
                                style={"width": "100%"},
                            ),
                        ]
                    ),
                    html.Div(
                        children=[
                            html.Label("Flood Repeats"),
                            dcc.Input(
                                id="flood-input",
                                type="number",
                                value=8,
                                min=1,
                                max=50,
                                step=1,
                                style={"width": "100%"},
                            ),
                        ]
                    ),
                    html.Button(
                        "▶ Run Real-Time Simulation",
                        id="run-btn",
                        n_clicks=0,
                        style={
                            "padding": "12px 20px",
                            "cursor": "pointer",
                            "fontSize": "16px",
                            "height": "45px",
                        },
                    ),
                ],
            ),
            html.Div(
                id="status-msg",
                style={"marginBottom": "20px", "fontWeight": "bold"},
            ),
            dcc.Loading(
                type="circle",
                children=[
                    html.Div(
                        style={
                            "display": "grid",
                            "gridTemplateColumns": "1.25fr 0.75fr",
                            "gap": "20px",
                        },
                        children=[
                            html.Div(
                                children=[
                                    html.H2("Real-Time Predictions"),
                                    html.Div(id="predictions-table-wrap"),
                                ]
                            ),
                            html.Div(
                                children=[
                                    html.H2("Confusion Matrix"),
                                    dcc.Graph(id="cm-graph", figure=_empty_figure()),
                                ]
                            ),
                        ],
                    ),
                    html.Div(
                        style={
                            "display": "grid",
                            "gridTemplateColumns": "1fr 1fr",
                            "gap": "20px",
                            "marginTop": "30px",
                        },
                        children=[
                            html.Div(
                                children=[
                                    html.H2("System Metrics"),
                                    html.Div(id="fpfn-metrics"),
                                ]
                            ),
                            html.Div(
                                children=[
                                    html.H2("Live Event Logs"),
                                    html.Pre(
                                        id="log-pre",
                                        style={
                                            "backgroundColor": "#111",
                                            "color": "#0f0",
                                            "padding": "15px",
                                            "height": "500px",
                                            "overflowY": "scroll",
                                            "fontSize": "12px",
                                        },
                                    ),
                                ]
                            ),
                        ],
                    ),
                    html.Div(
                        style={"marginTop": "40px"},
                        children=[
                            html.H2("Selected Attack Explanation"),
                            html.Div(
                                id="attack-explanations",
                                style={
                                    "padding": "20px",
                                    "backgroundColor": "#f5f5f5",
                                    "borderRadius": "8px",
                                    "whiteSpace": "pre-wrap",
                                },
                            ),
                        ],
                    ),
                    html.Div(
                        style={"marginTop": "40px"},
                        children=[
                            html.H2("🔍 Interactive Fraud Probability Audit"),
                            html.P("Click on any row in the Predictions Table above to show a detailed explanation of the decision, probability source, formula, and top feature evidence."),
                            html.Div(
                                id="audit-details-wrap",
                                style={
                                    "padding": "20px",
                                    "backgroundColor": "#f9f9f9",
                                    "border": "1px solid #ddd",
                                    "borderRadius": "8px",
                                    "minHeight": "100px",
                                },
                                children="No transaction selected. Click on a row in the Predictions Table above to audit."
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )

    @callback(
        Output("status-msg", "children"),
        Output("predictions-table-wrap", "children"),
        Output("cm-graph", "figure"),
        Output("fpfn-metrics", "children"),
        Output("log-pre", "children"),
        Output("attack-explanations", "children"),
        Output("audit-store", "data"),
        Input("run-btn", "n_clicks"),
        State("attack-dropdown", "value"),
        State("alert-threshold-input", "value"),
        State("block-threshold-input", "value"),
        State("flood-input", "value"),
        prevent_initial_call=True,
    )
    def on_run(_clicks, selected_attack, alert_threshold, block_threshold, flood_repeats):
        root = _project_root()

        alert_threshold = float(alert_threshold)
        block_threshold = float(block_threshold)
        flood_repeats = int(flood_repeats)

        if alert_threshold > block_threshold:
            return (
                "Error: Alert threshold must be <= Block threshold.",
                html.Div(),
                _empty_figure("Invalid thresholds"),
                html.Div(),
                "",
                "Alert threshold must be lower than or equal to block threshold.",
                [],
            )

        data_path = _find_data_path(root)
        model, scaler, feature_names = _load_model_bundle(root)
        X_sample, y_sample = _sample_base_transactions(data_path, feature_names)

        if selected_attack == "all":
            attacks_to_run = [item["value"] for item in ATTACK_OPTIONS if item["value"] != "all"]
        else:
            attacks_to_run = [selected_attack]

        all_rows = []
        all_y_true: List[int] = []
        all_y_pred: List[int] = []
        all_latencies: List[float] = []
        per_attack_summaries = []
        all_audit_dfs = []

        for attack_type in attacks_to_run:
            X_attack, y_attack = _apply_attack(
                attack_type,
                X_sample,
                y_sample,
                flood_repeats=flood_repeats,
            )
            attack_summary = _run_realtime_stream(
                model,
                scaler,
                feature_names,
                X_attack,
                y_attack,
                attack_type=attack_type,
                alert_threshold=alert_threshold,
                block_threshold=block_threshold,
            )
            per_attack_summaries.append(attack_summary)
            all_rows.extend(attack_summary["rows"])
            all_y_true.extend(attack_summary["y_true"])
            all_y_pred.extend(attack_summary["y_pred"])
            all_latencies.extend(attack_summary["latencies_ms"])

            # Audit predictions on real-time stream data
            audit_df = audit_predictions(
                model=model,
                X=X_attack,
                y_true=y_attack,
                scenario_name=attack_type,
                feature_names=feature_names,
                scaler=scaler,
                alert_threshold=alert_threshold,
                block_threshold=block_threshold,
                max_rows=300,
                top_k=5,
            )
            all_audit_dfs.append(audit_df)

        if all_audit_dfs:
            consolidated_audit_df = pd.concat(all_audit_dfs, ignore_index=True)
            save_audit_logs(consolidated_audit_df)
            audit_records = consolidated_audit_df.to_dict("records")
        else:
            audit_records = []

        cm, counts = _confusion_counts(all_y_true, all_y_pred)
        total_tx = len(all_rows)

        status = (
            f"Processed {total_tx} transactions in real-time mode. "
            f"Selected attack: {selected_attack}. "
            f"Alert={alert_threshold}, Block={block_threshold}."
        )

        # Predictions table
        df = pd.DataFrame(all_rows)
        if not df.empty:
            df_table = df.copy()
            df_table["probability"] = df_table["probability"].round(4)
            df_table["latency_ms"] = df_table["latency_ms"].round(3)
            if "amount" in df_table.columns:
                df_table["amount"] = df_table["amount"].round(2)
        else:
            df_table = pd.DataFrame()

        table = DataTable(
            id="predictions-table",
            columns=[{"name": c, "id": c} for c in df_table.columns],
            data=df_table.head(500).to_dict("records"),
            page_size=12,
            style_table={"overflowX": "auto"},
            style_cell={
                "textAlign": "left",
                "padding": "8px",
                "whiteSpace": "normal",
                "height": "auto",
                "fontSize": "12px",
            },
            style_header={"fontWeight": "bold"},
            style_data_conditional=[
                {
                    "if": {"filter_query": "{decision} = BLOCK"},
                    "backgroundColor": "#ffe6e6",
                },
                {
                    "if": {"filter_query": "{decision} = REVIEW"},
                    "backgroundColor": "#fff4cc",
                },
                {
                    "if": {"filter_query": "{decision} = ALLOW"},
                    "backgroundColor": "#e9ffe9",
                },
            ],
        )

        fig = _confusion_figure(cm)

        allow_count = sum(1 for r in all_rows if r["decision"] == "ALLOW")
        review_count = sum(1 for r in all_rows if r["decision"] == "REVIEW")
        block_count = sum(1 for r in all_rows if r["decision"] == "BLOCK")

        accuracy = _safe_div(counts["TP"] + counts["TN"], total_tx)
        precision = _safe_div(counts["TP"], counts["TP"] + counts["FP"])
        recall = _safe_div(counts["TP"], counts["TP"] + counts["FN"])
        f1 = _safe_div(2 * precision * recall, precision + recall)

        metrics = html.Div(
            children=[
                html.Ul(
                    [
                        html.Li(f"Total Transactions: {total_tx}"),
                        html.Li(f"ALLOW: {allow_count}"),
                        html.Li(f"REVIEW: {review_count}"),
                        html.Li(f"BLOCK: {block_count}"),
                        html.Li(f"True Positives: {counts['TP']}"),
                        html.Li(f"True Negatives: {counts['TN']}"),
                        html.Li(f"False Positives: {counts['FP']}"),
                        html.Li(f"False Negatives: {counts['FN']}"),
                        html.Li(f"Accuracy: {accuracy:.4f}"),
                        html.Li(f"Precision: {precision:.4f}"),
                        html.Li(f"Recall: {recall:.4f}"),
                        html.Li(f"F1: {f1:.4f}"),
                        html.Li(f"Average Latency: {np.mean(all_latencies):.3f} ms" if all_latencies else "Average Latency: 0 ms"),
                        html.Li(f"P95 Latency: {np.percentile(all_latencies, 95):.3f} ms" if all_latencies else "P95 Latency: 0 ms"),
                    ]
                )
            ]
        )

        logs = []
        for i, r in enumerate(all_rows[:500]):
            logs.append(
                f"""
TX-{i}
ATTACK: {r['attack']}
PROBABILITY: {r['probability']:.4f}
DECISION: {r['decision']}
ACTION: {r['action']}
TRUE LABEL: {r['true_fraud']}
PREDICTION: {r['pred_fraud']}
LATENCY: {r['latency_ms']:.3f} ms
================================================
"""
            )
        log_text = "\n".join(logs)

        attack_text_parts = []
        for s in per_attack_summaries:
            m = s["metrics"]
            dc = s["decision_counts"]
            attack_text_parts.append(
                f"=== {s['attack_name']} ===\n\n"
                f"{s['attack_description']}\n\n"
                f"Transactions: {m['total_transactions']}\n"
                f"ALLOW / REVIEW / BLOCK: {dc['ALLOW']} / {dc['REVIEW']} / {dc['BLOCK']}\n"
                f"Accuracy: {m['accuracy']:.4f}\n"
                f"Precision: {m['precision']:.4f}\n"
                f"Recall: {m['recall']:.4f}\n"
                f"F1: {m['f1']:.4f}\n"
                f"Average latency: {m['avg_latency_ms']:.3f} ms\n"
            )
        attack_text = "\n".join(attack_text_parts)

        return status, table, fig, metrics, log_text, attack_text, audit_records

    @app.callback(
        Output("audit-details-wrap", "children"),
        Input("predictions-table", "active_cell"),
        State("predictions-table", "data"),
        State("audit-store", "data"),
        prevent_initial_call=True,
    )
    def display_audit_details(active_cell, table_data, audit_records):
        if not active_cell or not table_data or not audit_records:
            return "No transaction selected. Click on a row in the Predictions Table above to audit."
        
        row_idx = active_cell["row"]
        if row_idx >= len(table_data):
            return "Selected row index out of bounds."
            
        tx_row = table_data[row_idx]
        tx_id = tx_row.get("tx_id")
        
        try:
            parts = tx_id.split("-TX-")
            if len(parts) == 2:
                scenario_name = parts[0]
                tx_index = int(parts[1])
            else:
                scenario_name = tx_row.get("attack")
                tx_index = row_idx
        except Exception:
            scenario_name = tx_row.get("attack")
            tx_index = row_idx
            
        matched_record = None
        for record in audit_records:
            if record.get("scenario") == scenario_name and record.get("transaction_id") == tx_index:
                matched_record = record
                break
                
        if not matched_record:
            if row_idx < len(audit_records):
                matched_record = audit_records[row_idx]
                
        if not matched_record:
            return "Could not find audit logs for the selected transaction."
            
        decision = matched_record.get("decision", "UNKNOWN")
        color_map = {
            "ALLOW": "#2ecc71",
            "REVIEW": "#f39c12",
            "BLOCK": "#e74c3c"
        }
        color = color_map.get(decision, "#7f8c8d")
        
        top_features = matched_record.get("top_features", "")
        reason_text = matched_record.get("reason", "")
        
        features_list = [f.strip() for f in top_features.split(",") if f.strip()]
        
        # Build explanation view with elite UI aesthetics
        return html.Div(
            style={
                "fontFamily": "Segoe UI, Arial, sans-serif",
                "color": "#333",
            },
            children=[
                html.Div(
                    style={
                        "display": "flex",
                        "justifyContent": "space-between",
                        "alignItems": "center",
                        "borderBottom": "2px solid #eee",
                        "paddingBottom": "12px",
                        "marginBottom": "15px"
                    },
                    children=[
                        html.Div([
                            html.H3(f"Transaction ID: {tx_id}", style={"margin": "0", "color": "#2c3e50", "fontWeight": "600"}),
                            html.Span(f"Scenario: {matched_record.get('scenario')}", style={"fontSize": "12px", "color": "#7f8c8d", "marginTop": "4px", "display": "block"})
                        ]),
                        html.Span(
                            decision,
                            style={
                                "backgroundColor": color,
                                "color": "white",
                                "padding": "6px 16px",
                                "borderRadius": "20px",
                                "fontWeight": "bold",
                                "fontSize": "14px",
                                "boxShadow": "0 2px 5px rgba(0,0,0,0.1)"
                            }
                        )
                    ]
                ),
                html.Div(
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "1fr 1fr",
                        "gap": "20px"
                    },
                    children=[
                        html.Div(
                            style={"backgroundColor": "#fff", "padding": "15px", "borderRadius": "6px", "boxShadow": "0 1px 3px rgba(0,0,0,0.05)"},
                            children=[
                                html.H4("Decision Summary", style={"marginTop": "0", "borderBottom": "1px solid #f1f1f1", "paddingBottom": "8px", "color": "#34495e"}),
                                html.P([html.Strong("Fraud Probability: "), f"{matched_record.get('fraud_probability'):.6f}"], style={"margin": "6px 0"}),
                                html.P([html.Strong("Alert Threshold: "), f"{matched_record.get('alert_threshold'):.2f}"], style={"margin": "6px 0"}),
                                html.P([html.Strong("Block Threshold: "), f"{matched_record.get('block_threshold'):.2f}"], style={"margin": "6px 0"}),
                                html.P([html.Strong("System Action: "), matched_record.get("action")], style={"margin": "6px 0"}),
                                html.P(
                                    [
                                        html.Strong("Ground Truth vs Pred: "),
                                        html.Span(f"True={matched_record.get('true_fraud')}, Pred={matched_record.get('pred_fraud')}", style={"fontWeight": "bold"}),
                                        f" ({matched_record.get('result_type')})"
                                    ],
                                    style={"margin": "6px 0"}
                                ),
                            ]
                        ),
                        html.Div(
                            style={"backgroundColor": "#fff", "padding": "15px", "borderRadius": "6px", "boxShadow": "0 1px 3px rgba(0,0,0,0.05)"},
                            children=[
                                html.H4("Top Feature Evidence", style={"marginTop": "0", "borderBottom": "1px solid #f1f1f1", "paddingBottom": "8px", "color": "#34495e"}),
                                html.Ul(
                                    [html.Li(feat, style={"padding": "4px 0", "fontSize": "13px"}) for feat in features_list],
                                    style={"paddingLeft": "20px", "margin": "0"}
                                )
                            ]
                        )
                    ]
                ),
                html.Div(
                    style={"marginTop": "20px"},
                    children=[
                        html.H4("Detailed Audit & Explanation", style={"color": "#34495e", "marginBottom": "8px"}),
                        html.Pre(
                            reason_text,
                            style={
                                "backgroundColor": "#f8f9fa",
                                "color": "#2c3e50",
                                "padding": "15px",
                                "borderRadius": "6px",
                                "whiteSpace": "pre-wrap",
                                "fontFamily": "Consolas, Monaco, monospace",
                                "fontSize": "12px",
                                "borderLeft": f"5px solid {color}",
                                "boxShadow": "inset 0 1px 3px rgba(0,0,0,0.02)",
                                "border": "1px solid #e9ecef",
                                "borderLeftWidth": "5px"
                            }
                        )
                    ]
                )
            ]
        )

    return app


# =========================================================
# Main
# =========================================================


def main():
    app = create_app()
    app.run(debug=True, host="127.0.0.1", port=8050)


if __name__ == "__main__":
    main()
