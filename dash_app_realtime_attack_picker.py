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
from plotly.subplots import make_subplots

from fraud_probability_audit import audit_predictions, save_audit_logs


# =========================================================
# Visual theme (fintech / SOC operations)
# =========================================================

THEME = {
    "bg": "#0b1220",
    "bg_gradient": "linear-gradient(145deg, #0b1220 0%, #111827 45%, #0f172a 100%)",
    "surface": "#151f32",
    "surface_raised": "#1c2942",
    "border": "rgba(148, 163, 184, 0.18)",
    "text": "#f1f5f9",
    "text_muted": "#94a3b8",
    "accent": "#38bdf8",
    "accent_soft": "rgba(56, 189, 248, 0.15)",
    "allow": "#22c55e",
    "review": "#f59e0b",
    "block": "#ef4444",
    "tp": "#10b981",
    "tn": "#0ea5e9",
    "fp": "#fb923c",
    "fn": "#f43f5e",
    "font": "'Segoe UI', 'Inter', system-ui, sans-serif",
    "mono": "'JetBrains Mono', 'Consolas', monospace",
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family=THEME["font"], color=THEME["text"], size=12),
    margin=dict(l=48, r=24, t=52, b=40),
    title_font=dict(size=15, color=THEME["text"]),
    xaxis=dict(
        gridcolor="rgba(148,163,184,0.12)",
        zerolinecolor="rgba(148,163,184,0.2)",
        tickfont=dict(color=THEME["text_muted"]),
        title_font=dict(color=THEME["text_muted"]),
    ),
    yaxis=dict(
        gridcolor="rgba(148,163,184,0.12)",
        zerolinecolor="rgba(148,163,184,0.2)",
        tickfont=dict(color=THEME["text_muted"]),
        title_font=dict(color=THEME["text_muted"]),
    ),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=THEME["text"])),
)

TABLE_STYLES = {
    "style_table": {
        "overflowX": "auto",
        "borderRadius": "10px",
        "border": f"1px solid {THEME['border']}",
    },
    "style_header": {
        "backgroundColor": THEME["surface_raised"],
        "color": THEME["accent"],
        "fontWeight": "600",
        "fontSize": "12px",
        "textTransform": "uppercase",
        "letterSpacing": "0.04em",
        "border": "none",
    },
    "style_cell": {
        "backgroundColor": THEME["surface"],
        "color": THEME["text"],
        "textAlign": "left",
        "padding": "10px 12px",
        "whiteSpace": "normal",
        "height": "auto",
        "fontSize": "12px",
        "fontFamily": THEME["mono"],
        "border": f"1px solid {THEME['border']}",
    },
    "style_data_conditional": [
        {"if": {"filter_query": "{decision} = BLOCK"}, "backgroundColor": "rgba(239,68,68,0.18)", "color": "#fecaca"},
        {"if": {"filter_query": "{decision} = REVIEW"}, "backgroundColor": "rgba(245,158,11,0.15)", "color": "#fde68a"},
        {"if": {"filter_query": "{decision} = ALLOW"}, "backgroundColor": "rgba(34,197,94,0.12)", "color": "#bbf7d0"},
        {"if": {"state": "active"}, "backgroundColor": THEME["accent_soft"], "border": f"1px solid {THEME['accent']}"},
    ],
}


def _card(children, *, style=None):
    base = {
        "background": THEME["surface"],
        "border": f"1px solid {THEME['border']}",
        "borderRadius": "14px",
        "padding": "18px 20px",
        "boxShadow": "0 8px 32px rgba(0,0,0,0.35)",
    }
    if style:
        base.update(style)
    return html.Div(children=children, style=base)


def _section_title(text: str, subtitle: str | None = None):
    parts = [
        html.H2(
            text,
            style={
                "margin": "0 0 6px 0",
                "fontSize": "17px",
                "fontWeight": "600",
                "color": THEME["text"],
                "letterSpacing": "-0.02em",
            },
        )
    ]
    if subtitle:
        parts.append(
            html.P(
                subtitle,
                style={"margin": "0 0 14px 0", "fontSize": "13px", "color": THEME["text_muted"]},
            )
        )
    return html.Div(parts)


def _kpi_card(label: str, value: str, *, accent: str, hint: str = ""):
    return html.Div(
        style={
            "background": f"linear-gradient(135deg, {THEME['surface_raised']} 0%, {THEME['surface']} 100%)",
            "border": f"1px solid {THEME['border']}",
            "borderLeft": f"4px solid {accent}",
            "borderRadius": "12px",
            "padding": "16px 18px",
            "minHeight": "88px",
        },
        children=[
            html.Div(label, style={"fontSize": "11px", "color": THEME["text_muted"], "textTransform": "uppercase", "letterSpacing": "0.06em"}),
            html.Div(value, style={"fontSize": "26px", "fontWeight": "700", "color": THEME["text"], "margin": "6px 0 4px 0"}),
            html.Div(hint, style={"fontSize": "12px", "color": accent}) if hint else None,
        ],
    )


def _status_banner(message: str, *, variant: str = "info"):
    colors = {
        "info": (THEME["accent_soft"], THEME["accent"]),
        "error": ("rgba(239,68,68,0.15)", THEME["block"]),
        "success": ("rgba(34,197,94,0.12)", THEME["allow"]),
    }
    bg, border = colors.get(variant, colors["info"])
    return html.Div(
        message,
        style={
            "padding": "12px 18px",
            "borderRadius": "10px",
            "background": bg,
            "border": f"1px solid {border}",
            "color": THEME["text"],
            "fontSize": "14px",
            "fontWeight": "500",
        },
    )


def _build_kpi_row(
    *,
    total_tx: int,
    allow_count: int,
    review_count: int,
    block_count: int,
    counts: dict,
    accuracy: float,
    recall: float,
    avg_latency: float,
):
    return html.Div(
        style={
            "display": "grid",
            "gridTemplateColumns": "repeat(auto-fit, minmax(160px, 1fr))",
            "gap": "14px",
            "marginBottom": "22px",
        },
        children=[
            _kpi_card("Transactions", f"{total_tx:,}", accent=THEME["accent"], hint="Streamed in real time"),
            _kpi_card("Blocked", f"{block_count:,}", accent=THEME["block"], hint=f"Review: {review_count:,} · Allow: {allow_count:,}"),
            _kpi_card("Fraud caught", f"{counts['TP']}", accent=THEME["tp"], hint=f"Missed (FN): {counts['FN']}"),
            _kpi_card("Accuracy", f"{accuracy:.1%}", accent=THEME["tn"], hint=f"Recall: {recall:.1%}"),
            _kpi_card("Avg latency", f"{avg_latency:.2f} ms", accent="#a78bfa", hint="Per transaction score"),
        ],
    )


def _attack_explanation_cards(per_attack_summaries: list) -> html.Div:
    if not per_attack_summaries:
        return html.P("Run a simulation to see attack scenario breakdowns.", style={"color": THEME["text_muted"]})

    cards = []
    for s in per_attack_summaries:
        m = s["metrics"]
        dc = s["decision_counts"]
        name = s["attack_name"].replace("_", " ").title()
        cards.append(
            html.Div(
                style={
                    "background": THEME["surface_raised"],
                    "border": f"1px solid {THEME['border']}",
                    "borderRadius": "12px",
                    "padding": "16px",
                    "marginBottom": "12px",
                },
                children=[
                    html.Div(
                        style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "8px"},
                        children=[
                            html.Strong(name, style={"color": THEME["accent"], "fontSize": "14px"}),
                            html.Span(
                                f"F1 {m['f1']:.2%}",
                                style={
                                    "fontSize": "12px",
                                    "padding": "4px 10px",
                                    "borderRadius": "20px",
                                    "background": THEME["accent_soft"],
                                    "color": THEME["accent"],
                                },
                            ),
                        ],
                    ),
                    html.P(s["attack_description"], style={"fontSize": "13px", "color": THEME["text_muted"], "margin": "0 0 10px 0", "lineHeight": "1.5"}),
                    html.Div(
                        style={"display": "flex", "gap": "16px", "flexWrap": "wrap", "fontSize": "12px", "color": THEME["text"]},
                        children=[
                            html.Span([html.Span("● ", style={"color": THEME["allow"]}), f"Allow {dc['ALLOW']}"]),
                            html.Span([html.Span("● ", style={"color": THEME["review"]}), f"Review {dc['REVIEW']}"]),
                            html.Span([html.Span("● ", style={"color": THEME["block"]}), f"Block {dc['BLOCK']}"]),
                            html.Span(f"· {m['total_transactions']} tx · {m['avg_latency_ms']:.1f} ms avg"),
                        ],
                    ),
                ],
            )
        )
    return html.Div(cards)


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


def _apply_plotly_theme(fig: go.Figure, title: str, height: int = 360) -> go.Figure:
    fig.update_layout(**PLOTLY_LAYOUT, title=dict(text=title, x=0.02, xanchor="left"), height=height)
    return fig


def _confusion_figure(cm: np.ndarray) -> go.Figure:
    """Annotated 2×2 matrix with fraud-ops semantics (TN, FP, FN, TP)."""
    labels = ["Legitimate", "Fraud"]
    cell_labels = [
        ["True Negative", "False Positive"],
        ["False Negative", "True Positive"],
    ]
    colors = [
        [THEME["tn"], THEME["fp"]],
        [THEME["fn"], THEME["tp"]],
    ]
    z_text = [[str(int(cm[i, j])) for j in range(2)] for i in range(2)]

    fig = go.Figure()
    for i in range(2):
        for j in range(2):
            val = int(cm[i, j])
            fig.add_trace(
                go.Scatter(
                    x=[j],
                    y=[1 - i],
                    mode="markers+text",
                    marker=dict(
                        size=110,
                        color=colors[i][j],
                        opacity=0.92,
                        line=dict(width=2, color="rgba(255,255,255,0.25)"),
                    ),
                    text=[f"{val}\n{cell_labels[i][j]}"],
                    textposition="middle center",
                    textfont=dict(color="#fff", size=13),
                    hovertemplate=(
                        f"{cell_labels[i][j]}<br>Count: {val}<br>"
                        f"Actual: {labels[i]} · Predicted: {labels[j]}<extra></extra>"
                    ),
                    showlegend=False,
                )
            )

    fig.update_layout(
        xaxis=dict(
            tickvals=[0, 1],
            ticktext=labels,
            title="Predicted class",
            range=[-0.55, 1.55],
            constrain="domain",
        ),
        yaxis=dict(
            tickvals=[0, 1],
            ticktext=list(reversed(labels)),
            title="Actual class",
            range=[-0.55, 1.55],
            scaleanchor="x",
            scaleratio=1,
        ),
    )
    return _apply_plotly_theme(fig, "Detection outcomes", height=380)


def _decision_pie_figure(allow: int, review: int, block: int) -> go.Figure:
    labels = ["ALLOW", "REVIEW", "BLOCK"]
    values = [allow, review, block]
    colors = [THEME["allow"], THEME["review"], THEME["block"]]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.55,
                marker=dict(colors=colors, line=dict(color=THEME["surface"], width=2)),
                textinfo="label+percent",
                textfont=dict(color=THEME["text"], size=12),
                hovertemplate="<b>%{label}</b><br>%{value} tx (%{percent})<extra></extra>",
            )
        ]
    )
    fig.update_layout(showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=-0.12))
    return _apply_plotly_theme(fig, "Operational decisions", height=380)


def _performance_figure(
    *,
    accuracy: float,
    precision: float,
    recall: float,
    f1: float,
    counts: dict,
) -> go.Figure:
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Model quality", "Error breakdown"),
        horizontal_spacing=0.14,
        specs=[[{"type": "bar"}, {"type": "bar"}]],
    )

    metrics = ["Accuracy", "Precision", "Recall", "F1"]
    scores = [accuracy, precision, recall, f1]
    fig.add_trace(
        go.Bar(
            x=metrics,
            y=scores,
            marker=dict(
                color=[THEME["accent"], THEME["tp"], THEME["review"], "#a78bfa"],
                line=dict(width=0),
            ),
            text=[f"{s:.1%}" for s in scores],
            textposition="outside",
            textfont=dict(color=THEME["text"]),
            hovertemplate="%{x}: %{y:.2%}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.update_yaxes(range=[0, 1.08], tickformat=".0%", row=1, col=1)

    err_labels = ["True +", "True −", "False +", "False −"]
    err_vals = [counts["TP"], counts["TN"], counts["FP"], counts["FN"]]
    err_colors = [THEME["tp"], THEME["tn"], THEME["fp"], THEME["fn"]]
    fig.add_trace(
        go.Bar(
            x=err_labels,
            y=err_vals,
            marker=dict(color=err_colors),
            text=err_vals,
            textposition="outside",
            textfont=dict(color=THEME["text"]),
            hovertemplate="%{x}: %{y}<extra></extra>",
        ),
        row=1,
        col=2,
    )

    for ann in fig.layout.annotations:
        ann.font.color = THEME["text_muted"]
        ann.font.size = 12

    fig = _apply_plotly_theme(fig, "Performance snapshot", height=380)
    return fig


def _latency_figure(latencies: List[float]) -> go.Figure:
    if not latencies:
        fig = go.Figure()
        fig.add_annotation(
            text="No latency data yet",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(color=THEME["text_muted"], size=14),
        )
        return _apply_plotly_theme(fig, "Scoring latency", height=280)

    arr = np.asarray(latencies, dtype=float)
    fig = go.Figure(
        data=[
            go.Histogram(
                x=arr,
                nbinsx=min(40, max(10, len(arr) // 20)),
                marker=dict(color=THEME["accent"], line=dict(color=THEME["surface"], width=1)),
                hovertemplate="%{x:.2f} ms · %{y} transactions<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        xaxis_title="Latency (ms)",
        yaxis_title="Count",
        bargap=0.05,
    )
    p95 = float(np.percentile(arr, 95))
    fig.add_vline(x=p95, line_dash="dash", line_color=THEME["review"], annotation_text=f"p95: {p95:.1f} ms")
    return _apply_plotly_theme(fig, "Scoring latency distribution", height=280)


def _empty_figure(title: str = "Awaiting simulation run") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=title,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(color=THEME["text_muted"], size=14),
    )
    return _apply_plotly_theme(fig, title, height=360)


# =========================================================
# Dash App
# =========================================================


def _badge_style(color: str) -> dict:
    return {
        "fontSize": "11px",
        "padding": "6px 12px",
        "borderRadius": "20px",
        "background": "rgba(255,255,255,0.06)",
        "border": f"1px solid {color}",
        "color": color,
    }


def _control_label(text: str):
    return html.Label(
        text,
        style={
            "display": "block",
            "fontSize": "11px",
            "fontWeight": "600",
            "color": THEME["text_muted"],
            "textTransform": "uppercase",
            "letterSpacing": "0.05em",
            "marginBottom": "6px",
        },
    )


def _control_input_style():
    return {
        "width": "100%",
        "padding": "10px 12px",
        "borderRadius": "8px",
        "border": f"1px solid {THEME['border']}",
        "background": THEME["surface_raised"],
        "color": THEME["text"],
        "fontSize": "14px",
    }


def create_app():
    app = Dash(__name__)
    app.title = "FraudShield · Real-Time Simulation"

    app.layout = html.Div(
        style={
            "fontFamily": THEME["font"],
            "minHeight": "100vh",
            "background": THEME["bg_gradient"],
            "color": THEME["text"],
            "padding": "24px 28px 48px",
        },
        children=[
            dcc.Store(id="audit-store", data=[]),
            # —— Header ——
            html.Div(
                style={
                    "display": "flex",
                    "justifyContent": "space-between",
                    "alignItems": "flex-end",
                    "flexWrap": "wrap",
                    "gap": "16px",
                    "marginBottom": "24px",
                    "paddingBottom": "20px",
                    "borderBottom": f"1px solid {THEME['border']}",
                },
                children=[
                    html.Div(
                        children=[
                            html.Div(
                                "FRAUDSHIELD",
                                style={
                                    "fontSize": "11px",
                                    "fontWeight": "700",
                                    "letterSpacing": "0.2em",
                                    "color": THEME["accent"],
                                    "marginBottom": "8px",
                                },
                            ),
                            html.H1(
                                "Real-Time Attack Simulator",
                                style={
                                    "margin": 0,
                                    "fontSize": "clamp(1.6rem, 3vw, 2.1rem)",
                                    "fontWeight": "700",
                                    "letterSpacing": "-0.03em",
                                },
                            ),
                            html.P(
                                "Stream PaySim transactions through your trained model. "
                                "Each row receives ALLOW · REVIEW · BLOCK based on live fraud probability.",
                                style={"margin": "10px 0 0", "color": THEME["text_muted"], "maxWidth": "640px", "lineHeight": "1.55"},
                            ),
                        ]
                    ),
                    html.Div(
                        style={"display": "flex", "gap": "10px", "flexWrap": "wrap"},
                        children=[
                            html.Span("● Live scoring", style=_badge_style(THEME["allow"])),
                            html.Span("● 8 attack vectors", style=_badge_style(THEME["accent"])),
                            html.Span("● Row audit", style=_badge_style(THEME["review"])),
                        ],
                    ),
                ],
            ),
            # —— Controls ——
            _card(
                [
                    html.Div(
                        style={
                            "display": "grid",
                            "gridTemplateColumns": "2fr repeat(3, 1fr) auto",
                            "gap": "16px",
                            "alignItems": "end",
                        },
                        children=[
                            html.Div(
                                children=[
                                    _control_label("Attack scenario"),
                                    dcc.Dropdown(
                                        id="attack-dropdown",
                                        options=ATTACK_OPTIONS,
                                        value="01_normal_baseline",
                                        clearable=False,
                                        style={"color": "#0f172a"},
                                    ),
                                ]
                            ),
                            html.Div(
                                children=[
                                    _control_label("Alert ≥ review"),
                                    dcc.Input(
                                        id="alert-threshold-input",
                                        type="number",
                                        value=0.12,
                                        min=0.0,
                                        max=0.99,
                                        step=0.01,
                                        style=_control_input_style(),
                                    ),
                                ]
                            ),
                            html.Div(
                                children=[
                                    _control_label("Block ≥ stop"),
                                    dcc.Input(
                                        id="block-threshold-input",
                                        type="number",
                                        value=0.3,
                                        min=0.01,
                                        max=0.99,
                                        step=0.01,
                                        style=_control_input_style(),
                                    ),
                                ]
                            ),
                            html.Div(
                                children=[
                                    _control_label("Flood repeats"),
                                    dcc.Input(
                                        id="flood-input",
                                        type="number",
                                        value=8,
                                        min=1,
                                        max=50,
                                        step=1,
                                        style=_control_input_style(),
                                    ),
                                ]
                            ),
                            html.Button(
                                "Run simulation",
                                id="run-btn",
                                n_clicks=0,
                                style={
                                    "padding": "12px 28px",
                                    "cursor": "pointer",
                                    "fontSize": "14px",
                                    "fontWeight": "600",
                                    "height": "42px",
                                    "border": "none",
                                    "borderRadius": "10px",
                                    "background": f"linear-gradient(135deg, {THEME['accent']} 0%, #0284c7 100%)",
                                    "color": "#0b1220",
                                    "boxShadow": "0 4px 20px rgba(56,189,248,0.35)",
                                },
                            ),
                        ],
                    ),
                ],
                style={"marginBottom": "18px"},
            ),
            html.Div(id="status-msg", style={"marginBottom": "18px"}),
            html.Div(id="kpi-row"),
            dcc.Loading(
                type="dot",
                color=THEME["accent"],
                children=[
                    html.Div(
                        style={
                            "display": "grid",
                            "gridTemplateColumns": "1.2fr 0.8fr",
                            "gap": "18px",
                            "marginBottom": "18px",
                        },
                        children=[
                            _card(
                                [
                                    _section_title(
                                        "Transaction stream",
                                        "Click a row below for the full probability audit.",
                                    ),
                                    html.Div(
                                        id="predictions-table-wrap",
                                        children=[
                                            DataTable(
                                                id="predictions-table",
                                                columns=[],
                                                data=[],
                                                page_size=12,
                                                **TABLE_STYLES,
                                            ),
                                        ],
                                    ),
                                ]
                            ),
                            html.Div(
                                style={"display": "flex", "flexDirection": "column", "gap": "18px"},
                                children=[
                                    _card([dcc.Graph(id="cm-graph", figure=_empty_figure("Detection outcomes"), config={"displayModeBar": False})]),
                                    _card([dcc.Graph(id="decision-chart", figure=_empty_figure("Operational decisions"), config={"displayModeBar": False})]),
                                ],
                            ),
                        ],
                    ),
                    html.Div(
                        style={
                            "display": "grid",
                            "gridTemplateColumns": "1fr 1fr",
                            "gap": "18px",
                            "marginBottom": "18px",
                        },
                        children=[
                            _card([dcc.Graph(id="performance-chart", figure=_empty_figure("Performance snapshot"), config={"displayModeBar": False})]),
                            _card(
                                [
                                    _section_title("Live event log", "Most recent scored transactions."),
                                    html.Pre(
                                        id="log-pre",
                                        children="Waiting for simulation…",
                                        style={
                                            "background": "#050a14",
                                            "color": "#7dd3fc",
                                            "padding": "16px",
                                            "height": "320px",
                                            "overflowY": "auto",
                                            "fontSize": "11px",
                                            "fontFamily": THEME["mono"],
                                            "borderRadius": "8px",
                                            "border": f"1px solid {THEME['border']}",
                                            "margin": 0,
                                            "lineHeight": "1.55",
                                        },
                                    ),
                                ]
                            ),
                        ],
                    ),
                    _card(
                        [
                            _section_title("Scoring latency"),
                            dcc.Graph(id="latency-chart", figure=_empty_figure("Latency"), config={"displayModeBar": False}),
                        ],
                        style={"marginBottom": "18px"},
                    ),
                    _card(
                        [
                            _section_title("Attack scenario intelligence"),
                            html.Div(id="attack-explanations"),
                        ],
                        style={"marginBottom": "18px"},
                    ),
                    _card(
                        [
                            _section_title(
                                "Transaction audit",
                                "Select any row in the stream table for probability breakdown and feature evidence.",
                            ),
                            html.Div(
                                id="audit-details-wrap",
                                style={"minHeight": "120px"},
                                children=html.P(
                                    "No transaction selected.",
                                    style={"color": THEME["text_muted"], "margin": 0},
                                ),
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )

    @callback(
        Output("status-msg", "children"),
        Output("kpi-row", "children"),
        Output("predictions-table", "data"),
        Output("predictions-table", "columns"),
        Output("cm-graph", "figure"),
        Output("decision-chart", "figure"),
        Output("performance-chart", "figure"),
        Output("latency-chart", "figure"),
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
            empty = _empty_figure("Invalid thresholds")
            return (
                _status_banner(
                    "Alert threshold must be ≤ block threshold.",
                    variant="error",
                ),
                html.Div(),
                [],
                [],
                empty,
                empty,
                empty,
                empty,
                "Fix thresholds and run again.",
                html.P("Invalid configuration.", style={"color": THEME["text_muted"]}),
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

        attack_label = next(
            (o["label"] for o in ATTACK_OPTIONS if o["value"] == selected_attack),
            selected_attack,
        )
        status = _status_banner(
            f"✓ Scored {total_tx:,} transactions · {attack_label} · "
            f"review ≥ {alert_threshold:.2f} · block ≥ {block_threshold:.2f}",
            variant="success",
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

        table_columns = [{"name": c, "id": c} for c in df_table.columns]
        table_data = df_table.head(500).to_dict("records")

        fig_cm = _confusion_figure(cm)

        allow_count = sum(1 for r in all_rows if r["decision"] == "ALLOW")
        review_count = sum(1 for r in all_rows if r["decision"] == "REVIEW")
        block_count = sum(1 for r in all_rows if r["decision"] == "BLOCK")

        accuracy = _safe_div(counts["TP"] + counts["TN"], total_tx)
        precision = _safe_div(counts["TP"], counts["TP"] + counts["FP"])
        recall = _safe_div(counts["TP"], counts["TP"] + counts["FN"])
        f1 = _safe_div(2 * precision * recall, precision + recall)
        avg_lat = float(np.mean(all_latencies)) if all_latencies else 0.0

        kpi_row = _build_kpi_row(
            total_tx=total_tx,
            allow_count=allow_count,
            review_count=review_count,
            block_count=block_count,
            counts=counts,
            accuracy=accuracy,
            recall=recall,
            avg_latency=avg_lat,
        )

        fig_decisions = _decision_pie_figure(allow_count, review_count, block_count)
        fig_performance = _performance_figure(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            counts=counts,
        )
        fig_latency = _latency_figure(all_latencies)

        logs = []
        for i, r in enumerate(all_rows[:120]):
            icon = {"ALLOW": "✓", "REVIEW": "!", "BLOCK": "✕"}.get(r["decision"], "·")
            logs.append(
                f"[{icon}] {r.get('tx_id', f'TX-{i}')}  "
                f"P={r['probability']:.4f}  {r['decision']:<6}  "
                f"truth={r['true_fraud']} pred={r['pred_fraud']}  "
                f"{r['latency_ms']:.2f}ms"
            )
        log_text = "\n".join(logs) if logs else "No events."

        attack_cards = _attack_explanation_cards(per_attack_summaries)

        return (
            status,
            kpi_row,
            table_data,
            table_columns,
            fig_cm,
            fig_decisions,
            fig_performance,
            fig_latency,
            log_text,
            attack_cards,
            audit_records,
        )

    @callback(
        Output("audit-details-wrap", "children"),
        Input("predictions-table", "active_cell"),
        State("predictions-table", "data"),
        State("audit-store", "data"),
        prevent_initial_call=True,
    )
    def display_audit_details(active_cell, table_data, audit_records):
        if not active_cell or not table_data or not audit_records:
            return html.P(
                "Select a row in the transaction stream to open the probability audit.",
                style={"color": THEME["text_muted"], "margin": 0},
            )
        
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
            "ALLOW": THEME["allow"],
            "REVIEW": THEME["review"],
            "BLOCK": THEME["block"],
        }
        color = color_map.get(decision, THEME["text_muted"])
        
        top_features = matched_record.get("top_features", "")
        reason_text = matched_record.get("reason", "")
        
        features_list = [f.strip() for f in top_features.split(",") if f.strip()]
        
        prob = matched_record.get("fraud_probability", 0)
        return html.Div(
            children=[
                html.Div(
                    style={
                        "display": "flex",
                        "justifyContent": "space-between",
                        "alignItems": "center",
                        "borderBottom": f"1px solid {THEME['border']}",
                        "paddingBottom": "14px",
                        "marginBottom": "16px",
                    },
                    children=[
                        html.Div(
                            children=[
                                html.H3(
                                    tx_id,
                                    style={"margin": 0, "color": THEME["text"], "fontSize": "16px", "fontFamily": THEME["mono"]},
                                ),
                                html.Span(
                                    matched_record.get("scenario", ""),
                                    style={"fontSize": "12px", "color": THEME["text_muted"], "marginTop": "4px", "display": "block"},
                                ),
                            ]
                        ),
                        html.Span(
                            decision,
                            style={
                                "backgroundColor": color,
                                "color": "#0b1220",
                                "padding": "8px 18px",
                                "borderRadius": "24px",
                                "fontWeight": "700",
                                "fontSize": "13px",
                                "letterSpacing": "0.04em",
                            },
                        ),
                    ],
                ),
                html.Div(
                    style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "16px"},
                    children=[
                        html.Div(
                            style={
                                "background": THEME["surface_raised"],
                                "padding": "16px",
                                "borderRadius": "10px",
                                "border": f"1px solid {THEME['border']}",
                            },
                            children=[
                                html.H4("Decision", style={"margin": "0 0 12px", "color": THEME["accent"], "fontSize": "13px"}),
                                html.Div(
                                    style={"fontSize": "32px", "fontWeight": "700", "color": color, "marginBottom": "8px"},
                                    children=f"{prob:.4f}",
                                ),
                                html.P(f"Alert ≥ {matched_record.get('alert_threshold'):.2f}", style={"margin": "4px 0", "fontSize": "12px", "color": THEME["text_muted"]}),
                                html.P(f"Block ≥ {matched_record.get('block_threshold'):.2f}", style={"margin": "4px 0", "fontSize": "12px", "color": THEME["text_muted"]}),
                                html.P(f"Action: {matched_record.get('action')}", style={"margin": "8px 0 0", "fontSize": "12px"}),
                                html.P(
                                    f"Truth {matched_record.get('true_fraud')} → Pred {matched_record.get('pred_fraud')} ({matched_record.get('result_type')})",
                                    style={"margin": "6px 0 0", "fontSize": "12px", "color": THEME["text_muted"]},
                                ),
                            ],
                        ),
                        html.Div(
                            style={
                                "background": THEME["surface_raised"],
                                "padding": "16px",
                                "borderRadius": "10px",
                                "border": f"1px solid {THEME['border']}",
                            },
                            children=[
                                html.H4("Feature evidence", style={"margin": "0 0 12px", "color": THEME["accent"], "fontSize": "13px"}),
                                html.Ul(
                                    [
                                        html.Li(feat, style={"padding": "5px 0", "fontSize": "12px", "color": THEME["text"]})
                                        for feat in features_list
                                    ]
                                    or [html.Li("No features recorded.", style={"color": THEME["text_muted"]})],
                                    style={"paddingLeft": "18px", "margin": 0},
                                ),
                            ],
                        ),
                    ],
                ),
                html.Pre(
                    reason_text,
                    style={
                        "marginTop": "16px",
                        "background": "#050a14",
                        "color": "#cbd5e1",
                        "padding": "16px",
                        "borderRadius": "10px",
                        "whiteSpace": "pre-wrap",
                        "fontFamily": THEME["mono"],
                        "fontSize": "11px",
                        "borderLeft": f"4px solid {color}",
                        "border": f"1px solid {THEME['border']}",
                    },
                ),
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
