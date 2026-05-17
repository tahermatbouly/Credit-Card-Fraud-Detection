"""
Advanced Dash Fraud Detection Dashboard

This dashboard:
- Loads the best trained fraud detection model
- Simulates multiple attack scenarios
- Runs real-time fraud predictions
- Uses a fraud decision engine
- Displays:
    • confusion matrices
    • FP/FN metrics
    • attack explanations
    • feature evidence
    • live event logs
    • fraud reasoning

Run:
    python dash_app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from dash import (
    Dash,
    Input,
    Output,
    State,
    callback,
    dcc,
    html,
)

from dash.dash_table import DataTable

from real_simulation import run_real_simulation


# =========================================================
# Paths
# =========================================================

def _project_root() -> Path:
    return Path(__file__).resolve().parent


# =========================================================
# Load Model + Metadata
# =========================================================

def _load_model_bundle(root: Path):

    model_path = root / "output" / "models" / "best_model.pkl"

    meta_path = root / "output" / "models" / "best_model_meta.json"

    scaler_path = root / "output" / "models" / "best_model_scaler.pkl"

    if not model_path.exists():

        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    model = joblib.load(model_path)

    scaler = None

    if meta_path.exists():

        meta = json.loads(
            meta_path.read_text(
                encoding="utf-8"
            )
        )

        if meta.get("has_scaler") and scaler_path.exists():

            scaler = joblib.load(
                scaler_path
            )

    # =====================================================
    # Feature Names
    # =====================================================

    if hasattr(model, "feature_names_in_"):

        feature_names = [
            str(c)
            for c in model.feature_names_in_
        ]

    else:

        ref = (
            root
            / "data"
            / "processed"
            / "processed_paysim.csv"
        )

        head = pd.read_csv(
            ref,
            nrows=1
        )

        feature_names = [

            c

            for c in head.columns

            if c not in {
                "isFraud",
                "isFlaggedFraud",
            }
        ]

    return model, scaler, feature_names


# =========================================================
# Sample Transactions
# =========================================================

def _sample_base_transactions(
    data_path: Path,
    *,
    max_base_sample: int = 300,
    random_state: int = 42,
):

    df = pd.read_csv(data_path)

    sample_n = min(
        max_base_sample,
        len(df)
    )

    fraud_df = df[
        df["isFraud"] == 1
    ]

    legit_df = df[
        df["isFraud"] == 0
    ]

    fraud_take = min(
        len(fraud_df),
        max(35, sample_n // 6)
    )

    legit_take = min(
        len(legit_df),
        max(sample_n - fraud_take, 1)
    )

    sample_df = pd.concat(
        [
            fraud_df.sample(
                fraud_take,
                random_state=random_state
            ),

            legit_df.sample(
                legit_take,
                random_state=random_state
            ),
        ],

        ignore_index=True,
    ).sample(
        frac=1.0,
        random_state=random_state
    )

    y_sample = sample_df["isFraud"]

    X_sample = sample_df.drop(
        columns=[
            "isFraud",
            "isFlaggedFraud",
        ]
    )

    return X_sample, y_sample


# =========================================================
# Confusion Matrix Figure
# =========================================================

def _confusion_figure(cm: np.ndarray):

    z = cm.astype(int).tolist()

    labels = [
        "Legitimate",
        "Fraud"
    ]

    fig = go.Figure(

        data=go.Heatmap(

            z=z,

            x=labels,

            y=labels,

            colorscale="Blues",

            text=z,

            texttemplate="%{text}",

            hovertemplate=(
                "Actual %{y}<br>"
                "Predicted %{x}<br>"
                "Count %{z}<extra></extra>"
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


# =========================================================
# Dash App
# =========================================================

def create_app():

    root = _project_root()

    app = Dash(__name__)

    app.title = "Fraud Detection Dashboard"

    # =====================================================
    # Layout
    # =====================================================

    app.layout = html.Div(

        style={

            "fontFamily": "Arial",

            "padding": "20px",

            "maxWidth": "1600px",

            "margin": "0 auto",
        },

        children=[

            html.H1(
                "🚨 Fraud Detection Attack Simulator"
            ),

            html.P(
                "Run 8 virtual attack scenarios (real_simulation.py) against the "
                "trained model with ALLOW / REVIEW / BLOCK decisions."
            ),

            # =================================================
            # Controls
            # =================================================

            html.Div(

                style={
                    "display": "flex",
                    "gap": "20px",
                    "marginBottom": "20px",
                    "alignItems": "center",
                },

                children=[

                    html.Button(

                        "▶ Run Attack Simulation",

                        id="run-btn",

                        n_clicks=0,

                        style={
                            "padding": "12px 20px",
                            "cursor": "pointer",
                            "fontSize": "16px",
                        },
                    ),

                    html.Div(

                        children=[

                            html.Label(
                                "Block threshold"
                            ),

                            dcc.Input(

                                id="threshold-input",

                                type="number",

                                value=0.3,

                                min=0.01,

                                max=0.99,

                                step=0.01,
                            ),
                        ]
                    ),

                    html.Div(

                        children=[

                            html.Label(
                                "Alert threshold"
                            ),

                            dcc.Input(

                                id="alert-threshold-input",

                                type="number",

                                value=0.12,

                                min=0.01,

                                max=0.99,

                                step=0.01,
                            ),
                        ]
                    ),

                    html.Div(

                        children=[

                            html.Label(
                                "Flood Repeats"
                            ),

                            dcc.Input(

                                id="flood-input",

                                type="number",

                                value=8,

                                min=1,

                                max=50,

                                step=1,
                            ),
                        ]
                    ),
                ],
            ),

            html.Div(
                id="status-msg",
                style={
                    "marginBottom": "20px",
                    "fontWeight": "bold",
                },
            ),

            # =================================================
            # Main Grid
            # =================================================

            dcc.Loading(

                type="circle",

                children=[

                    html.Div(

                        style={
                            "display": "grid",
                            "gridTemplateColumns": "1fr 1fr",
                            "gap": "20px",
                        },

                        children=[

                            html.Div(

                                children=[

                                    html.H2(
                                        "Predictions"
                                    ),

                                    html.Div(
                                        id="predictions-table-wrap"
                                    ),
                                ]
                            ),

                            html.Div(

                                children=[

                                    html.H2(
                                        "Confusion Matrix"
                                    ),

                                    dcc.Graph(
                                        id="cm-graph"
                                    ),
                                ]
                            ),
                        ],
                    ),

                    # =============================================
                    # Metrics + Logs
                    # =============================================

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

                                    html.H2(
                                        "System Metrics"
                                    ),

                                    html.Div(
                                        id="fpfn-metrics"
                                    ),
                                ]
                            ),

                            html.Div(

                                children=[

                                    html.H2(
                                        "Live Event Logs"
                                    ),

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

                    # =============================================
                    # Attack Explanations
                    # =============================================

                    html.Div(

                        style={
                            "marginTop": "40px",
                        },

                        children=[

                            html.H2(
                                "Attack Scenario Explanations"
                            ),

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
                ],
            ),
        ],
    )

    # =====================================================
    # Callback
    # =====================================================

    @callback(

        Output("status-msg", "children"),

        Output("predictions-table-wrap", "children"),

        Output("cm-graph", "figure"),

        Output("fpfn-metrics", "children"),

        Output("log-pre", "children"),

        Output("attack-explanations", "children"),

        Input("run-btn", "n_clicks"),

        State("threshold-input", "value"),

        State("alert-threshold-input", "value"),

        State("flood-input", "value"),

        prevent_initial_call=True,
    )

    def on_run(
        _clicks,
        threshold,
        alert_threshold,
        flood_repeats,
    ):

        root = _project_root()

        summary = run_real_simulation(
            project_root=root,
            block_threshold=float(threshold),
            alert_threshold=float(alert_threshold),
            flood_repeats=int(flood_repeats),
            save_outputs=True,
        )

        # =================================================
        # Metrics
        # =================================================

        overall = summary["overall"]

        counts = overall["counts"]

        cm = overall["confusion_matrix"]

        total_tx = overall["total_transactions"]

        n_scenarios = summary.get("scenario_count", 8)
        avg_lat = overall.get("avg_latency_ms", 0.0)
        status = (
            f"Ran {n_scenarios} scenarios — {total_tx} transactions scored. "
            f"Block≥{summary['threshold']}, review≥{summary['alert_threshold']}. "
            f"Avg latency: {avg_lat:.2f} ms/tx. "
            f"Results saved to output/simulation/."
        )

        # =================================================
        # Prediction Table
        # =================================================

        flat = summary["flat_results"]

        rows = []

        for r in flat:

            ex = r.get(
                "explanation",
                {}
            )

            attack_analysis = ex.get(
                "attack_analysis",
                ""
            )

            prediction_summary = ex.get(
                "prediction_summary",
                ""
            )

            truth_analysis = ex.get(
                "truth_analysis",
                ""
            )

            features = ex.get(
                "feature_evidence",
                []
            )

            feature_text = ""

            if features:

                feature_text = "\n".join(

                    [
                        f.get(
                            "interpretation",
                            ""
                        )

                        for f in features
                    ]
                )

            rows.append({

                "attack": r["attack_type"],

                "probability": round(
                    r["probability"],
                    4
                ),

                "decision": r["decision"],

                "true": r["true_fraud"],

                "pred": r["pred_fraud"],

                "attack_analysis": attack_analysis,

                "prediction_summary": prediction_summary,

                "truth_analysis": truth_analysis,

                "feature_evidence": feature_text,
            })

        df = pd.DataFrame(rows)

        table = DataTable(

            columns=[
                {
                    "name": c,
                    "id": c
                }

                for c in df.columns
            ],

            data=df.head(300).to_dict(
                "records"
            ),

            page_size=10,

            style_table={
                "overflowX": "auto"
            },

            style_cell={

                "textAlign": "left",

                "padding": "8px",

                "whiteSpace": "normal",

                "height": "auto",

                "fontSize": "12px",
            },

            style_header={
                "fontWeight": "bold"
            },
        )

        # =================================================
        # Confusion Matrix
        # =================================================

        fig = _confusion_figure(cm)

        # =================================================
        # Metrics View
        # =================================================

        allow_n = sum(1 for r in flat if r["decision"] == "ALLOW")
        review_n = sum(1 for r in flat if r["decision"] == "REVIEW")
        block_n = sum(1 for r in flat if r["decision"] == "BLOCK")

        metrics = html.Ul(

            [

                html.Li(
                    f"True Positives: {counts['TP']}"
                ),

                html.Li(
                    f"True Negatives: {counts['TN']}"
                ),

                html.Li(
                    f"False Positives: {counts['FP']}"
                ),

                html.Li(
                    f"False Negatives: {counts['FN']}"
                ),

                html.Li(
                    f"Decisions — ALLOW: {allow_n} | REVIEW: {review_n} | BLOCK: {block_n}"
                ),

                html.Li(
                    f"Avg scoring latency: {avg_lat:.2f} ms / transaction"
                ),
            ]
        )

        # =================================================
        # Event Logs
        # =================================================

        logs = []

        for i, r in enumerate(flat[:300]):

            logs.append(

                f"""
TX-{i}

ATTACK: {r['attack_type']}

PROBABILITY: {r['probability']:.4f}

DECISION: {r['decision']}

TRUE LABEL: {r['true_fraud']}

PRED FRAUD (binary): {r['pred_fraud']}

================================================
"""
            )

        log_text = "\n".join(logs)

        # =================================================
        # Attack Explanations
        # =================================================

        attack_text = ""

        for attack in summary["per_attack"]:

            attack_text += (
                f"\n=== {attack['attack_name']} ===\n\n"
            )

            attack_text += (
                attack["attack_description"]
            )

            attack_text += "\n\n"

        return (

            status,

            table,

            fig,

            metrics,

            log_text,

            attack_text,
        )

    return app


# =========================================================
# Main
# =========================================================

def main():

    app = create_app()

    app.run(

        debug=True,

        host="127.0.0.1",

        port=8050,
    )


if __name__ == "__main__":

    main()