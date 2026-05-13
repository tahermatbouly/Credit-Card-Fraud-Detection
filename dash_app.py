"""
Dash dashboard: attack simulator → model scoring → decision engine → live metrics.

Run from the project root:

    python dash_app.py

Requires: dash, plotly, pandas, scikit-learn, joblib (and project training deps).
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, callback, dcc, html
from dash.dash_table import DataTable

from src.simulation.simulation_runner import run_simulation


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def _load_model_bundle(root: Path):
    model_path = root / "output" / "models" / "best_model.pkl"
    meta_path = root / "output" / "models" / "best_model_meta.json"
    scaler_path = root / "output" / "models" / "best_model_scaler.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found at {model_path}. Run src/training/train.py first."
        )
    model = joblib.load(model_path)
    scaler = None
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("has_scaler") and scaler_path.exists():
            scaler = joblib.load(scaler_path)
    if hasattr(model, "feature_names_in_"):
        feature_names = [str(c) for c in model.feature_names_in_]
    else:
        ref = root / "data" / "processed" / "processed_paysim.csv"
        if not ref.exists():
            raise FileNotFoundError(f"Need {ref} to infer feature columns.")
        head = pd.read_csv(ref, nrows=1)
        feature_names = [c for c in head.columns if c not in {"isFraud", "isFlaggedFraud"}]
    return model, scaler, feature_names


def _sample_base_transactions(
    data_path: Path,
    *,
    max_base_sample: int = 280,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(data_path)
    sample_n = min(max_base_sample, len(df))
    fraud_df = df[df["isFraud"] == 1]
    legit_df = df[df["isFraud"] == 0]
    fraud_take = min(len(fraud_df), max(35, sample_n // 6))
    legit_take = min(len(legit_df), max(sample_n - fraud_take, 1))
    sample_df = pd.concat(
        [
            fraud_df.sample(fraud_take, random_state=random_state),
            legit_df.sample(legit_take, random_state=random_state),
        ],
        ignore_index=True,
    ).sample(frac=1.0, random_state=random_state)
    y_sample = sample_df["isFraud"]
    X_sample = sample_df.drop(columns=["isFraud", "isFlaggedFraud"])
    return X_sample, y_sample


def _rows_to_event_logs(rows: list[dict], *, cap: int = 400) -> list[str]:
    lines: list[str] = []
    for i, r in enumerate(rows[:cap]):
        tid = f"tx-{i}"
        attack = r["attack_type"]
        p = r["probability"]
        dec = r["decision"]
        yt = r["true_fraud"]
        yp = r["pred_fraud"]
        if yt == yp:
            tag = "ok"
        elif yt == 0 and yp == 1:
            tag = "FP"
        else:
            tag = "FN"
        lines.append(
            f"{tid} | {attack} | p={p:.4f} | {dec} | "
            f"true={yt} pred={yp} | {tag}"
        )
    if len(rows) > cap:
        lines.append(f"... truncated {len(rows) - cap} more events")
    return lines


def _confusion_figure(cm: np.ndarray) -> go.Figure:
    z = cm.astype(int).tolist()
    labels = ["Legitimate (0)", "Fraud (1)"]
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=labels,
            y=labels,
            colorscale="Blues",
            text=z,
            texttemplate="%{text}",
            textfont={"size": 14},
            hovertemplate="Actual %{y}<br>Predicted %{x}<br>Count %{z}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Confusion matrix (rows = actual, columns = predicted)",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        margin=dict(l=60, r=20, t=60, b=60),
        height=360,
    )
    return fig


def create_app() -> Dash:
    root = _project_root()
    app = Dash(__name__)
    app.title = "Fraud detection — attack simulation"

    app.layout = html.Div(
        className="app",
        style={"fontFamily": "system-ui,sans-serif", "maxWidth": "1280px", "margin": "0 auto", "padding": "1rem"},
        children=[
            html.H1("Credit-card fraud — attack simulation dashboard"),
            html.P(
                "Click the button to flood attack scenarios through the trained model and "
                "the decision engine (BLOCK if P(fraud) ≥ threshold, else ALLOW)."
            ),
            html.Div(
                style={"display": "flex", "flexWrap": "wrap", "gap": "1rem", "alignItems": "center", "marginBottom": "1rem"},
                children=[
                    html.Button(
                        "Run attack simulator → model → decision engine",
                        id="run-btn",
                        n_clicks=0,
                        style={"padding": "0.6rem 1rem", "cursor": "pointer"},
                    ),
                    html.Label(
                        children=[
                            " Decision threshold ",
                            dcc.Input(
                                id="threshold-input",
                                type="number",
                                value=0.3,
                                min=0.01,
                                max=0.99,
                                step=0.01,
                                style={"width": "5rem", "marginLeft": "0.3rem"},
                            ),
                        ],
                        style={"display": "inline-flex", "alignItems": "center"},
                    ),
                    html.Label(
                        children=[
                            " Rapid-fire repeats ",
                            dcc.Input(
                                id="flood-input",
                                type="number",
                                value=8,
                                min=1,
                                max=40,
                                step=1,
                                style={"width": "4rem", "marginLeft": "0.3rem"},
                            ),
                        ],
                        style={"display": "inline-flex", "alignItems": "center"},
                    ),
                ],
            ),
            html.Div(id="status-msg", style={"marginBottom": "1rem", "color": "#333"}),
            dcc.Loading(
                id="loading",
                type="circle",
                children=[
                    html.Div(
                        style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "1.5rem"},
                        children=[
                            html.Div(
                                children=[
                                    html.H3("Predictions"),
                                    html.Div(id="predictions-table-wrap"),
                                ]
                            ),
                            html.Div(
                                children=[
                                    html.H3("Confusion matrix"),
                                    dcc.Graph(id="cm-graph", config={"displayModeBar": True}),
                                ]
                            ),
                        ],
                    ),
                    html.Div(
                        style={"marginTop": "1.5rem", "display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "1.5rem"},
                        children=[
                            html.Div(
                                children=[
                                    html.H3("False positives / false negatives"),
                                    html.Div(id="fpfn-metrics"),
                                ]
                            ),
                            html.Div(
                                children=[
                                    html.H3("Event log"),
                                    html.Pre(
                                        id="log-pre",
                                        style={
                                            "background": "#1e1e1e",
                                            "color": "#d4d4d4",
                                            "padding": "1rem",
                                            "borderRadius": "6px",
                                            "maxHeight": "360px",
                                            "overflowY": "auto",
                                            "fontSize": "12px",
                                        },
                                    ),
                                ]
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
        Input("run-btn", "n_clicks"),
        State("threshold-input", "value"),
        State("flood-input", "value"),
        prevent_initial_call=True,
    )
    def on_run(_n_clicks, threshold, flood_repeats):
        try:
            thr = float(threshold) if threshold is not None else 0.3
            flood = int(flood_repeats) if flood_repeats is not None else 8
            flood = max(1, min(flood, 99))
        except (TypeError, ValueError):
            return ("Invalid threshold or flood repeats.", html.Div(), go.Figure(), html.Div(), "")

        data_path = root / "data" / "processed" / "processed_paysim.csv"
        if not data_path.exists():
            msg = f"Data file missing: {data_path}"
            return (msg, html.Div(), go.Figure(), html.Div(), msg)

        try:
            model, scaler, feature_names = _load_model_bundle(root)
            X_sample, y_sample = _sample_base_transactions(data_path)
            summary = run_simulation(
                model,
                X_sample,
                y_sample,
                feature_names,
                threshold=thr,
                flood_repeats=flood,
                scaler=scaler,
                max_explanations=0,
            )
        except Exception as exc:  # noqa: BLE001 — show errors in UI
            err = f"Run failed: {exc}"
            return (err, html.Div(), go.Figure(), html.Div(), err)

        ocm = summary["overall"]["confusion_matrix"]
        oc = summary["overall"]["counts"]
        n_tx = summary["overall"]["total_transactions"]

        status = (
            f"Scored {n_tx} transactions across all scenarios "
            f"(threshold={summary['threshold']}). Per-scenario matrices are in the event log header."
        )

        flat = summary["flat_results"]
        df_pred = pd.DataFrame(
            {
                "attack": [r["attack_type"] for r in flat],
                "P(fraud)": [round(r["probability"], 4) for r in flat],
                "decision": [r["decision"] for r in flat],
                "true": [r["true_fraud"] for r in flat],
                "pred": [r["pred_fraud"] for r in flat],
            }
        )
        display_n = min(500, len(df_pred))
        table = DataTable(
            columns=[{"name": c, "id": c} for c in df_pred.columns],
            data=df_pred.head(display_n).to_dict("records"),
            page_size=15,
            style_table={"overflowX": "auto"},
            style_cell={"textAlign": "left", "padding": "6px", "fontSize": "13px"},
            style_header={"fontWeight": "bold"},
        )
        wrap = html.Div(
            children=[
                table,
                html.P(f"Showing first {display_n} of {len(df_pred)} rows.", style={"fontSize": "12px", "color": "#666"}),
            ]
        )

        fig = _confusion_figure(ocm)

        fpfn = html.Ul(
            [
                html.Li([html.Strong("False positives (FP): "), f"{oc['FP']} — legitimate flagged as fraud"]),
                html.Li([html.Strong("False negatives (FN): "), f"{oc['FN']} — fraud missed"]),
                html.Li([html.Strong("True positives (TP): "), str(oc["TP"])]),
                html.Li([html.Strong("True negatives (TN): "), str(oc["TN"])]),
            ]
        )

        header_lines = [
            "=== pipeline log: attack → model → decision_engine ===",
            f"threshold={summary['threshold']} | total_tx={n_tx}",
        ]
        for block in summary["per_attack"]:
            c = block["counts"]
            header_lines.append(
                f"[{block['attack_name']}] n={len(block['y_true'])} "
                f"TP={c['TP']} TN={c['TN']} FP={c['FP']} FN={c['FN']}"
            )
        header_lines.append("--- events ---")
        log_body = _rows_to_event_logs(flat, cap=500)
        log_text = "\n".join(header_lines + log_body)

        return status, wrap, fig, fpfn, log_text

    return app


def main() -> None:
    app = create_app()
    app.run(debug=True, host="127.0.0.1", port=8050)


if __name__ == "__main__":
    main()
