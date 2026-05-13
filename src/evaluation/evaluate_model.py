import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)


# =========================
# Store results globally for comparison
# =========================
all_models_results = []


def evaluate_model(name, model, X_test, y_test):

    y_pred = model.predict(X_test)

    # some models support predict_proba
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    # =========================
    # Metrics
    # =========================
    metrics = {
        "model_name": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob) if y_prob is not None else None
    }

    all_models_results.append(metrics)

    # =========================
    # Confusion Matrix Plot
    # =========================
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    # =========================
    # ROC Curve
    # =========================
    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC={metrics['roc_auc']:.2f})")

    return metrics


def plot_model_comparison():

    if not all_models_results:
        return

    names = [m["model_name"] for m in all_models_results]
    roc = [m["roc_auc"] for m in all_models_results]
    f1 = [m["f1"] for m in all_models_results]
    recall = [m["recall"] for m in all_models_results]

    x = range(len(names))

    plt.figure(figsize=(10, 5))
    plt.bar(x, roc, label="ROC-AUC")
    plt.bar(x, f1, label="F1")
    plt.bar(x, recall, label="Recall")

    plt.xticks(x, names)
    plt.title("Model Comparison")
    plt.legend()
    plt.show()