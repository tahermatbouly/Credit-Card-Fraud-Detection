import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


def evaluate_model(model_name, model, X_test, y_test):

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model_name": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob)
    }

    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    return {
        "metrics": metrics,
        "confusion_matrix": cm,
        "roc": (fpr, tpr, metrics["roc_auc"])
    }
    
def show_final_visualization(results, cms, roc_data):

    import matplotlib.pyplot as plt
    import seaborn as sns

    # ================= BEST MODEL =================
    best = max(results, key=lambda x: x["roc_auc"])

    print("\nBEST MODEL:", best["model_name"])
    print("ROC-AUC:", best["roc_auc"])

    # ================= CONFUSION MATRICES =================
    n = len(cms)
    
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    
    plt.figure(figsize=(5 * cols, 4 * rows))
    
    for i, (name, cm) in enumerate(cms):
    
        plt.subplot(rows, cols, i + 1)
    
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False
        )
    
        plt.title(name)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
    
    plt.tight_layout()
    plt.show()
    # ================= ROC CURVES =================
    plt.figure(figsize=(8, 6))

    for name, (fpr, tpr, auc) in roc_data:
        plt.plot(
            fpr,
            tpr,
              label=f"{name} (AUC={auc:.2f})"
    )

        # baseline (IMPORTANT: also label it)
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Random Classifier")

    plt.title("ROC Curve Comparison")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.legend()
    plt.show()

    # ================= METRICS BAR =================
    names = [r["model_name"] for r in results]
    roc = [r["roc_auc"] for r in results]

    plt.bar(names, roc)
    plt.title("ROC-AUC Comparison")
    plt.show()