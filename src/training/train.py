from sklearn.model_selection import train_test_split

from src.data.load_data import load_processed_data
from src.features.split_features import split_features

from src.models.logistic_regression import run_model as lr_model
from src.models.random_forest import run_model as rf_model
from src.models.xg_boost import run_model as xgb_model

# from src.evaluation.evaluate_model import plot_model_comparison
from src.evaluation.evaluate_model import evaluate_model, show_final_visualization

results = []
cms = []
roc_data = []

def main():

    # =========================
    # Load + split data
    # =========================
    df = load_processed_data()
    X, y = split_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # =========================
    # Run models
    # =========================
    model_outputs = [
        lr_model(X_train, X_test, y_train, y_test),
        rf_model(X_train, X_test, y_train, y_test),
        xgb_model(X_train, X_test, y_train, y_test)
    ]

    results = []
    cms = []
    roc_data = []

    for output in model_outputs:

        results.append({
        "model_name": output["model_name"],
        "accuracy": output["accuracy"],
        "f1": output["f1"],
        "roc_auc": output["roc_auc"]
    })


        cms.append((output["model_name"], output["confusion_matrix"]))

        fpr, tpr, auc = output["roc"]
        roc_data.append((output["model_name"], (fpr, tpr, auc)))
    # =========================
    # Print results
    # =========================
    print("\n================ MODEL RESULTS ================\n")

    for r in results:
        print(f"Model: {r['model_name']}")
        print(f"Accuracy: {r['accuracy']:.4f}")
        print(f"F1: {r['f1']:.4f}")
        print(f"ROC-AUC: {r['roc_auc']:.4f}")
        print("--------------------------------")

    # =========================
    # Best model selection
    # =========================
    best = max(results, key=lambda x: x["roc_auc"])

    print("\n================ BEST MODEL ================")
    print(f"Model: {best['model_name']}")
    print(f"ROC-AUC: {best['roc_auc']:.4f}")

    # =========================
    # Visualization
    # =========================
    # plot_model_comparison()
    show_final_visualization(results, cms, roc_data)


if __name__ == "__main__":
    main()