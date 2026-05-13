from sklearn.model_selection import train_test_split

from src.data.load_data import load_processed_data
from src.features.split_features import split_features

from src.models.logistic_regression import run_model as lr_model
from src.models.random_forest import run_model as rf_model
from src.models.xg_boost import run_model as xgb_model


def main():

    df = load_processed_data()
    X, y = split_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    results = []

    # Run all models
    results.append(lr_model(X_train, X_test, y_train, y_test))
    results.append(rf_model(X_train, X_test, y_train, y_test))
    results.append(xgb_model(X_train, X_test, y_train, y_test))

    # Show results
    print("\n================ MODEL RESULTS ================\n")

    for r in results:
        print(r["model_name"])
        print("Accuracy:", r["accuracy"])
        print("F1:", r["f1"])
        print("ROC-AUC:", r["roc_auc"])
        print("--------------------------------")

    # Best model
    best = max(results, key=lambda x: x["roc_auc"])

    print("\n================ BEST MODEL ================")
    print(best["model_name"])
    print("ROC-AUC:", best["roc_auc"])


if __name__ == "__main__":
    main()