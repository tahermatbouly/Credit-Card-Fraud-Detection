import joblib

from src.training.train import train_pipeline


def main():

    print("\n===================================")
    print("🚀 CREDIT CARD FRAUD ML SYSTEM")
    print("===================================\n")

    # =========================
    # TRAIN ALL MODELS
    # =========================
    best_model, results = train_pipeline()

    # =========================
    # SAVE BEST MODEL
    # =========================
    model_name = best_model[0]
    model_obj = best_model[1]

    joblib.dump(model_obj, "models/best_model.pkl")

    print("\n💾 Best model saved as: models/best_model.pkl")

    # =========================
    # FINAL REPORT
    # =========================
    print("\n===================================")
    print("📊 FINAL MODEL COMPARISON")
    print("===================================\n")

    for r in results:
        print(f"Model: {r['model']}")
        print(f"Precision: {r['precision']:.4f}")
        print(f"Recall:    {r['recall']:.4f}")
        print(f"F1 Score:  {r['f1']:.4f}")
        print(f"ROC-AUC:   {r['roc_auc']:.4f}")
        print("-----------------------------------")

    print("\n🏆 BEST MODEL SELECTED:")
    print(f"Model Name: {model_name}")
    print(f"ROC-AUC: {max([r['roc_auc'] for r in results]):.4f}")

    print("\n✅ Training pipeline completed successfully!")


if __name__ == "__main__":
    main()