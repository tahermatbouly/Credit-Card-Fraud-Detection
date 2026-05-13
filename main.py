from src.training.train import train_pipeline


def main():
    print("\n===================================")
    print("🚀 CREDIT CARD FRAUD ML PIPELINE")
    print("===================================\n")

    best_model, results = train_pipeline()

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
    print(f"Model Name: {best_model[0]}")
    print(best_model[1])

    print("\n✅ Pipeline execution completed successfully!")


if __name__ == "__main__":
    main()