from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

def train_model(X_train, y_train):
    """Train Logistic Regression model"""
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    return lr

def evaluate_model(model, X_test, y_test):
    """Evaluate model and print reports"""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    print("Logistic Regression Results:")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("ROC-AUC Score:", roc_auc_score(y_test, y_prob))
    
    return y_prob

def tune_threshold(y_prob, y_test):
    """Test different thresholds to improve precision/recall trade-off"""
    print("\n--- Threshold Tuning ---")
    for t in [0.3, 0.4, 0.5, 0.6, 0.7]:
        y_pred_t = (y_prob > t).astype(int)
        print(f"\nThreshold = {t}")
        print(classification_report(y_test, y_pred_t))

def run_training_pipeline(X_train, X_test, y_train, y_test):
    """Main function for training and evaluation"""
    model = train_model(X_train, y_train)
    y_prob = evaluate_model(model, X_test, y_test)
    tune_threshold(y_prob, y_test)
    return model