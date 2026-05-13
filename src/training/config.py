# =========================
# General Settings
# =========================

RANDOM_STATE = 42
TEST_SIZE = 0.2


# =========================
# Imbalance Handling
# =========================

SMOTE_ENABLED = True
SMOTE_RANDOM_STATE = 42


# =========================
# Evaluation Settings
# =========================

THRESHOLD = 0.75   # can be tuned later for fraud detection


# =========================
# Logistic Regression Params
# =========================

LOGISTIC_PARAMS = {
    "max_iter": 1000
}


# =========================
# Random Forest Params
# =========================

RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": 15,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "class_weight": "balanced_subsample"
}


# =========================
# XGBoost Params
# =========================

XGB_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "logloss",
    "random_state": RANDOM_STATE,
    "n_jobs": -1
}