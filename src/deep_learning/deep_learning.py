"""
Deep Neural Network for Financial Fraud Detection
Using Preprocessed PaySim Dataset

Author: Taher
"""

# ============================================
# IMPORTS
# ============================================
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)

from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam


# ============================================
# LOAD PREPROCESSED DATASET
# ============================================
DATA_PATH = "data/processed/processed_paysim.csv"

print("Loading preprocessed dataset...")
df = pd.read_csv(DATA_PATH)

print(df.head())
print(df.shape)


# ============================================
# FEATURES / TARGET
# ============================================
X = df.drop("isFraud", axis=1)
y = df["isFraud"]

print("\nFeature shape:", X.shape)


# ============================================
# TRAIN / TEST SPLIT
# ============================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ============================================
# FEATURE SCALING (IMPORTANT FOR DNN)
# ============================================
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ============================================
# CLASS IMBALANCE HANDLING
# ============================================
classes = np.unique(y_train)

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weights = dict(enumerate(class_weights))

print("\nClass weights:", class_weights)


# ============================================
# BUILD DEEP NEURAL NETWORK
# ============================================
model = Sequential([

    Dense(128, activation="relu", input_shape=(X_train_scaled.shape[1],)),
    BatchNormalization(),
    Dropout(0.3),

    Dense(64, activation="relu"),
    BatchNormalization(),
    Dropout(0.3),

    Dense(32, activation="relu"),
    BatchNormalization(),
    Dropout(0.2),

    Dense(16, activation="relu"),

    Dense(1, activation="sigmoid")
])


# ============================================
# COMPILE MODEL
# ============================================
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
    ]
)


model.summary()


# ============================================
# EARLY STOPPING
# ============================================
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)


# ============================================
# TRAIN MODEL
# ============================================
print("\nTraining model...\n")

history = model.fit(
    X_train_scaled,
    y_train,
    validation_split=0.2,
    epochs=30,
    batch_size=1024,
    class_weight=class_weights,
    callbacks=[early_stopping],
    verbose=1
)


# ============================================
# EVALUATION
# ============================================
print("\nEvaluating model...\n")

y_prob = model.predict(X_test_scaled)
y_pred = (y_prob > 0.5).astype(int)


print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))


print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))


# ============================================
# SAVE MODEL
# ============================================
model.save("models/dnn_fraud_model.h5")

print("\nModel saved successfully!")