# 🚨 Credit Card Fraud Detection System

A machine learning-based fraud detection system that trains multiple models, simulates fraud attacks, evaluates performance under stress conditions, and provides real-time fraud detection with decision-making and explainability.

---

# 🧠 Project Overview

This project goes beyond a traditional ML classifier. It is designed as a **complete fraud detection pipeline**, including:

- Multiple ML models (Logistic Regression, Random Forest, XGBoost)
- Automated best model selection
- Fraud attack simulation engine
- Decision-making system (ALLOW / REVIEW / BLOCK)
- Explainability layer for predictions
- Real-time simulation engine
- Evaluation using confusion matrix and classification metrics

---

# 🏗️ System Architecture
data/
├── raw/
└── processed/

src/
├── data/ # Data loading
├── features/ # Feature engineering
├── models/ # ML models
├── evaluation/ # Evaluation & metrics
├── simulation/ # Attack simulation system
├── decision/ # Decision engine
├── explain/ # Explainability layer
├── training/ # Training pipeline

dashboard/
├── Dash application (control panel)

output/
├── models/ # Saved best model + metadata


---

# 🚀 Features

## 🤖 Machine Learning Models
- Logistic Regression
- Random Forest
- XGBoost
- Automatic model selection using ROC-AUC

---

## ⚖️ Model Evaluation
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix (TP / TN / FP / FN)

---

## 💣 Fraud Attack Simulation
The system simulates realistic fraud scenarios such as:

- High-value transaction attacks
- Rapid transaction bursts
- Synthetic fraud patterns
- Normal vs malicious behavior comparison

---

## 🧠 Decision Engine
Transforms predictions into actions:

| Risk Level | Action |
|------------|--------|
| Low Risk | ALLOW |
| Medium Risk | REVIEW |
| High Risk | BLOCK |

---

## 🧾 Explainability Layer
Explains fraud decisions based on features:

- High transaction amount
- Balance inconsistencies
- Suspicious behavior patterns

---

## 🔄 Real-Time Simulation
- Continuous transaction stream simulation
- Live fraud detection pipeline
- Real-time decision output

---

## 📊 Dashboard (Dash)
Interactive control panel including:

- Run attack simulation button
- Live fraud predictions
- Confusion matrix visualization
- Model comparison metrics
- System performance monitoring

---

# 🏆 Model Selection Strategy

The best model is selected based on:

- ROC-AUC Score (primary metric)

Saved as:
output/models/best_model.pkl
output/models/best_model_meta.json


---

# ⚙️ Installation

```bash
git clone <repo-url>
cd Credit-Card-Fraud-Detection

pip install -r requirements.txt
```

📦 Requirements
pandas
numpy
scikit-learn
xgboost
imblearn
joblib
matplotlib
seaborn
dash


1️⃣ Train Models
python -m src.training.train
2️⃣ Run Simulation
python run_simulation.py
3️⃣ Run Dashboard (if enabled)
python dashboard/app.py
🧪 Simulation System

The system supports:

Normal transactions
Fraud attack generation
Flood simulation (large-scale testing)
Mixed traffic scenarios

Outputs:

Fraud probability
Decision (ALLOW / REVIEW / BLOCK)
Explanation
Confusion matrix metrics
📊 Example Output
BEST MODEL: Random Forest
ROC-AUC: 0.99987

True Positives: 124
False Positives: 18
True Negatives: 4801
False Negatives: 3
🧠 Key Insight

This system is not just a classifier — it is a:

Fraud detection simulation and decision-making engine

It replicates real-world fraud monitoring systems used in financial institutions.

🔥 Future Improvements
SHAP explainability integration
Real-time API (FastAPI)
Kafka streaming pipeline
Fraud drift detection system
Database logging (PostgreSQL)
Authentication dashboard
👨‍💻 Author

Built as a machine learning + systems engineering project focused on fraud detection, simulation, and real-time decision systems.
