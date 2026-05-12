# main.py
from data_cleaning import load_data, inspect_data, clean_data
from feature_engineering import engineer_features
from preprocessing import preprocess_pipeline
from model_training import run_training_pipeline

def main():
    print("Starting Credit Card Fraud Detection Pipeline...")
    
    # 1. Data Cleaning
    print("Step 1: Loading and Cleaning Data...")
    data = load_data()
    inspect_data(data) # Optional for inspection
    data = clean_data(data)
    
    # 2. Feature Engineering
    print("Step 2: Engineering Features...")
    data = engineer_features(data)
    
    # 3. Preprocessing (Split, SMOTE, Scale)
    print("Step 3: Preprocessing (Split, SMOTE, Scale)...")
    X_train, X_test, y_train, y_test = preprocess_pipeline(data)
    
    # 4. Training & Evaluation
    print("Step 4: Training and Evaluating Model...")
    model = run_training_pipeline(X_train, X_test, y_train, y_test)
    
    print("Pipeline Finished Successfully!")

if __name__ == "__main__":
    main()