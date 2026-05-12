import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

def split_data(data):
    """Split independent and dependent variables and split data"""
    X = data.drop(['isFraud', 'isFlaggedFraud'], axis=1)
    y = data['isFraud']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )
    return X_train, X_test, y_train, y_test

def handle_imbalance(X_train, y_train):
    """Handle class imbalance using SMOTE"""
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    return X_train_res, y_train_res

def scale_data(X_train_res, X_test):
    """Standardize data"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_res)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

def preprocess_pipeline(data):
    """Main function for preprocessing steps"""
    X_train, X_test, y_train, y_test = split_data(data)
    X_train_res, y_train_res = handle_imbalance(X_train, y_train)
    X_train_scaled, X_test_scaled = scale_data(X_train_res, X_test)
    
    return X_train_scaled, X_test_scaled, y_train_res, y_test