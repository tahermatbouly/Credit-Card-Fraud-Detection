import numpy as np
import pandas as pd

def create_balance_features(data):
    """Create balance difference features"""
    data['balance_diff_orig'] = data['newbalanceOrig'] - data['oldbalanceOrg']
    data['balance_diff_dest'] = data['newbalanceDest'] - data['oldbalanceDest']
    return data

def encode_transaction_type(data):
    """Convert transaction type to numeric columns (One-Hot Encoding)"""
    data = pd.get_dummies(data, columns=['type'], drop_first=True)
    # Convert resulting columns from Boolean to Int
    type_cols = [col for col in data.columns if col.startswith('type_')]
    data[type_cols] = data[type_cols].astype(int)
    return data

def create_amount_features(data):
    """Create amount-related features (Log Transform & High Amount Flag)"""
    mean_amount = data['amount'].mean()
    data['is_high_amount'] = (data['amount'] > mean_amount).astype(int)
    data['amount_log'] = np.log1p(data['amount'])
    return data

def create_error_features(data):
    """Create error features in balance (to detect manipulation)"""
    data['error_orig'] = abs(data['oldbalanceOrg'] - data['newbalanceOrig'] - data['amount'])
    data['error_dest'] = abs(data['newbalanceDest'] - data['oldbalanceDest'] - data['amount'])
    return data

def create_risk_features(data):
    """Create high-risk transaction feature"""
    if 'type_TRANSFER' in data.columns and 'is_high_amount' in data.columns:
        data['high_risk_transaction'] = ((data['type_TRANSFER'] == 1) & (data['is_high_amount'] == 1)).astype(int)
    return data

def drop_irrelevant_columns(data):
    """Drop unused columns for training"""
    data.drop(['nameOrig', 'nameDest'], axis=1, inplace=True)
    return data

def engineer_features(data):
    """Main function gathering all engineering steps"""
    data = create_balance_features(data)
    data = encode_transaction_type(data)
    data = create_amount_features(data)
    data = create_error_features(data)
    data = create_risk_features(data)
    data = drop_irrelevant_columns(data)
    return data