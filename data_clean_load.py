import pandas as pd

def load_data(path="paysim.csv"):
    """Load transaction data"""
    data = pd.read_csv(path)
    return data

def inspect_data(data):
    """Initial inspection of data (shape, types, missing values)"""
    print("Data Shape:", data.shape)
    print("\nData Info:")
    data.info()
    print("\nMissing Values:\n", data.isna().sum())
    print("\nDuplicate Rows:", data.duplicated().sum())
    return data

def clean_data(data):
    """Clean data (remove duplicates and missing values if any)"""
    data.drop_duplicates(inplace=True)
    data.dropna(inplace=True)
    return data