def split_features(df):
    X = df.drop(['isFraud', 'isFlaggedFraud'], axis=1)
    y = df['isFraud']
    return X, y