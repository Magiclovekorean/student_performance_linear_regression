import pandas as pd

def split_data(X: pd.DataFrame, y: pd.Series, train_ratio: float = 0.8): 
    train_size = int(train_ratio * len(X))

    X_train = X.iloc[:train_size]
    y_train = y.iloc[:train_size]
    X_test = X.iloc[train_size:]
    y_test = y.iloc[train_size:]

    return X_train, X_test, y_train, y_test
