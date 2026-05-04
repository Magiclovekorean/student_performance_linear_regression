def split_data(X, y, train_ratio: float = 0.8):
    train_size = int(train_ratio * len(X))

    X_train = X[:train_size]
    y_train = y[:train_size]
    X_test = X[train_size:]
    y_test = y[train_size:]

    return X_train, X_test, y_train, y_test
