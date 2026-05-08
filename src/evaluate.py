from src.cost import compute_cost

def evaluate_model(X_train, X_test, y_train, y_test, w, b):
    train_mse = compute_cost(X_train, y_train, w, b)
    test_mse = compute_cost(X_test, y_test, w, b)

    print(f"MSE for training: {train_mse}")
    print(f"MSE for testing: {test_mse}")
