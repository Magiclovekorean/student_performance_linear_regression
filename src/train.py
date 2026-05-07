import numpy as np
from src.model import predict_value
from src.cost import compute_cost

def train_model(X, w, b, y, alpha, iterations):
    for i in range(iterations):
        dj_dw, dj_db = compute_gradient(X, w, b, y)

        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        cost = compute_cost(X, y, w, b)
        if i % 10000 == 0 and i > 0:
            print(f"Iteration {i:4}, Cost {cost:.4f}")
    print(f"Finished gradient descent with cost {cost:.4f}, w {w} and b {b}")


    return w, b
    

def compute_gradient(X, w, b, y):
    f_wb = predict_value(X, w, b)
    error = f_wb - y

    m = X.shape[0]
    dj_dw  = X.T @ error / m
    dj_db  = np.sum(error) / m

    return dj_dw, dj_db

