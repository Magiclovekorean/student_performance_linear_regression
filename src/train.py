import numpy as np
from src.model import predict_value

def train_model(X, w, b, y, alpha, iterations):
    for i in range(iterations):
        dj_dw, dj_db = compute_gradient(X, w, b, y)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        f_wb = predict_value(X, w, b)
        m = X.shape[0]
        cost = (1/m) * f_wb - y
        if i % 10000 == 0 and i > 0:
            print(f"Iteration {i:4}, Cost {cost:4}")
    print(10000)

    return w, b
    

def compute_gradient(X, w, b, y):
    f_wb = predict_value(X, w, b)
    error = f_wb - y

    m = X.shape[0]
    dj_dw  = X.T @ error / m
    dj_db  = np.sum(error) / m

    return dj_dw, dj_db
