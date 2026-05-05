import numpy as np
from src.model import predict_value

def train_model(X, w, b, y, alpha, iterations):
    for i in range(iterations):
        dj_dw, dj_db = compute_gradient(X, w, b, y)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        print(i)
        

    f_wb = predict_value(X, w, b)
    error = f_wb - y
    return error
    

def compute_gradient(X, w, b, y):
    f_wb = predict_value(X, w, b)
    error = f_wb - y

    m = X.shape[0]
    dj_dw  = X.T @ error / m
    dj_db  = np.sum(error) / m

    return dj_dw, dj_db
