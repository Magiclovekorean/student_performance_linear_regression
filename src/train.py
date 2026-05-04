import numpy as np
from src.model import predict_value

def train_model(x, w, b, y, alpha, iterations):
    for i in range(iterations):
        dj_dw, dj_db = compute_gradient(x, w, b, y)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        

    f_wb = x * w + b
    error = f_wb - y
    return error
    

def compute_gradient(x, w, b, y):
    f_wb = x * w + b
    error = f_wb - y
    
    dj_dw  = np.sum(error * x)
    dj_db  = np.sum(error)

    return dj_dw, dj_db
