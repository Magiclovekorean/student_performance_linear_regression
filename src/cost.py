import numpy as np

def compute_cost(X, y, w, b):
    m = X.shape[0]
    
    f_wb = X @ w + b        
    error = f_wb - y        
    cost = np.sum(error**2)
    
    total_cost = cost / (2 * m)
    
    return total_cost
