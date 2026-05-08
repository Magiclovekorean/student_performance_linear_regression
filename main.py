import numpy as np

from config import LEARNING_RATE, TRAIN_RATIO, GRADIENT_DESCENT_ITERS
from src.preprocessing import preprocess_data
from src.split import split_data
from src.train import train_model
from src.evaluate import evaluate_model

def main():
    X, y = preprocess_data()
    X_train, X_test, y_train, y_test = split_data(X, y, TRAIN_RATIO)
    initial_w = np.zeros(X_train.shape[1])
    initial_b = 0
    print("Starting training")
    w, b = train_model(X_train, initial_w, initial_b, y_train, LEARNING_RATE, GRADIENT_DESCENT_ITERS)
    print("Evaluating model")
    evaluate_model(X_train, X_test, y_train, y_test, w, b)

if __name__ == "__main__":
    main()
