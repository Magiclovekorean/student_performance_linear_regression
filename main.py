from src.preprocessing import preprocess_data
from src.split import split_data
from config import LEARNING_RATE, TRAIN_RATIO

def main():
    X, y = preprocess_data()
    X_train, X_test, y_train, y_test = split_data(X, y, TRAIN_RATIO)

if __name__ == "__main__":
    main()
