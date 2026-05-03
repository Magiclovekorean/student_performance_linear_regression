from src.preprocessing import preprocess_data
from src.split import split_data

def main():
    X, y = preprocess_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

if __name__ == "__main__":
    main()
