import pandas as pd

def preprocess_data():
    df = pd.read_csv("data/dataset.csv")
    X = df.drop(["student_id", "participation_score", "assignment_score", "midterm_score", "final_exam_score", "overall_score", "grade"], axis=1)
    y = df["overall_score"]
    X = pd.get_dummies(X, dtype=int)
    X = X.to_numpy(dtype=float)
    y = y.to_numpy(dtype=float)
    return X, y

