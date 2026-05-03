import pandas as pd

def preprocess_data():
    df = pd.read_csv("data/dataset.csv")
    x = df.drop(["student_id", "participation_score", "assignment_score", "midterm_score", "final_exam_score", "overall_score", "grade"], axis=1)
    y = df[["overall_score"]]
    x = pd.get_dummies(x, dtype=int)
    return x, y

