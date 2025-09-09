# main.py
# Predict exam_score with Linear Regression

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
CSV_PATH = "student_exam_scores.csv"

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Drop Unused
    df.drop(columns=["student_id"])
    return df

def main():
    df = load_data(CSV_PATH)
    assert "exam_score" in df.columns, "CSV must include 'exam_score'."

    # Features: all numeric columns except exam score
    X = df.drop(columns=["exam_score"])
    X = X.select_dtypes(include=[np.number])
    y = df["exam_score"].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE
    )

    # Regression Model
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LinearRegression())
    ])
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print("LinearRegression")
    print(f"Rows: {len(df)} | Features used: {X.shape[1]}")
    print("Target: exam_score\n")
    print(f"MAE : {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R^2 : {r2:.3f}")

    # Show coefficients
    lr = model.named_steps["lr"]
    scaler = model.named_steps["scaler"]
    coef = lr.coef_
    # Original feature names
    coef_table = pd.DataFrame({
        "feature": X.columns,
        "coefficient": coef
    }).sort_values("coefficient", key=abs, ascending=False)
    print("\nTop coefficients:")
    print(coef_table.to_string(index=False))

    # Classification (is_high_score if >= median threshold)
    thr = float(y.median())
    y_clf = (y >= thr).astype(int)

    Xc_train, Xc_test, yc_train, yc_test = train_test_split(
        X, y_clf, test_size=0.25, random_state=RANDOM_STATE, stratify=y_clf
    )

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(solver="liblinear", random_state=RANDOM_STATE, max_iter=1000))
    ])
    clf.fit(Xc_train, yc_train)

    c_probs = clf.predict_proba(Xc_test)[:, 1]
    c_preds = (c_probs >= 0.5).astype(int)

    c_acc  = accuracy_score(yc_test, c_preds)
    c_prec = precision_score(yc_test, c_preds, zero_division=0)
    c_rec  = recall_score(yc_test, c_preds, zero_division=0)
    c_f1   = f1_score(yc_test, c_preds, zero_division=0)
    try:
        c_auc = roc_auc_score(yc_test, c_probs)
    except ValueError:
        c_auc = float("nan")

    print("\nLogisticRegression")
    print(f"Threshold: exam_score >= {thr:.2f}\n")
    print(f"Accuracy : {c_acc:.3f}")
    print(f"Precision: {c_prec:.3f}")
    print(f"Recall   : {c_rec:.3f}")
    print(f"F1       : {c_f1:.3f}")
    print("ROC_AUC  :", "n/a" if np.isnan(c_auc) else f"{c_auc:.3f}")

    # Coefficients (standardized space)
    logreg = clf.named_steps["logreg"]
    clf_coef = pd.DataFrame({
        "feature": X.columns,
        "coefficient": logreg.coef_.ravel()
    }).sort_values("coefficient", key=np.abs, ascending=False)
    print("\nTop classification coefficients:")
    print(clf_coef.to_string(index=False))

if __name__ == "__main__":
    main()
