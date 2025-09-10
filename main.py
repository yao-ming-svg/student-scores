# main.py
# Predict exam_score with Linear Regression

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

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
    mse = mean_squared_error(y_test, preds)
    rmse = float(np.sqrt(mse))

    r2 = r2_score(y_test, preds)

    print("LinearRegression")
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

    # Polynomial Model
    poly_model = Pipeline([
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("scaler", StandardScaler()),
        ("lr", LinearRegression())
    ])
    poly_model.fit(X_train, y_train)
    poly_preds = poly_model.predict(X_test)

    poly_mae = mean_absolute_error(y_test, poly_preds)
    poly_mse = mean_squared_error(y_test, poly_preds)
    poly_rmse = float(np.sqrt(poly_mse))

    poly_r2 = r2_score(y_test, poly_preds)

    print("\nPolynomialRegression")
    print(f"MAE : {poly_mae:.3f}")
    print(f"RMSE: {poly_rmse:.3f}")
    print(f"R^2 : {poly_r2:.3f}")

    # Show coefficients
    poly_lr = poly_model.named_steps["lr"]
    poly_step = poly_model.named_steps["poly"]
    n_out = poly_step.n_output_features_
    poly_feat_names = [f"f{i}" for i in range(n_out)]

    poly_coef = poly_lr.coef_
    poly_coef_table = pd.DataFrame({
        "feature": poly_feat_names,
        "coefficient": poly_coef
    }).sort_values("coefficient", key=abs, ascending=False)

    print("\nTop coefficients:")
    print(poly_coef_table.head(12).to_string(index=False))

if __name__ == "__main__":
    main()
