# Student Exam Scores — Two Regression Models (Linear vs Polynomial)

Predict student **exam scores** from simple study/attendance features.

This project trains **two regression models** on the same train/test split and compares them:

1) **Linear Regression** - fast, interpretable baseline; good when relationships are close to straight-line.
2) **Polynomial Regression** - adds squared interaction for curvature; can help if relationships are non-linear, but may overfit on small datasets.


## Dataset

**File:** `student_exam_scores.csv` (repo root)

Columns:
- `student_id` *(identifier — dropped)*
- `hours_studied` *(numeric)*
- `sleep_hours` *(numeric)*
- `attendance_percent` *(numeric)*
- `previous_scores` *(numeric)*
- `exam_score` *(numeric target)*

The dataset is clean and fully numeric, easy to use!


## Running It

```bash
pip install -U numpy pandas scikit-learn
python main.py
```

## What the script does:

- Drops student_id if present.
- Splits data once (75/25) with a fixed random_state for a fair, reproducible comparison.
- Trains Linear Regression (with scaling).
- Trains Polynomial Regression (degree=2) (expand → scale → linear fit).
- Prints MAE, RMSE, R² for both models and shows the most-influential coefficients.

## Metrics

- **MAE** (Mean Absolute Error): average miss in exam points. Lower is better.
- **RMSE** (Root Mean Squared Error): like MAE but punishes large misses more. Same units (points). Lower is better.
- **R²** how much of the ups and downs in scores the model explains. 1.0 is perfect; 0.0 means “just predict the average.”

## Test Results 
Using the current repo code:

```bash
LinearRegression

MAE : 2.316
RMSE: 2.759
R^2 : 0.871
```

```bash
PolynomialRegression

MAE : 2.429
RMSE: 2.908
R^2 : 0.857
```

## What these numbers mean

LinearRegression performs better on this dataset.
- Lower MAE (2.316 vs 2.429)
- Lower RMSE (2.759 vs 2.908)
- Higher R² (0.871 vs 0.857)

Pick Linear Regression for this dataset. It has lower errors and higher R² on the test set.

For these features, the relationship to exam_score is largely linear. 

Adding curvature (degree-2 polynomial) did not help the model generalize on the test set; it added complexity without improving accuracy.
