# Student Exam Performance
Predict student exam performance from student habit features:
The project implements two models on the same dataset:
1. **Regression** — predict `exam_score`
2. **Classification** — predict whether a student is a high scorer (derived from `exam_score`)

## Dataset

**File:** `student_exam_scores.csv` (included in repo)

Columns:
- `student_id` *(dropped as unused data)*
- `hours_studied` *(numeric)*
- `sleep_hours` *(numeric)*
- `attendance_percent` *(numeric)*
- `previous_scores` *(numeric)*
- `exam_score` *(target)*

The features are clean and numeric, easy to use dataset.

