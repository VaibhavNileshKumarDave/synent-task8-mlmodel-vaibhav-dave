# Task 8 — Machine Learning Model (Student Performance)
### Synent Technologies Data Analyst Internship

---

## Problem Statement

Can we predict a student's math score before their exam — and if yes, what factors matter most? This project builds and compares two regression models (Linear Regression and Random Forest Regressor) on student performance data to answer both questions. The goal is not just to build a model, but to surface actionable insights that school administrators can actually use to improve outcomes.

---

## Dataset Details

| Property | Value |
|---|---|
| File | `StudentsPerformance.csv` |
| Source | Kaggle — Students Performance in Exams |
| Link | https://www.kaggle.com/datasets/spscientist/students-performance-in-exams |
| Rows | 1,000 |
| Columns | 8 |

**Columns:**

| Column | Type | Role |
|---|---|---|
| `gender` | Categorical | Feature |
| `race/ethnicity` | Categorical | Feature |
| `parental level of education` | Categorical | Feature |
| `lunch` | Categorical | Feature |
| `test preparation course` | Categorical | Feature |
| `math score` | Numeric (0–100) | **Target (y)** |
| `reading score` | Numeric (0–100) | Feature |
| `writing score` | Numeric (0–100) | Dropped (multicollinearity) |

**Why was writing score dropped?**
Reading score (corr: 0.82) and writing score (corr: 0.80) are both strongly correlated with math score, but they are also heavily correlated with each other (corr: ~0.95), causing multicollinearity. Reading score was kept as it has a slightly higher correlation with the target. Keeping both would distort the Linear Regression coefficients.

---

## Approach

```
1. Setup              →  Import libraries
2. Data Loading       →  Load CSV, inspect shape, nulls, duplicates
3. Preprocessing      →  Encode categoricals (map + get_dummies),
                         drop writing score
4. EDA                →  Score distribution, correlation heatmap,
                         boxplot (test prep vs math score)
5. Outlier Removal    →  IQR method on math score
                         (1000 → 997 rows, 3 outliers removed)
6. Model 1            →  Linear Regression with StandardScaler
                         + Actual vs Predicted plot
                         + Feature importance (coefficients)
7. Model 2            →  Random Forest Regressor (100 trees, OOB scoring)
                         + Actual vs Predicted plot
                         + Feature importance (impurity-based)
8. Comparison         →  Side-by-side MAE / MSE / R² table
9. Insights           →  Actionable recommendations for school admin
```

**Encoding strategy:**
- Ordinal columns (`parental level of education`): `map()` with ordered integer values (0–5)
- Binary columns (`gender`, `lunch`, `test preparation course`): `map()` to 0/1
- Nominal column (`race/ethnicity`): `pd.get_dummies(drop_first=True, dtype=int)`

**Train/Test Split:** 80% train, 20% test, `random_state=42`

**Libraries:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`

---

## Key Results

### EDA Findings

- Math scores follow a **normal distribution**, with most students scoring between 55–80
- `reading score` is the strongest predictor (correlation: **0.82**)
- `lunch` type is the strongest non-academic predictor (correlation: **0.35**) — a proxy for socioeconomic status
- Students who **completed the test preparation course** scored noticeably higher (visible in boxplot)
- `race/ethnicity`, `gender`, and `parental education` show weak linear relationships (below 0.20)

### Outlier Removal

| | Rows |
|---|---|
| Original dataset | 1,000 |
| After IQR outlier removal | 997 |
| Removed | 3 |

### Model Evaluation

| Model | MAE | MSE | R² Score |
|---|---|---|---|
| **Linear Regression** | **4.730935** | **35.051296** | **0.846836** |
| Random Forest Regressor | 5.588892 | 48.846352 | 0.786555 |

**Linear Regression outperformed Random Forest on this dataset.** This is expected — when the relationship between features and target is mostly linear (which the correlation heatmap confirmed), a simple linear model generalises better than a complex ensemble model that can overfit to training noise.

### Feature Importance

**Linear Regression (by standardized coefficient):**
- `reading score` — by far the dominant predictor
- `gender` — 2nd most dominant predictor
- `lunch` — moderate positive weight (standard lunch = higher score)
- `test preparation course` — small negative effect

**Random Forest (by impurity-based importance %):**
- `reading score` — ~74% of total decision-making power
- `gender` — ~12%
- `parental level of education` — ~4.2%

Both models agree: **reading score is the most important feature by a wide margin.**

---

## Visualizations

| Chart | File | Purpose |
|---|---|---|
| Math Score Distribution | `math_score_distribution.png` | Target variable overview |
| Correlation Heatmap | `correlation_heatmap.png` | Feature relationships |
| Boxplot — Test Prep vs Math Score | `boxplot_test_prep.png` | Impact of test prep |
| LR — Actual vs Predicted | `lr_actual_vs_predicted.png` | Linear Regression performance |
| LR Feature Importance | `lr_feature_importance.png` | Coefficient weights |
| RFR — Actual vs Predicted | `rfr_actual_vs_predicted.png` | Random Forest performance |
| RF Feature Importance | `rf_feature_importance.png` | Impurity-based importance |

---

## Actionable Insights for School Administration

1. **Invest in reading programs first** — both models agree that reading comprehension is the single biggest driver of math performance. Schools wanting to improve math scores should start with literacy and language support, especially for word-problem comprehension.

2. **Close the socioeconomic gap** — lunch status (a proxy for household income) is the strongest non-academic predictor. Students on free/reduced lunch consistently score lower. Targeted after-school tutoring and academic support for this group is the highest-leverage intervention available.

3. **Support female students in STEM** — male students showed a slight lean toward higher math scores in this dataset. STEM mentorship programs and math clubs specifically designed for female students can help address this gap over time.

4. **Re-evaluate the test preparation course** — the Random Forest model shows that test prep has a surprisingly small standalone impact once reading ability and socioeconomic status are accounted for. The school should audit whether this course is delivering real value, or whether that budget is better redirected toward foundational literacy programs.

---

## Repository Structure

```
synent-task8-mlmodel-vaibhav-dave/
│
├── data/
│   └── StudentsPerformance.csv
│
├── notebooks/
│   └── StudentsPerformance.ipynb
│
├── outputs/
│   ├── math_score_distribution.png
│   ├── correlation_heatmap.png
│   ├── boxplot_test_prep.png
│   ├── lr_actual_vs_predicted.png
│   ├── lr_feature_importance.png
│   ├── rfr_actual_vs_predicted.png
│   └── rf_feature_importance.png
│
└── README.md
```

---

## How to Run

```bash
# 1. Clone the repository
git clone https://github.com/VaibhavNileshKumarDave/synent-task8-mlmodel-vaibhav-dave

# 2. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn jupyter

# 3. Place dataset in the data/ folder
source: https://www.kaggle.com/datasets/spscientist/students-performance-in-exams

# 4. Open the notebook
jupyter notebook notebooks/students_performance_analysis.ipynb
```

---

## Author

**Vaibhav Dave** — Data Analyst Intern, Synent Technologies
B.Tech Computer Engineering, LDRP-ITR, Gandhinagar (2027)
Internship Submission — June 2026