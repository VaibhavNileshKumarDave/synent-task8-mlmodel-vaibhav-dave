# Task 8 — Machine Learning Model Report

## Objective
Build and compare two regression models (Linear Regression and Random
Forest Regressor) to predict student math scores using academic and
socioeconomic features. The goal is twofold — first, to build a working
predictive model with proper evaluation, and second, to identify which
factors most strongly influence student performance so that school
administrators can make data-driven decisions to improve outcomes.

## Dataset
Students Performance in Exams (Kaggle)
File: StudentsPerformance.csv
Link: https://www.kaggle.com/datasets/spscientist/students-performance-in-exams
Rows: 1,000
Columns: 8

| Column                       | Type        | Role    | Description                              |
|------------------------------|-------------|---------|------------------------------------------|
| gender                       | Categorical | Feature | Male or Female                           |
| race/ethnicity               | Categorical | Feature | Group A through Group E                  |
| parental level of education  | Categorical | Feature | Some high school to Master's degree      |
| lunch                        | Categorical | Feature | Standard or Free/Reduced                 |
| test preparation course      | Categorical | Feature | Completed or None                        |
| math score                   | Integer     | Target  | Score out of 100 — variable to predict   |
| reading score                | Integer     | Feature | Score out of 100                         |
| writing score                | Integer     | Dropped | Dropped due to multicollinearity         |

Dataset quality: No missing values. No duplicate rows.
Clean dataset requiring encoding and outlier handling before modelling.

## Steps Performed

### Phase 1 — Data Loading and Inspection
1. Loaded StudentsPerformance.csv using pandas
2. Performed initial inspection — df.head(), df.info(), df.describe()
3. Verified dataset shape: 1,000 rows × 8 columns
4. Checked for null values — result: 0 nulls across all columns
5. Checked for duplicate rows — result: 0 duplicates

### Phase 2 — Preprocessing and Encoding

6. Encoded Gender column using map():
   female → 0, male → 1

7. Encoded Lunch column using map():
   free/reduced → 0, standard → 1

8. Encoded Test Preparation Course using map():
   none → 0, completed → 1

9. Encoded Parental Level of Education using ordered map():
   This column is ordinal — education level has a natural order,
   so integer mapping preserves that relationship:
   some high school → 0
   high school      → 1
   some college     → 2
   associate's degree → 3
   bachelor's degree  → 4
   master's degree    → 5

10. Encoded Race/Ethnicity using pd.get_dummies():
    This column is nominal — no natural order exists between groups,
    so one-hot encoding is the correct approach.
    drop_first=True used to avoid the dummy variable trap.
    dtype=int specified to ensure integer output (not boolean),
    which is required for correct .corr() computation in the heatmap.
    Result: 4 new binary columns added (group B, C, D, E;
    group A becomes the reference category)

11. Dropped writing score column to resolve multicollinearity:
    Reading score (corr: 0.82) and writing score (corr: 0.80) are
    both strongly correlated with math score, but they are also
    almost perfectly correlated with each other (corr: ~0.95).
    Keeping both in a Linear Regression model would distort the
    coefficients and make feature importance uninterpretable.
    Reading score was kept as it has a marginally higher correlation
    with the target variable.

### Phase 3 — Exploratory Data Analysis (EDA)

#### Math Score Distribution
12. Plotted histogram with KDE curve for math score
13. Observed normal distribution centered around 65–68
14. Majority of students (90%+) score between 40 and 100
15. Small cluster of extreme low scores visible in the left tail

#### Correlation Heatmap
16. Computed correlation matrix for all features and target
17. Plotted as annotated heatmap using coolwarm colormap
18. Key correlations identified:
    - reading score vs math score: 0.82 (strongest predictor)
    - writing score vs math score: 0.80 (dropped — multicollinearity)
    - lunch vs math score: 0.35 (strongest non-academic predictor)
    - test preparation course vs math score: below 0.20
    - parental level of education vs math score: below 0.20
    - race/ethnicity groups vs math score: below 0.20

#### Boxplot — Test Preparation Course vs Math Score
19. Plotted boxplot comparing math score distribution between
    students who completed test prep (1) vs those who did not (0)
20. Identified extreme low-score outliers in the no-prep group
21. Confirmed that test prep students score noticeably higher
    on average, though the difference is moderate

### Phase 4 — Outlier Removal
22. Applied IQR method to identify and remove math score outliers:
    Q1 = 25th percentile of math score
    Q3 = 75th percentile of math score
    IQR = Q3 - Q1
    Lower bound = Q1 - (1.5 × IQR)
    Upper bound = Q3 + (1.5 × IQR)
23. Removed rows where math score fell outside these bounds
    Original rows: 1,000
    Clean rows: 997
    Rows removed: 3
    (Only extreme outliers removed — data integrity preserved)

### Phase 5 — Linear Regression Model

24. Defined target variable: y = math score
25. Defined feature matrix: X = all columns except math score
    and writing score
26. Applied train/test split:
    80% training (797 rows), 20% testing (200 rows)
    random_state=42 for reproducibility
27. Applied StandardScaler to feature matrix:
    Fitted scaler on training data only (fit_transform)
    Applied fitted scaler to test data (transform only)
    This prevents data leakage from test set into training
28. Trained LinearRegression model on scaled training data
29. Generated predictions on scaled test data
30. Evaluated model performance:
    - Mean Absolute Error (MAE)
    - Mean Squared Error (MSE)
    - RMSE (Root Mean Squared Error) = sqrt(MSE)
    - R² Score
31. Plotted Actual vs Predicted scatter chart with perfect
    prediction reference line (red dashed diagonal)
32. Extracted standardized coefficients for each feature
33. Built feature importance dataframe sorted by absolute
    coefficient value
34. Plotted horizontal bar chart of feature importances
    with zero reference line to show positive vs negative impact

### Phase 6 — Random Forest Regressor Model

35. Used same cleaned feature matrix X and target y
36. Applied same 80/20 train/test split with random_state=42
    Note: Random Forest does not require feature scaling —
    tree-based models are not distance-based, so raw feature
    values are used directly
37. Trained RandomForestRegressor:
    n_estimators=100 (100 decision trees in the ensemble)
    random_state=42 for reproducibility
    oob_score=True (Out-of-Bag scoring enabled for internal
    validation using samples not seen during each tree's training)
    bootstrap=True (each tree trained on a bootstrap sample)
38. Printed Out-of-Bag Score as an additional internal
    validation metric
39. Generated predictions on test data
40. Evaluated model with same metrics as Linear Regression:
    MAE, MSE, RMSE, R² Score
41. Plotted Actual vs Predicted scatter chart with perfect
    prediction reference line
42. Extracted impurity-based feature importances from
    rfr_model.feature_importances_
43. Built feature importance dataframe with importance as
    percentage of total decision-making power
44. Plotted horizontal bar chart of RF feature importances

### Phase 7 — Model Comparison
45. Built side-by-side comparison table with both models:
    MAE, RMSE, R² Score printed cleanly using to_string()
46. Identified winning model based on all three metrics
47. Explained why Linear Regression outperformed Random Forest
    on this specific dataset

### Phase 8 — Insights and Reporting
48. Wrote actionable insights for school administration based
    on model findings
49. Framed recommendations around real decisions administrators
    can make — not just statistical observations

## Tools Used
Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn,
Jupyter Notebook

## Charts Produced

| Chart                              | File                          | Type         |
|------------------------------------|-------------------------------|--------------|
| Math Score Distribution            | math_score_distribution.png   | Histogram    |
| Correlation Heatmap                | correlation_heatmap.png       | Heatmap      |
| Test Prep vs Math Score            | boxplot_test_prep.png         | Boxplot      |
| LR — Actual vs Predicted           | lr_actual_vs_predicted.png    | Scatter plot |
| LR — Feature Importance            | lr_feature_importance.png     | Bar chart    |
| RFR — Actual vs Predicted          | rfr_actual_vs_predicted.png   | Scatter plot |
| RF — Feature Importance            | rf_feature_importance.png     | Bar chart    |

## Key Results

### Outlier Removal Summary
| Stage           | Rows  |
|-----------------|-------|
| Original dataset| 1,000 |
| After IQR removal| 997  |
| Removed         | 3     |

### Model Evaluation Summary

| Model                   | MAE | RMSE | R² Score |
|-------------------------|-----|------|----------|
| **Linear Regression**       | **4.730935** | **35.051296** | **0.846836** |
| Random Forest Regressor | 5.588892 | 48.846352 | 0.786555 |

(Fill in actual values from Section 9 comparison table output
in your notebook before pushing to GitHub)

### Why Linear Regression Won
Linear Regression outperformed Random Forest on this dataset across
all three metrics. This is expected and explainable — the correlation
heatmap confirmed that the dominant predictor (reading score at 0.82)
has a strong linear relationship with the target. When feature-target
relationships are largely linear, a simple linear model generalises
better than a complex ensemble that risks overfitting to training noise.
This is an important lesson in model selection: complexity is not
always better. The right model depends on the structure of the data.

### Feature Importance — Both Models Agree
Both Linear Regression (by standardized coefficient) and Random Forest
(by impurity-based importance) identified the same top predictor:

- Reading score accounts for approximately 83% of Random Forest's
  total decision-making power
- All other features — lunch, gender, test prep, parental education,
  race/ethnicity — collectively account for the remaining ~17%

This level of dominance from a single feature is a strong, reliable
signal — not a coincidence or model artifact.

### Key Feature Findings

Reading score (strongest predictor):
Students with strong reading comprehension consistently score higher
in math. This is likely driven by the ability to understand and
decode word problems, which form a significant portion of math exams.

Lunch type — standard vs free/reduced (strongest non-academic predictor):
Lunch status is a well-established proxy for household income and
socioeconomic background. Students on free/reduced lunch consistently
score lower — not because of the lunch itself, but because of the
broader socioeconomic disadvantages that often accompany it, including
limited access to tutoring, study resources, and stable home environments.

Test preparation course (surprisingly weak standalone predictor):
Once reading ability and socioeconomic status are accounted for in
the model, the test preparation course shows limited additional impact
on math scores. This suggests either that the course content is not
targeting the right skills, or that students who already have strong
foundational reading skills benefit from it more than those who do not.

Gender (small but visible):
Male students showed a marginal lean toward higher math scores in this
dataset. The effect is small but visible in the Linear Regression
coefficients. This reflects a well-documented historical pattern
in standardized math testing.

## Business Recommendations for School Administration

1. Invest in reading and literacy programs first — both models
   confirm that reading comprehension is the biggest single
   driver of math performance. Schools wanting to lift math
   scores should start with foundational language and literacy
   support, especially for interpreting complex word problems.
   This is counterintuitive but the data is clear.

2. Target socioeconomic support at free/reduced lunch students —
   lunch status is the strongest non-academic predictor. Providing
   free after-school tutoring, study materials, and academic
   mentoring specifically for this group is the highest-leverage
   intervention available outside of classroom instruction.

3. Introduce STEM mentorship for female students — the gender gap
   in math scores is small but present. Dedicated math clubs,
   female STEM role models, and mentorship programs can help
   close this gap progressively over time.

4. Re-evaluate the test preparation course curriculum — the model
   shows that the course has surprisingly limited standalone impact
   once other factors are controlled for. The administration should
   audit whether the course is actually delivering measurable
   improvement, or whether that budget would generate more impact
   if redirected toward foundational literacy programs instead.

5. Use this model as an early warning system — with an R² score
   showing meaningful predictive power, a version of this model
   could theoretically be deployed at the start of a semester
   to flag students at risk of low math performance based on
   their reading scores and background — enabling early
   intervention before exam season rather than after.