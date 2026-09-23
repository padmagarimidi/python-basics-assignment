# Module 2 - Analytics Pipeline (`/analytics`)

One cohesive Titanic pipeline: profile -> clean -> visual story -> predictive modeling.
The raw dataset is loaded **exactly once** (`sns.load_dataset('titanic')` in `01_eda.py`) and
saved as `titanic.csv`; everything afterwards continues from that one load.

## Run order

```bash
pip install -r requirements.txt
python 01_eda.py             # load ONCE -> titanic.csv, profile, clean, EDA + data story
python 02_modeling.py        # reads titanic.csv (no second load): models, tuning, regression, saves joblib
python 03_reload_pipeline.py # reloads best_pipeline.joblib and predicts on RAW input
python 04_build_readme.py    # regenerates this README from the reports
```

## Layout

| path | what it is |
|---|---|
| `01_eda.py` | Part A (Tasks 1-6): load once, save `titanic.csv`, profile, missing-value rule, univariate/bivariate/multivariate EDA, z-score check |
| `02_modeling.py` | Part B (Tasks 7-15): stratified split, leak-free Pipeline, 3 classifiers, metrics, imbalance study, GridSearchCV + OOB, fare regression, comparison + recommendation, `joblib.dump` |
| `03_reload_pipeline.py` | `joblib.load` + end-to-end prediction on raw, unpreprocessed rows |
| `titanic.csv` | the single committed offline fallback (`pd.read_csv("titanic.csv")`) |
| `best_pipeline.joblib` | complete fitted Pipeline (imputer/encoder/scaler + estimator) |
| `figures/` | all chart images |
| `reports/`, `results/` | auto-generated write-ups, comparison CSV, hold-out rows used by the reload check |

## Notes

* **Where cleaning happens.** `01_eda.py` applies the threshold rule to the whole frame for EDA. `02_modeling.py`
  starts from the same `titanic.csv` and repeats only the *statistic-free* parts (drop 2 rows with missing `embarked`,
  drop 77 %-missing `deck`); `age` is imputed **inside the Pipeline, fit on the training split only**, so nothing is
  learned from test rows.
* **Decision-tree thresholds** in `figures/11_decision_tree.png` are on *standardised* features (e.g. `age <= -1.99`
  means about 2 standard deviations below the training mean), because the tree sits after the scaler in the Pipeline.
* Random seed is fixed (`42`); numbers below are the ones produced by that seed.

---


# Task 1 - Load, save offline fallback, profile

Raw dataset loaded once with `sns.load_dataset('titanic')` and saved to `titanic.csv` (`df.to_csv('titanic.csv', index=False)`).

**df.shape:** `(891, 15)`

**df.info():**

```
<class 'pandas.DataFrame'>
RangeIndex: 891 entries, 0 to 890
Data columns (total 15 columns):
 #   Column       Non-Null Count  Dtype   
---  ------       --------------  -----   
 0   survived     891 non-null    int64   
 1   pclass       891 non-null    int64   
 2   sex          891 non-null    str     
 3   age          714 non-null    float64 
 4   sibsp        891 non-null    int64   
 5   parch        891 non-null    int64   
 6   fare         891 non-null    float64 
 7   embarked     889 non-null    str     
 8   class        891 non-null    category
 9   who          891 non-null    str     
 10  adult_male   891 non-null    bool    
 11  deck         203 non-null    category
 12  embark_town  889 non-null    str     
 13  alive        891 non-null    str     
 14  alone        891 non-null    bool    
dtypes: bool(2), category(2), float64(2), int64(4), str(5)
memory usage: 80.7 KB
```

**df.describe():**

```
       survived   pclass      age    sibsp    parch     fare
count   891.000  891.000  714.000  891.000  891.000  891.000
mean      0.384    2.309   29.699    0.523    0.382   32.204
std       0.487    0.836   14.526    1.103    0.806   49.693
min       0.000    1.000    0.420    0.000    0.000    0.000
25%       0.000    2.000   20.125    0.000    0.000    7.910
50%       0.000    3.000   28.000    0.000    0.000   14.454
75%       1.000    3.000   38.000    1.000    0.000   31.000
max       1.000    3.000   80.000    8.000    6.000  512.329
```

**Percentage of missing values (columns with any):**

| column | missing count | missing % |
|---|---|---|
| deck | 688 | 77.22% |
| age | 177 | 19.87% |
| embarked | 2 | 0.22% |
| embark_town | 2 | 0.22% |


# Task 2 - Missing-value handling

Threshold rule: **< 5 % missing -> drop those rows; 5 %-30 % -> impute; > 30 % -> imputation is unreliable, so explicitly drop the column or encode 'Missing' as its own category.**

- **`deck`: measured missing = 77.22%**
  - 77.22% > 30%  ->  rule: imputation unreliable; decision = **category**.
  - Justification: Missingness in `deck` is itself informative: cabin numbers were mostly recorded for first-class passengers, so 'no deck recorded' is a proxy for lower class/fare. Dropping the column would throw that signal away, and imputing ~77 % of the values would mean inventing most of the column. Chosen: keep the column and encode NaN as its own category 'Missing'.
- **`age`: measured missing = 19.87%**
  - 5% <= 19.87% <= 30%  ->  rule: *impute*; filled with 28.0 (median chosen over mean because the distribution is mildly right-skewed and the median is robust to outliers).
- **`embarked`: measured missing = 0.22%**
  - 0.22% < 5%  ->  rule: *drop rows* with missing `embarked` (2 rows).
- **`embark_town`: measured missing = 0.22%**
  - 0.22% < 5%  ->  rule: *drop rows* with missing `embark_town` (2 rows).

Rows dropped by the <5 % rule: 2 (`embarked` and `embark_town` are missing on the same rows, so they are dropped once). Cleaned shape: `(889, 16)`; remaining NaNs in original columns: 0.


# Task 3 - Univariate analysis (age, fare)

*(figure saved: `figures/03_univariate_age_fare.png`)*

**IQR-rule outliers** (outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR]):

| column | Q1 | Q3 | IQR | lower fence | upper fence | # outliers |
|---|---|---|---|---|---|---|
| age | 22.00 | 35.00 | 13.00 | 2.50 | 54.50 | **65** |
| fare | 7.90 | 31.00 | 23.10 | -26.76 | 65.66 | **114** |

**Fare central tendency:** mean = 32.097, median = 14.454, mode = 8.050.

**Conclusion:** the fare distribution is **right-skewed (positively skewed)** (mean > median > mode: 32.10 vs 14.45 vs 8.05). A minority of very expensive first-class tickets (max fare 512.3) drags the mean far above the median, while the mode sits at the cheap end where most third-class passengers paid. Sample skewness = 4.80.


# Task 4 - Bivariate analysis

**(a) Survival rate by sex**

| group | n | survival rate |
|---|---|---|
| female | 312 | 74.0% |
| male | 577 | 18.9% |

**(b) Survival rate by pclass**

| group | n | survival rate |
|---|---|---|
| 1st class | 214 | 62.6% |
| 2nd class | 184 | 47.3% |
| 3rd class | 491 | 24.2% |
| 1st **or** 2nd (`c1 | c2`) | 398 | 55.5% |

**(c) Survival rate by sex AND pclass** (masks combined with `&`)

| group | n | survival rate |
|---|---|---|
| female & 1st class | 92 | 96.7% |
| female & 2nd class | 76 | 92.1% |
| female & 3rd class | 144 | 50.0% |
| male & 1st class | 122 | 36.9% |
| male & 2nd class | 108 | 15.7% |
| male & 3rd class | 347 | 13.5% |

**Correlation matrix (exactly 6 columns; `adult_male` and `alone` deliberately excluded - they are derived/redundant flags computable from sex/age and sibsp+parch):**

```
          survived  pclass    age  sibsp  parch   fare
survived     1.000  -0.336 -0.070 -0.034  0.083  0.255
pclass      -0.336   1.000 -0.337  0.082  0.017 -0.548
age         -0.070  -0.337  1.000 -0.233 -0.171  0.094
sibsp       -0.034   0.082 -0.233  1.000  0.415  0.161
parch        0.083   0.017 -0.171  0.415  1.000  0.218
fare         0.255  -0.548  0.094  0.161  0.218  1.000
```

*(figure saved: `figures/04_correlation_heatmap.png`)*

**Two strongest correlations (largest |r| among off-diagonal pairs):**

1. `pclass` vs `fare`: r = -0.548 (negative).
2. `sibsp` vs `parch`: r = +0.415 (positive).

- **Interpretation `pclass` ~ `fare` (r = -0.55):** Lower class number (1st class) means a higher ticket price, so as pclass rises the fare falls - a strong negative link that mirrors the ticket structure of the ship.
- **Interpretation `sibsp` ~ `parch` (r = +0.41):** Passengers travelling with siblings/spouses often also travelled with parents/children, i.e. these two counts both measure family group size and move together.

*(Pearson r only captures linear association, and `survived` is binary, so correlations with it are best read as directional hints, not effect sizes.)*


# Task 5 - Multivariate data story: who survived, and why?


### Chart 1 - Survival rate by sex and passenger class

*(figure saved: `figures/05_story1_sex_class.png`)*

**Interpretation.** Sex and class together are the two dominant drivers. Women in 1st class survived at 96.7% and in 2nd class at 92.1%, while men in 3rd class survived at only 13.5%. Within every class women outlived men, consistent with a 'women and children first' evacuation order, and the female-3rd-class rate (50.0%) shows that class disadvantage still cut into women's chances.

### Chart 2 - Age distribution of survivors vs non-survivors, by sex

*(figure saved: `figures/06_story2_age_survival.png`)*

**Interpretation.** Children under 13 survived at 58.0% versus 38.6% for everyone else. The effect is strongest for males: boys under 13 survived at 56.8% against 17.3% for adult men (13+). Women and girls of every age band survived at well over half (see Chart 5), so age matters mainly as a modifier for males. Only passengers with a recorded age are plotted (712 of 889) so the imputed median does not create a fake spike.

### Chart 3 - Fare paid by class and survival

*(figure saved: `figures/07_story3_fare_class_survival.png`)*

**Interpretation.** Overall, survivors paid a median fare of 26.0 versus 10.5 for non-survivors, which looks like a wealth effect. But the boxes show most of that gap is really a class effect: within 1st class the medians are 77.3 (survived) vs 44.8 (died) and in 3rd class 8.5 vs 8.1: the gap shrinks sharply once class is held fixed, and is almost gone in 3rd class. Ticket price acts largely as a proxy for cabin location and status rather than an independent cause.

### Chart 4 - Survival rate by family size

*(figure saved: `figures/08_story4_family_size.png`)*

**Interpretation.** Passengers travelling alone survived at 30.1%, small families of 2-4 at 57.9% and families of 5+ at only 16.1%. Small groups could look after one another and reach lifeboats together, whereas large families (mostly 3rd class) struggled to keep everyone together. Several family-size groups are small (see the n labels), so those bars are noisy.

### Chart 5 - Survival heatmap: age group x sex

*(figure saved: `figures/09_story5_agegroup_sex_heatmap.png`)*

**Interpretation.** The best-off cell is female / Senior (60+) at 100.0% survival, and the worst is male / Teen (13-19) at 9.6%. Across nearly every age band the female column is far greener than the male one; the child row is the exception, where girls and boys are almost level and boys do far better than adult men. Cell sizes (n) are shown because the oldest bands hold few passengers.

### Data story - putting it together

Being **female** was the strongest single predictor of survival (74.0% vs 18.9% for men), **class** modulated it (1st 62.6%, 3rd 24.2%), **age** helped mainly young boys, and **family size** showed a sweet spot at 2-4 people. Fare mostly echoes class. Together this describes a 'women and children first' evacuation that was filtered through class-based access to the boat deck.


# Task 6 - Exploratory z-score standardisation check (age, fare)

*EDA-stage sanity check only - fitted on the full cleaned DataFrame and NOT used by the modeling pipeline, which performs its own train-only scaling.*

```
      mean (before)  std (before)  mean (after)  std (after)
age       29.315152     12.977627           0.0          1.0
fare      32.096681     49.669545           0.0          1.0
```

Manual check `z = (x - mean) / std` matches the scaler: max |difference| for age = 0.00e+00, fare = 0.00e+00.

*(figure saved: `figures/10_standardisation_before_after.png`)*

The shape of each distribution is unchanged (z-scoring is a linear rescale); only the centre moves to 0 and the spread to 1.


---



# Part B - Predictive modeling (continuation of the same dataset)

Read `titanic.csv` (shape (891, 15)); no second `sns.load_dataset` call anywhere.

- `embarked` missing 0.22 % (< 5 %) -> dropped 2 rows.
- `deck` missing 77.2 % (> 30 %) -> dropped for modeling; its signal is largely carried by `pclass` and `fare` (in EDA it was kept as a 'Missing' category).
- `age` missing 19.9 % (5-30 %) -> **median-imputed inside the Pipeline**, fit on the training split only.
- Redundant/derived/leaky columns are not used as features: `alive` (identical to the target), `class`, `who`, `adult_male`, `alone`, `embark_town` (duplicates of pclass / sex+age / sibsp+parch / embarked).


# Task 7 - Stratified train/test split

Class balance in the data: not survived = 549 (61.8%), survived = 340 (38.2%).

Split 80/20 with `stratify=y`: train = 711 rows (38.3% survived), test = 178 rows (38.2% survived).

**Why stratify:** the positive class is a ~38% minority. A plain random split of only 889 rows can easily give the test set a noticeably different survival rate than the training set, which makes metrics noisy and can starve either side of positives. Stratifying keeps the class ratio (within one row) identical in both splits, so precision/recall/F1 are measured on a representative test set.


# Task 8 - Preprocessing (fit on train only, transform-only on test)

Implemented as `Pipeline([('prep', ColumnTransformer), ('clf', estimator)])`:

- numeric (`pclass, age, sibsp, parch, fare`): `SimpleImputer(median)` -> `StandardScaler`
- categorical (`sex, embarked`): `SimpleImputer(most_frequent)` -> `OneHotEncoder(handle_unknown='ignore')`

`pipe.fit(X_train, y_train)` fits every imputer/encoder/scaler on the training rows only; `pipe.predict(X_test)` / `predict_proba(X_test)` only call `transform`. Cross-validation and GridSearchCV below clone the whole pipeline per fold, so validation folds are never used to fit preprocessing either.

Proof: fitted scaler mean of `age` = 29.404 (imputed training ages) vs mean over the FULL data = 29.315; they differ because the scaler only saw the training split.


# Task 9 - Logistic Regression, Decision Tree, Random Forest

All three models are fit on the same `X_train, y_train` and evaluated on the same `X_test, y_test`.

*(figure saved: `figures/11_decision_tree.png`)*

Top decision-tree features: `sex_female` (0.63), `pclass` (0.20), `age` (0.08).


# Task 10 - Evaluation on the held-out test set

**Side-by-side comparison (test set, positive class = survived):**

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.809 | 0.783 | 0.691 | 0.734 | 0.861 |
| Decision Tree | 0.798 | 0.776 | 0.662 | 0.714 | 0.851 |
| Random Forest | 0.803 | 0.762 | 0.706 | 0.733 | 0.824 |

**Confusion-matrix counts (rows = actual, cols = predicted):**

| Model | TN | FP | FN | TP |
|---|---|---|---|---|
| Logistic Regression | 97 | 13 | 21 | 47 |
| Decision Tree | 97 | 13 | 23 | 45 |
| Random Forest | 95 | 15 | 20 | 48 |

*(figure saved: `figures/12_confusion_matrices.png`)*

*(figure saved: `figures/13_roc_curves.png`)*


# Task 11 - Imbalance handling comparison (Random Forest)

Class balance: not survived = 549 (61.8%), survived = 340 (38.2%) -> imbalance ratio 1.61:1 (moderate, not extreme).

| Variant | Test Precision | Test Recall | Test F1 | CV Precision | CV Recall | CV F1 |
|---|---|---|---|---|---|---|
| (a) Baseline (no handling) | 0.762 | 0.706 | 0.733 | 0.753 | 0.750 | 0.750 |
| (b) class_weight='balanced' | 0.774 | 0.706 | 0.738 | 0.755 | 0.757 | 0.755 |
| (c) SMOTE (train only) | 0.761 | 0.750 | 0.756 | 0.739 | 0.772 | 0.754 |

*Test = one held-out split of 178 rows (noisy). CV = 5-fold stratified CV on the training split only; for (c) SMOTE is re-applied inside every training fold and never touches the validation fold, so there is no leakage.*

**Conclusion.** Ranked by cross-validated F1, the best strategy is **(b) class_weight='balanced'** (CV F1 0.755, recall 0.757, precision 0.755) versus baseline (CV F1 0.750, recall 0.750, precision 0.753). Because the minority class is ~38% of the data (moderate imbalance) and a Random Forest already copes reasonably, differences between strategies are small. Relative to baseline, class_weight changes CV recall by +0.007 and precision by +0.002, while SMOTE changes CV recall by +0.022 and precision by -0.014: rebalancing mainly shifts the precision/recall trade-off rather than delivering a large F1 gain. With only 178 test rows, gaps of 1-2 F1 points on the test column are within noise, which is why the CV columns drive this conclusion. SMOTE is also imperfect here: it interpolates between rows in a space that includes one-hot columns, so its synthetic passengers can have fractional 'sex'/'embarked' values.


# Task 12 - Hyperparameter tuning of Random Forest (GridSearchCV, OOB score)

Estimator: `RandomForestClassifier(oob_score=True, random_state=42)` inside the same preprocessing Pipeline. Grid: `n_estimators` in [100, 300, 500], `max_depth` in [3, 5, 8, None], `max_features` in ['sqrt', 'log2', None]; 5-fold stratified CV, scoring = ROC-AUC (threshold-free).

- **Best parameters:** `{'max_depth': 5, 'max_features': 'sqrt', 'n_estimators': 100}`
- **Best CV ROC-AUC:** 0.8783
- **OOB score of the best model (`oob_score_`, accuracy on out-of-bag rows):** 0.8158

Tuned model on the held-out test set: Accuracy = 0.826, Precision = 0.849, Recall = 0.662, F1 = 0.744, AUC = 0.837.


# Task 13 - Regression side-task: predict `fare` (multivariate linear regression)

Target: `fare`. Features: ['pclass', 'age', 'sibsp', 'parch', 'sex', 'embarked']. `survived` is excluded because it is the classification target rather than a passenger attribute; `deck`, `class`, `who`, `adult_male`, `alone`, `embark_town`, `alive` are derived duplicates. Same imputer/encoder/scaler discipline (fit on train only) via a Pipeline; `OneHotEncoder(drop='first')` avoids the dummy-variable trap for OLS.

**Test metrics (n=178, p=7 encoded predictors):** MAE = 21.140, RMSE = 41.747, R² = 0.347, Adjusted R² = 0.320  (Adjusted R² = 1 - (1-R²)(n-1)/(n-p-1)).

*(figure saved: `figures/14_regression_residuals.png`)*

**Conclusion:** the residual plot **shows heteroscedasticity** - the residual spread is not constant: it is 4.3x wider for the upper half of predicted fares than the lower half (a fan/funnel shape), and |residual| rises with the prediction (Spearman +0.52, p < 0.001). Breusch-Pagan does NOT reject constant variance (p = 0.253); BP only tests a linear link between the features and the squared residuals, so it can miss this pattern, which grows with the *prediction* itself. Fare is strongly right-skewed (a few fares above 200), so an untransformed linear model fits cheap tickets tightly and expensive ones poorly, and can even predict negative fares. Modelling log(fare) or using weighted/robust regression is the natural fix; R² is only 0.35, i.e. these attributes explain a limited share of fare variance.


# Task 14 - Model comparison and recommendation

Classification and regression metrics are **on different scales and not comparable**; they are shown as two separate metric groups, one per model type. Cells that do not apply to a model type are blank (—).

| Model | [CLS] Accuracy | [CLS] Precision | [CLS] Recall | [CLS] F1 | [CLS] AUC | [REG] MAE | [REG] RMSE | [REG] R² | [REG] Adjusted R² |
|---|---|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.809 | 0.783 | 0.691 | 0.734 | 0.861 | — | — | — | — |
| Decision Tree | 0.798 | 0.776 | 0.662 | 0.714 | 0.851 | — | — | — | — |
| Random Forest | 0.803 | 0.762 | 0.706 | 0.733 | 0.824 | — | — | — | — |
| Random Forest (tuned) | 0.826 | 0.849 | 0.662 | 0.744 | 0.837 | — | — | — | — |
| Linear Regression (fare) | — | — | — | — | — | 21.140 | 41.747 | 0.347 | 0.320 |

`[CLS]` = classification metrics (higher is better, bounded 0-1). `[REG]` = regression metrics (MAE/RMSE in fare units - lower is better; R²/Adjusted R² unitless). Saved as `results/model_comparison.csv` with a two-level column header.

Selection rule: highest 5-fold CV ROC-AUC **on the training split** (the test set is not used to choose). Random Forest (tuned) = 0.878, Random Forest = 0.863, Logistic Regression = 0.844, Decision Tree = 0.843.

**Final recommendation.**
I would deploy **Random Forest (tuned)**, which had the best 5-fold training CV ROC-AUC (0.878) and on the held-out test set reached accuracy 0.826, precision 0.849, recall 0.662, F1 0.744 and AUC 0.837. The runner-up, Random Forest, scored F1 0.733 / AUC 0.824 on test, and the Decision Tree (F1 0.714, AUC 0.851) is the most interpretable but the least stable. Note that Logistic Regression actually has the higher test AUC (0.861 vs 0.837) and a similar F1 (0.734), so its edge is small and rests on CV, accuracy and precision rather than a clear-cut win; Logistic Regression is the simpler, more transparent fallback. Recall of 0.662 means about 34% of actual survivors are still missed, and with only 178 test rows differences of a few points are within sampling noise, so lower the decision threshold if missed survivors are costlier and re-validate on new data.


# Task 15 - Persist the best full pipeline

Saved `best_pipeline.joblib` = fitted `Pipeline(prep=ColumnTransformer[imputer/encoder/scaler], clf=RandomForestClassifier)` for **Random Forest (tuned)**. It expects RAW columns ['pclass', 'age', 'sibsp', 'parch', 'fare', 'sex', 'embarked'] (missing `age` allowed) and needs no manual preprocessing.

Reload check (also runnable on its own): `python analytics/03_reload_pipeline.py`.
In-script check: `joblib.load(...)` reproduces the exact test predictions on raw input: **OK**.