"""
Module 2 - Analytics Pipeline  ANd Predictive modeling
===========================================================
"""

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from scipy import stats
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score,
                             confusion_matrix, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold, cross_val_score,
                                     cross_validate, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

RANDOM_STATE = 42
BASE = Path(__file__).resolve().parent
FIG, REP, RES = BASE / "figures", BASE / "reports", BASE / "results"
for d in (FIG, REP, RES):
    d.mkdir(exist_ok=True)
sns.set_theme(style="whitegrid", context="notebook")


_report: list[str] = []


def say(text: str = "") -> None:
    print(text)
    _report.append(text)


def heading(text: str, level: int = 2) -> None:
    say()
    say(f"{'#' * level} {text}")
    say()


def save_fig(name: str) -> None:
    plt.tight_layout()
    plt.savefig(FIG / name, dpi=150, bbox_inches="tight")
    plt.close()
    say(f"*(figure saved: `figures/{name}`)*")
    say()


def md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}", index: bool = True) -> str:
    """Tiny DataFrame -> markdown-table helper (avoids the `tabulate` dependency)."""
    d = df.reset_index() if index else df.copy()
    cols = [str(c) for c in d.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in d.iterrows():
        cells = []
        for v in row:
            if isinstance(v, (float, np.floating)):
                cells.append("—" if np.isnan(v) else floatfmt.format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# DATA: continue from the single load (titanic.csv)

heading("Part B - Predictive modeling (continuation of the same dataset)", 1)

df = pd.read_csv(BASE / "titanic.csv")  # <-- the offline copy of the ONE raw load
say(f"Read `titanic.csv` (shape {df.shape}); no second `sns.load_dataset` call anywhere.")
say()

# Deterministic cleaning that mirrors 01_eda.py's threshold rule (no learned stats)
n0 = len(df)
df = df.dropna(subset=["embarked"]).reset_index(drop=True)  # 0.22 % < 5 %  -> drop rows
say(f"- `embarked` missing 0.22 % (< 5 %) -> dropped {n0 - len(df)} rows.")
say("- `deck` missing 77.2 % (> 30 %) -> dropped for modeling; its signal is largely "
    "carried by `pclass` and `fare` (in EDA it was kept as a 'Missing' category).")
say("- `age` missing 19.9 % (5-30 %) -> **median-imputed inside the Pipeline**, fit on "
    "the training split only.")
say("- Redundant/derived/leaky columns are not used as features: `alive` (identical to "
    "the target), `class`, `who`, `adult_male`, `alone`, `embark_town` (duplicates of "
    "pclass / sex+age / sibsp+parch / embarked).")
say()

NUM_FEATURES = ["pclass", "age", "sibsp", "parch", "fare"]
CAT_FEATURES = ["sex", "embarked"]
FEATURES = NUM_FEATURES + CAT_FEATURES
TARGET = "survived"

X, y = df[FEATURES], df[TARGET]


def build_preprocessor(drop=None) -> ColumnTransformer:
    """Imputer + encoder + scaler. Fit on TRAIN only when used inside a Pipeline."""
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop=drop)),
    ])
    return ColumnTransformer(
        [("num", numeric, NUM_FEATURES), ("cat", categorical, CAT_FEATURES)],
        verbose_feature_names_out=False,
    )


def make_pipe(estimator) -> Pipeline:
    return Pipeline([("prep", build_preprocessor()), ("clf", estimator)])



# TASK 7 - STRATIFIED SPLIT (before any preprocessing is fit)

heading("Task 7 - Stratified train/test split", 1)
balance = y.value_counts().sort_index()
minority_share = balance.min() / balance.sum()
say(f"Class balance in the data: not survived = {balance[0]} ({balance[0]/len(y):.1%}), "
    f"survived = {balance[1]} ({balance[1]/len(y):.1%}).")
say()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)
say(f"Split 80/20 with `stratify=y`: train = {len(X_train)} rows "
    f"({y_train.mean():.1%} survived), test = {len(X_test)} rows ({y_test.mean():.1%} survived).")
say()
say(f"**Why stratify:** the positive class is a ~{minority_share:.0%} minority. A plain "
    f"random split of only {len(y)} rows can easily give the test set a noticeably "
    f"different survival rate than the training set, which makes metrics noisy and "
    f"can starve either side of positives. Stratifying keeps the class ratio "
    f"(within one row) identical in both splits, so precision/recall/F1 are measured "
    f"on a representative test set.")
say()


# TASK 8 - PREPROCESSING (done through Pipeline => fit on train only)

heading("Task 8 - Preprocessing (fit on train only, transform-only on test)", 1)
say("Implemented as `Pipeline([('prep', ColumnTransformer), ('clf', estimator)])`:")
say()
say("- numeric (`pclass, age, sibsp, parch, fare`): `SimpleImputer(median)` -> `StandardScaler`")
say("- categorical (`sex, embarked`): `SimpleImputer(most_frequent)` -> `OneHotEncoder(handle_unknown='ignore')`")
say()
say("`pipe.fit(X_train, y_train)` fits every imputer/encoder/scaler on the training rows "
    "only; `pipe.predict(X_test)` / `predict_proba(X_test)` only call `transform`. "
    "Cross-validation and GridSearchCV below clone the whole pipeline per fold, so "
    "validation folds are never used to fit preprocessing either.")
say()

# quick, explicit proof that the scaler learned only train statistics
probe = make_pipe(LogisticRegression()).fit(X_train, y_train)
scaler_mean_age = probe.named_steps["prep"].named_transformers_["num"].named_steps["scale"].mean_[1]
say(f"Proof: fitted scaler mean of `age` = {scaler_mean_age:.3f} (imputed training ages) "
    f"vs mean over the FULL data = {X['age'].fillna(X['age'].median()).mean():.3f}; "
    f"they differ because the scaler only saw the training split.")
say()


# TASK 9 - TRAIN THREE CLASSIFIERS ON THE IDENTICAL SPLIT

heading("Task 9 - Logistic Regression, Decision Tree, Random Forest", 1)
models = {
    "Logistic Regression": make_pipe(LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    "Decision Tree": make_pipe(DecisionTreeClassifier(max_depth=4, min_samples_leaf=5,
                                                      random_state=RANDOM_STATE)),
    "Random Forest": make_pipe(RandomForestClassifier(n_estimators=300,
                                                      random_state=RANDOM_STATE, n_jobs=-1)),
}
for name, pipe in models.items():
    pipe.fit(X_train, y_train)
say("All three models are fit on the same `X_train, y_train` and evaluated on the same `X_test, y_test`.")
say()

# Decision-tree visualisation with feature + class names
dt_pipe = models["Decision Tree"]
feat_names = list(dt_pipe.named_steps["prep"].get_feature_names_out())
plt.figure(figsize=(22, 10))
plot_tree(dt_pipe.named_steps["clf"], feature_names=feat_names,
          class_names=["Not survived", "Survived"], filled=True, rounded=True,
          fontsize=9, proportion=False)
plt.title("Decision Tree (max_depth=4) - fitted on the training split")
save_fig("11_decision_tree.png")
imp = pd.Series(dt_pipe.named_steps["clf"].feature_importances_, index=feat_names)
say("Top decision-tree features: " + ", ".join(f"`{k}` ({v:.2f})" for k, v in
                                               imp.sort_values(ascending=False).head(3).items()) + ".")
say()



# TASK 10 - EVALUATION: confusion matrix, accuracy, precision, recall, F1, ROC/AUC

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def evaluate(pipe, Xt, yt) -> dict:
    pred = pipe.predict(Xt)
    proba = pipe.predict_proba(Xt)[:, 1]
    tn, fp, fn, tp = confusion_matrix(yt, pred).ravel()
    return {
        "Accuracy": accuracy_score(yt, pred),
        "Precision": precision_score(yt, pred, zero_division=0),
        "Recall": recall_score(yt, pred, zero_division=0),
        "F1": f1_score(yt, pred, zero_division=0),
        "AUC": roc_auc_score(yt, proba),
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
    }, pred, proba



# TASK 10 (cont.) evaluate everything, side-by-side table + plots

heading("Task 10 - Evaluation on the held-out test set", 1)
rows, preds, probas = {}, {}, {}
for name, pipe in models.items():
    m, p, pr = evaluate(pipe, X_test, y_test)
    rows[name], preds[name], probas[name] = m, p, pr
clf_table = pd.DataFrame(rows).T
metric_cols = ["Accuracy", "Precision", "Recall", "F1", "AUC"]

say("**Side-by-side comparison (test set, positive class = survived):**")
say()
say(md_table(clf_table[metric_cols], index=True).replace("| index |", "| Model |"))
say()
say("**Confusion-matrix counts (rows = actual, cols = predicted):**")
say()
say(md_table(clf_table[["TN", "FP", "FN", "TP"]].astype(int).astype(float), "{:.0f}")
    .replace("| index |", "| Model |"))
say()

# confusion matrices
fig, axes = plt.subplots(1, len(models), figsize=(4.6 * len(models), 4.2))
for ax, (name, p) in zip(axes, preds.items()):
    ConfusionMatrixDisplay.from_predictions(
        y_test, p, display_labels=["Not surv.", "Survived"], cmap="Blues", ax=ax, colorbar=False)
    ax.set_title(name, fontsize=11)
    ax.grid(False)
save_fig("12_confusion_matrices.png")

# ROC curves
fig, ax = plt.subplots(figsize=(7, 6))
for name, pr in probas.items():
    RocCurveDisplay.from_predictions(y_test, pr, name=f"{name} (AUC={rows[name]['AUC']:.3f})", ax=ax)
ax.plot([0, 1], [0, 1], "k--", label="Chance")
ax.set_title("ROC curves - test set")
ax.legend(loc="lower right", fontsize=9)
save_fig("13_roc_curves.png")


# TASK 11 - IMBALANCE HANDLING (baseline vs class_weight vs SMOTE-on-train-only)

heading("Task 11 - Imbalance handling comparison (Random Forest)", 1)
say(f"Class balance: not survived = {balance[0]} ({balance[0]/len(y):.1%}), survived = "
    f"{balance[1]} ({balance[1]/len(y):.1%}) -> imbalance ratio {balance[0]/balance[1]:.2f}:1 "
    f"(moderate, not extreme).")
say()

rf_params = dict(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)
variants = {
    "(a) Baseline (no handling)": make_pipe(RandomForestClassifier(**rf_params)),
    "(b) class_weight='balanced'": make_pipe(RandomForestClassifier(class_weight="balanced", **rf_params)),
    # imblearn Pipeline: SMOTE runs only inside .fit() on the training data it receives
    # (each CV training fold, or the training split) and is skipped at predict time.
    "(c) SMOTE (train only)": ImbPipeline([
        ("prep", build_preprocessor()),
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("clf", RandomForestClassifier(**rf_params)),
    ]),
}
imb_rows = {}
for name, pipe in variants.items():
    pipe.fit(X_train, y_train)
    m, _, _ = evaluate(pipe, X_test, y_test)
    cvres = cross_validate(pipe, X_train, y_train, cv=cv, n_jobs=1,
                           scoring=["precision", "recall", "f1"])
    imb_rows[name] = {
        "Test Precision": m["Precision"], "Test Recall": m["Recall"], "Test F1": m["F1"],
        "CV Precision": cvres["test_precision"].mean(), "CV Recall": cvres["test_recall"].mean(),
        "CV F1": cvres["test_f1"].mean(),
    }
imb = pd.DataFrame(imb_rows).T
say(md_table(imb).replace("| index |", "| Variant |"))
say()
say("*Test = one held-out split of "
    f"{len(y_test)} rows (noisy). CV = 5-fold stratified CV on the training split only; "
    "for (c) SMOTE is re-applied inside every training fold and never touches the "
    "validation fold, so there is no leakage.*")
say()
best_imb = imb["CV F1"].idxmax()
base_row, best_row = imb.loc["(a) Baseline (no handling)"], imb.loc[best_imb]
say(f"**Conclusion.** Ranked by cross-validated F1, the best strategy is **{best_imb}** "
    f"(CV F1 {best_row['CV F1']:.3f}, recall {best_row['CV Recall']:.3f}, precision "
    f"{best_row['CV Precision']:.3f}) versus baseline (CV F1 {base_row['CV F1']:.3f}, recall "
    f"{base_row['CV Recall']:.3f}, precision {base_row['CV Precision']:.3f}). Because the "
    f"minority class is ~{minority_share:.0%} of the data (moderate imbalance) and a "
    f"Random Forest already copes reasonably, differences between strategies are small. "
    f"Relative to baseline, class_weight changes CV recall by "
    f"{imb.iloc[1]['CV Recall'] - base_row['CV Recall']:+.3f} and precision by "
    f"{imb.iloc[1]['CV Precision'] - base_row['CV Precision']:+.3f}, while SMOTE changes CV recall by "
    f"{imb.iloc[2]['CV Recall'] - base_row['CV Recall']:+.3f} and precision by "
    f"{imb.iloc[2]['CV Precision'] - base_row['CV Precision']:+.3f}: rebalancing mainly shifts the "
    f"precision/recall trade-off rather than delivering a large F1 gain. With only {len(y_test)} test rows, gaps of 1-2 F1 points on the test "
    f"column are within noise, which is why the CV columns drive this conclusion. "
    f"SMOTE is also imperfect here: it interpolates between rows in a space that "
    f"includes one-hot columns, so its synthetic passengers can have fractional 'sex'/'embarked' values.")
say()


# TASK 12 - HYPERPARAMETER TUNING (run now so the tuned RF joins the comparison)

heading("Task 12 - Hyperparameter tuning of Random Forest (GridSearchCV, OOB score)", 1)
rf_for_grid = make_pipe(RandomForestClassifier(oob_score=True, random_state=RANDOM_STATE, n_jobs=-1))
param_grid = {
    "clf__n_estimators": [100, 300, 500],
    "clf__max_depth": [3, 5, 8, None],
    "clf__max_features": ["sqrt", "log2", None],
}
grid = GridSearchCV(rf_for_grid, param_grid, cv=cv, scoring="roc_auc", n_jobs=1, refit=True)
grid.fit(X_train, y_train)  # preprocessing is re-fit inside every CV fold, train rows only
best_rf_oob = grid.best_estimator_.named_steps["clf"].oob_score_
say("Estimator: `RandomForestClassifier(oob_score=True, random_state=42)` inside the same "
    "preprocessing Pipeline. Grid: `n_estimators` in [100, 300, 500], `max_depth` in "
    "[3, 5, 8, None], `max_features` in ['sqrt', 'log2', None]; 5-fold stratified CV, "
    "scoring = ROC-AUC (threshold-free).")
say()
say(f"- **Best parameters:** `{ {k.replace('clf__', ''): v for k, v in grid.best_params_.items()} }`")
say(f"- **Best CV ROC-AUC:** {grid.best_score_:.4f}")
say(f"- **OOB score of the best model (`oob_score_`, accuracy on out-of-bag rows):** {best_rf_oob:.4f}")
say()
models["Random Forest (tuned)"] = grid.best_estimator_

# evaluate the tuned model on the same held-out test split
m, p_, pr_ = evaluate(grid.best_estimator_, X_test, y_test)
rows["Random Forest (tuned)"], preds["Random Forest (tuned)"], probas["Random Forest (tuned)"] = m, p_, pr_
say("Tuned model on the held-out test set: " + ", ".join(f"{k} = {m[k]:.3f}" for k in metric_cols) + ".")
say()


# Final classifier comparison + selection on TRAIN-ONLY cross-validation

cv_auc = {n: cross_val_score(clone(p), X_train, y_train, cv=cv, scoring="roc_auc").mean()
          for n, p in models.items()}


# TASK 13 - REGRESSION SIDE-TASK: predict fare

heading("Task 13 - Regression side-task: predict `fare` (multivariate linear regression)", 1)
INCLUDE_SURVIVED_IN_REGRESSION = False  # flip to True to add the outcome as a feature
reg_num = ["pclass", "age", "sibsp", "parch"] + (["survived"] if INCLUDE_SURVIVED_IN_REGRESSION else [])
reg_cat = ["sex", "embarked"]
say(f"Target: `fare`. Features: {reg_num + reg_cat}. `survived` is excluded because it "
    f"is the classification target rather than a passenger attribute; `deck`, `class`, "
    f"`who`, `adult_male`, `alone`, `embark_town`, `alive` are derived duplicates. "
    f"Same imputer/encoder/scaler discipline (fit on train only) via a Pipeline; "
    f"`OneHotEncoder(drop='first')` avoids the dummy-variable trap for OLS.")
say()
Xr, yr = df[reg_num + reg_cat], df["fare"]
Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(Xr, yr, test_size=0.20, random_state=RANDOM_STATE)

reg_prep = ColumnTransformer([
    ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), reg_num),
    ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                      ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="first"))]), reg_cat),
], verbose_feature_names_out=False)
reg_pipe = Pipeline([("prep", reg_prep), ("lin", LinearRegression())]).fit(Xr_tr, yr_tr)

yr_hat = reg_pipe.predict(Xr_te)
n, p_feats = len(yr_te), reg_pipe.named_steps["prep"].transform(Xr_te).shape[1]
mae = mean_absolute_error(yr_te, yr_hat)
rmse = float(np.sqrt(mean_squared_error(yr_te, yr_hat)))
r2 = r2_score(yr_te, yr_hat)
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p_feats - 1)
say(f"**Test metrics (n={n}, p={p_feats} encoded predictors):** MAE = {mae:.3f}, "
    f"RMSE = {rmse:.3f}, R² = {r2:.3f}, Adjusted R² = {adj_r2:.3f}  "
    f"(Adjusted R² = 1 - (1-R²)(n-1)/(n-p-1)).")
say()

resid = yr_te.values - yr_hat
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
axes[0].scatter(yr_hat, resid, alpha=0.6, edgecolor="k", linewidth=0.3)
axes[0].axhline(0, color="red", linestyle="--")
axes[0].set_xlabel("Predicted fare")
axes[0].set_ylabel("Residual (actual - predicted)")
axes[0].set_title("Residual plot")
sns.histplot(resid, bins=30, kde=True, ax=axes[1], color="#4C72B0")
axes[1].set_title("Residual distribution")
save_fig("14_regression_residuals.png")

# Heteroscedasticity: Breusch-Pagan (manual: regress e^2 on X; LM = n*R2 ~ chi2(p))
Xt_te = reg_pipe.named_steps["prep"].transform(Xr_te)
aux_r2 = LinearRegression().fit(Xt_te, resid ** 2).score(Xt_te, resid ** 2)
bp_lm = n * aux_r2
bp_p = stats.chi2.sf(bp_lm, p_feats)
rho, rho_p = stats.spearmanr(np.abs(resid), yr_hat)
order = np.argsort(yr_hat)
lo_sd, hi_sd = resid[order[: n // 2]].std(), resid[order[n // 2:]].std()
bp_reject, rho_reject = bp_p < 0.05, rho_p < 0.05
hetero = bp_reject or rho_reject or (hi_sd > 1.5 * lo_sd)
say(f"**Conclusion:** the residual plot **{'shows' if hetero else 'does not show'} "
    f"heteroscedasticity** - the residual spread is not constant: it is "
    f"{hi_sd / lo_sd:.1f}x wider for the upper half of predicted fares than the lower half "
    f"(a fan/funnel shape), and |residual| rises with the prediction "
    f"(Spearman {rho:+.2f}, p {'<' if rho_p < 0.001 else '='} {max(rho_p, 0.001):.3g}). "
    f"Breusch-Pagan {'also rejects' if bp_reject else 'does NOT reject'} constant variance "
    f"(p = {bp_p:.3f}); BP only tests a linear link between the features and the squared "
    f"residuals, so it can miss this pattern, which grows with the *prediction* itself. "
    f"Fare is strongly right-skewed (a few fares above 200), so an untransformed "
    f"linear model fits cheap tickets tightly and expensive ones poorly, and can even "
    f"predict negative fares. Modelling log(fare) or using weighted/robust regression is "
    f"the natural fix; R² is only {r2:.2f}, i.e. these attributes explain a limited "
    f"share of fare variance.")
say()


# TASK 14 - MODEL COMPARISON TABLE + RECOMMENDATION

heading("Task 14 - Model comparison and recommendation", 1)
say("Classification and regression metrics are **on different scales and not comparable**; "
    "they are shown as two separate metric groups, one per model type. Cells that do not "
    "apply to a model type are blank (—).")
say()
cmp = pd.DataFrame(index=list(models) + ["Linear Regression (fare)"],
                   columns=pd.MultiIndex.from_tuples(
                       [("Classification metrics (survived)", m) for m in metric_cols] +
                       [("Regression metrics (fare)", m) for m in ["MAE", "RMSE", "R²", "Adjusted R²"]]),
                   dtype=float)
for name in models:
    for m in metric_cols:
        cmp.loc[name, ("Classification metrics (survived)", m)] = rows[name][m]
for m, v in zip(["MAE", "RMSE", "R²", "Adjusted R²"], [mae, rmse, r2, adj_r2]):
    cmp.loc["Linear Regression (fare)", ("Regression metrics (fare)", m)] = v
cmp.to_csv(RES / "model_comparison.csv")

flat = cmp.copy()
flat.columns = [("[CLS] " if a.startswith("Class") else "[REG] ") + b for a, b in cmp.columns]
say(md_table(flat).replace("| index |", "| Model |"))
say()
say("`[CLS]` = classification metrics (higher is better, bounded 0-1). `[REG]` = regression "
    "metrics (MAE/RMSE in fare units - lower is better; R²/Adjusted R² unitless). "
    "Saved as `results/model_comparison.csv` with a two-level column header.")
say()

# ---- choose the deployable classifier using TRAIN-ONLY CV AUC (no test peeking) ----
cv_series = pd.Series(cv_auc).sort_values(ascending=False)
say("Selection rule: highest 5-fold CV ROC-AUC **on the training split** (the test set is "
    "not used to choose). " + ", ".join(f"{k} = {v:.3f}" for k, v in cv_series.items()) + ".")
say()
best_name = cv_series.index[0]
runner = cv_series.index[1]
b, r_ = rows[best_name], rows[runner]
lr_ = rows["Logistic Regression"]
dt_ = rows["Decision Tree"]
say("**Final recommendation.**")
lr_better_on_test = lr_["AUC"] > b["AUC"]
say(f"I would deploy **{best_name}**, which had the best 5-fold training CV ROC-AUC "
    f"({cv_series.iloc[0]:.3f}) and on the held-out test set reached accuracy {b['Accuracy']:.3f}, "
    f"precision {b['Precision']:.3f}, recall {b['Recall']:.3f}, F1 {b['F1']:.3f} and AUC {b['AUC']:.3f}. "
    f"The runner-up, {runner}, scored F1 {r_['F1']:.3f} / AUC {r_['AUC']:.3f} on test, and the Decision Tree "
    f"(F1 {dt_['F1']:.3f}, AUC {dt_['AUC']:.3f}) is the most interpretable but the least stable. "
    + (f"Note that Logistic Regression actually has the higher test AUC ({lr_['AUC']:.3f} vs {b['AUC']:.3f}) "
       f"and a similar F1 ({lr_['F1']:.3f}), so its edge is small and rests on CV, accuracy and precision "
       f"rather than a clear-cut win; Logistic Regression is the simpler, more transparent fallback. "
       if lr_better_on_test else
       f"Logistic Regression, the transparent baseline, trails on test AUC ({lr_['AUC']:.3f} vs {b['AUC']:.3f}) "
       f"with F1 {lr_['F1']:.3f}, so the ensemble's advantage is worth its lower interpretability. ")
    + f"Recall of {b['Recall']:.3f} means about {1 - b['Recall']:.0%} of actual survivors are still missed, "
      f"and with only {len(y_test)} test rows differences of a few points are within sampling noise, "
      f"so lower the decision threshold if missed survivors are costlier and re-validate on new data.")
say()


# TASK 15 - SAVE THE COMPLETE FITTED PIPELINE (prep + estimator) WITH JOBLIB

heading("Task 15 - Persist the best full pipeline", 1)
full_pipeline = models[best_name]  # Pipeline([('prep', ColumnTransformer), ('clf', estimator)])
assert isinstance(full_pipeline, Pipeline) and "prep" in full_pipeline.named_steps
joblib.dump(full_pipeline, BASE / "best_pipeline.joblib")

# raw hold-out rows (+ target) so 03_reload_pipeline.py can prove end-to-end reuse
holdout = X_test.copy()
holdout[TARGET] = y_test.values
holdout.to_csv(RES / "holdout_test_raw.csv", index=False)
meta = {"best_model": best_name, "features": FEATURES,
        "test_accuracy": float(b["Accuracy"]), "test_auc": float(b["AUC"]),
        "n_test": int(len(y_test)),
        "test_predictions": [int(v) for v in preds[best_name]]}
(RES / "best_model_meta.json").write_text(json.dumps(meta, indent=2))

say(f"Saved `best_pipeline.joblib` = fitted `Pipeline(prep=ColumnTransformer[imputer/encoder/scaler], "
    f"clf={type(full_pipeline.named_steps['clf']).__name__})` for **{best_name}**. It expects RAW "
    f"columns {FEATURES} (missing `age` allowed) and needs no manual preprocessing.")
say()
say("Reload check (also runnable on its own): `python analytics/03_reload_pipeline.py`.")

# in-process sanity check right away
reloaded = joblib.load(BASE / "best_pipeline.joblib")
assert (reloaded.predict(X_test) == preds[best_name]).all()
say("In-script check: `joblib.load(...)` reproduces the exact test predictions on raw input: **OK**.")

(REP / "modeling_report.md").write_text("\n".join(_report), encoding="utf-8")
print(f"\nDone. Report: {REP / 'modeling_report.md'}")
