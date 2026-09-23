# MODULE 2

from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: figures are saved, never shown
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parent
FIG = BASE / "figures"
REP = BASE / "reports"
FIG.mkdir(exist_ok=True)
REP.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", context="notebook")


#  print AND collect everything for the markdown report

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


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"



#  TASK 1: LOAD ONCE, SAVE OFFLINE FALLBACK, PROFILE

heading("Task 1 - Load, save offline fallback, profile", 1)

# The ONE AND ONLY load of the raw dataset in the whole module.
df_raw = sns.load_dataset("titanic")

# Immediately save the loaded DataFrame as the offline fallback.
df_raw.to_csv(BASE / "titanic.csv", index=False)
say("Raw dataset loaded once with `sns.load_dataset('titanic')` and saved to "
    "`titanic.csv` (`df.to_csv('titanic.csv', index=False)`).")
say()

say(f"**df.shape:** `{df_raw.shape}`")
say()
say("**df.info():**")
say()
import io

buf = io.StringIO()
df_raw.info(buf=buf)
say("```\n" + buf.getvalue() + "```")
say()
say("**df.describe():**")
say()
say("```\n" + df_raw.describe().round(3).to_string() + "\n```")
say()

# ---- % missing per column (raw data, measured BEFORE any handling) ---------
missing_pct = (df_raw.isna().mean() * 100)
missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=False)
say("**Percentage of missing values (columns with any):**")
say()
say("| column | missing count | missing % |")
say("|---|---|---|")
for col, p in missing_pct.items():
    say(f"| {col} | {int(df_raw[col].isna().sum())} | {p:.2f}% |")
say()

#  TASK 2: MISSING-VALUE HANDLING (threshold rule)

heading("Task 2 - Missing-value handling", 1)
say("Threshold rule: **< 5 % missing -> drop those rows; 5 %-30 % -> impute; "
    "> 30 % -> imputation is unreliable, so explicitly drop the column or encode "
    "'Missing' as its own category.**")
say()

df = df_raw.copy()

# Decisions for columns above the 30 % ceiling (must be justified in writing)
HIGH_MISSING_DECISION = {
    "deck": ("category",
             "Missingness in `deck` is itself informative: cabin numbers were mostly "
             "recorded for first-class passengers, so 'no deck recorded' is a proxy "
             "for lower class/fare. Dropping the column would throw that signal away, "
             "and imputing ~77 % of the values would mean inventing most of the column. "
             "Chosen: keep the column and encode NaN as its own category 'Missing'."),
}

rows_to_drop = pd.Series(False, index=df.index)
for col, p in missing_pct.items():
    say(f"- **`{col}`: measured missing = {p:.2f}%**")
    if p < 5:
        rows_to_drop |= df_raw[col].isna()
        say(f"  - {p:.2f}% < 5%  ->  rule: *drop rows* with missing `{col}` "
            f"({int(df_raw[col].isna().sum())} rows).")
    elif p <= 30:
        if pd.api.types.is_numeric_dtype(df[col]):
            fill = float(df[col].median())
            why = ("median chosen over mean because the distribution is mildly "
                   "right-skewed and the median is robust to outliers")
        else:
            fill = str(df[col].mode().iloc[0])
            why = "mode used for a categorical column"
        # keep a flag so later plots can exclude imputed values
        df[f"{col}_was_missing"] = df[col].isna()
        df[col] = df[col].fillna(fill)
        say(f"  - 5% <= {p:.2f}% <= 30%  ->  rule: *impute*; filled with {fill!r} ({why}).")
    else:
        action, reason = HIGH_MISSING_DECISION[col]
        if action == "category":
            df[col] = df[col].astype("object").fillna("Missing")
        else:
            df = df.drop(columns=[col])
        say(f"  - {p:.2f}% > 30%  ->  rule: imputation unreliable; decision = "
            f"**{action}**.")
        say(f"  - Justification: {reason}")

n_before = len(df)
df = df.loc[~rows_to_drop].reset_index(drop=True)
say()
say(f"Rows dropped by the <5 % rule: {n_before - len(df)} "
    f"(`embarked` and `embark_town` are missing on the same rows, so they are "
    f"dropped once). Cleaned shape: `{df.shape}`; remaining NaNs in original "
    f"columns: {int(df[df_raw.columns].isna().sum().sum())}.")
say()


# TASK 3: UNIVARIATE ANALYSIS

heading("Task 3 - Univariate analysis (age, fare)", 1)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for j, col in enumerate(["age", "fare"]):
    sns.histplot(df[col], bins=30, kde=True, ax=axes[0, j], color="#4C72B0")
    axes[0, j].set_title(f"Histogram of {col}")
    sns.boxplot(x=df[col], ax=axes[1, j], color="#DD8452")
    axes[1, j].set_title(f"Box plot of {col}")
save_fig("03_univariate_age_fare.png")


def iqr_outliers(s: pd.Series):
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return q1, q3, iqr, lo, hi, int(((s < lo) | (s > hi)).sum())


say("**IQR-rule outliers** (outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR]):")
say()
say("| column | Q1 | Q3 | IQR | lower fence | upper fence | # outliers |")
say("|---|---|---|---|---|---|---|")
for col in ["age", "fare"]:
    q1, q3, iqr, lo, hi, n_out = iqr_outliers(df[col])
    say(f"| {col} | {q1:.2f} | {q3:.2f} | {iqr:.2f} | {lo:.2f} | {hi:.2f} | **{n_out}** |")
say()

fare_mean, fare_median = df["fare"].mean(), df["fare"].median()
fare_mode = df["fare"].mode().iloc[0]
say(f"**Fare central tendency:** mean = {fare_mean:.3f}, median = {fare_median:.3f}, "
    f"mode = {fare_mode:.3f}.")
say()
if fare_mean > fare_median > fare_mode:
    skew_word = "right-skewed (positively skewed)"
    ordering = "mean > median > mode"
elif fare_mean < fare_median < fare_mode:
    skew_word = "left-skewed (negatively skewed)"
    ordering = "mean < median < mode"
elif np.isclose(fare_mean, fare_median, rtol=0.02):
    skew_word, ordering = "approximately symmetric", "mean ~ median"
else:
    skew_word = "right-skewed" if fare_mean > fare_median else "left-skewed"
    ordering = "mean vs median ordering"
say(f"**Conclusion:** the fare distribution is **{skew_word}** ({ordering}: "
    f"{fare_mean:.2f} vs {fare_median:.2f} vs {fare_mode:.2f}). A minority of very "
    f"expensive first-class tickets (max fare {df['fare'].max():.1f}) drags the mean "
    f"far above the median, while the mode sits at the cheap end where most "
    f"third-class passengers paid. Sample skewness = {df['fare'].skew():.2f}.")
say()


#  TASK 4: BIVARIATE ANALYSIS

heading("Task 4 - Bivariate analysis", 1)

# --- boolean masks (combined with & and |) ---
male = df["sex"] == "male"
female = df["sex"] == "female"
c1, c2, c3 = df["pclass"] == 1, df["pclass"] == 2, df["pclass"] == 3


def rate(mask: pd.Series) -> tuple[float, int]:
    sub = df.loc[mask, "survived"]
    return sub.mean(), len(sub)


say("**(a) Survival rate by sex**")
say()
say("| group | n | survival rate |")
say("|---|---|---|")
for name, m in [("female", female), ("male", male)]:
    r, n = rate(m)
    say(f"| {name} | {n} | {pct(r)} |")
say()

say("**(b) Survival rate by pclass**")
say()
say("| group | n | survival rate |")
say("|---|---|---|")
for name, m in [("1st class", c1), ("2nd class", c2), ("3rd class", c3)]:
    r, n = rate(m)
    say(f"| {name} | {n} | {pct(r)} |")
r_12, n_12 = rate(c1 | c2)  # example of the | combination
say(f"| 1st **or** 2nd (`c1 | c2`) | {n_12} | {pct(r_12)} |")
say()

say("**(c) Survival rate by sex AND pclass** (masks combined with `&`)")
say()
say("| group | n | survival rate |")
say("|---|---|---|")
combo_rates = {}
for sname, sm in [("female", female), ("male", male)]:
    for cname, cm in [("1st", c1), ("2nd", c2), ("3rd", c3)]:
        r, n = rate(sm & cm)
        combo_rates[(sname, cname)] = r
        say(f"| {sname} & {cname} class | {n} | {pct(r)} |")
say()

# --- correlation matrix on EXACTLY six columns ---
CORR_COLS = ["survived", "pclass", "age", "sibsp", "parch", "fare"]
corr = df[CORR_COLS].corr()
say("**Correlation matrix (exactly 6 columns; `adult_male` and `alone` deliberately "
    "excluded - they are derived/redundant flags computable from sex/age and "
    "sibsp+parch):**")
say()
say("```\n" + corr.round(3).to_string() + "\n```")
say()

plt.figure(figsize=(7, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1,
            square=True, linewidths=0.5)
plt.title("Correlation matrix (6 numeric columns)")
save_fig("04_correlation_heatmap.png")

pairs = sorted(
    ((a, b, corr.loc[a, b]) for a, b in combinations(CORR_COLS, 2)),
    key=lambda t: abs(t[2]), reverse=True,
)
say("**Two strongest correlations (largest |r| among off-diagonal pairs):**")
say()
for rank, (a, b, r) in enumerate(pairs[:2], 1):
    direction = "negative" if r < 0 else "positive"
    say(f"{rank}. `{a}` vs `{b}`: r = {r:+.3f} ({direction}).")
say()
INTERP = {
    frozenset(["pclass", "fare"]):
        "Lower class number (1st class) means a higher ticket price, so as pclass "
        "rises the fare falls - a strong negative link that mirrors the ticket "
        "structure of the ship.",
    frozenset(["pclass", "age"]):
        "Older passengers tended to travel in better (lower-numbered) classes, "
        "giving a negative correlation between pclass and age.",
    frozenset(["survived", "pclass"]):
        "Passengers in better classes survived more often, so survival falls as "
        "the class number increases.",
    frozenset(["sibsp", "parch"]):
        "Passengers travelling with siblings/spouses often also travelled with "
        "parents/children, i.e. these two counts both measure family group size and "
        "move together.",
    frozenset(["survived", "fare"]):
        "Higher fares (wealthier passengers) are mildly associated with survival.",
    frozenset(["age", "sibsp"]):
        "Younger passengers had more siblings aboard, so age and sibsp are negatively "
        "related.",
    frozenset(["age", "fare"]):
        "Older passengers paid somewhat more, reflecting wealth and cabin class.",
    frozenset(["survived", "age"]):
        "Younger passengers were slightly more likely to survive.",
}
for a, b, r in pairs[:2]:
    say(f"- **Interpretation `{a}` ~ `{b}` (r = {r:+.2f}):** "
        f"{INTERP.get(frozenset([a, b]), 'These features move together, but correlation is not causation.')}")
say()
say("*(Pearson r only captures linear association, and `survived` is binary, so "
    "correlations with it are best read as directional hints, not effect sizes.)*")
say()


#  TASK 5: MULTIVARIATE DATA STORY (>= 4 charts, each interpreted)

heading("Task 5 - Multivariate data story: who survived, and why?", 1)

known_age = df.loc[~df["age_was_missing"]].copy()  # avoid plotting imputed ages

# ---- Chart 1: survival by sex x class --------------------------------------
heading("Chart 1 - Survival rate by sex and passenger class", 3)
plot_df = (df.groupby(["pclass", "sex"])["survived"].mean().reset_index())
plt.figure(figsize=(8, 5))
ax = sns.barplot(data=plot_df, x="pclass", y="survived", hue="sex",
                 palette={"female": "#C44E52", "male": "#4C72B0"})
for cont in ax.containers:
    ax.bar_label(cont, fmt="%.2f", padding=2)
plt.ylim(0, 1.05)
plt.ylabel("Survival rate")
plt.xlabel("Passenger class")
plt.title("Survival rate by class and sex")
save_fig("05_story1_sex_class.png")
say(f"**Interpretation.** Sex and class together are the two dominant drivers. Women "
    f"in 1st class survived at {pct(combo_rates[('female','1st')])} and in 2nd class at "
    f"{pct(combo_rates[('female','2nd')])}, while men in 3rd class survived at only "
    f"{pct(combo_rates[('male','3rd')])}. Within every class women outlived men, "
    f"consistent with a 'women and children first' evacuation order, and the "
    f"female-3rd-class rate ({pct(combo_rates[('female','3rd')])}) shows that "
    f"class disadvantage still cut into women's chances.")

# ---- Chart 2: age distribution by survival, faceted by sex -----------------
heading("Chart 2 - Age distribution of survivors vs non-survivors, by sex", 3)
g = sns.displot(data=known_age, x="age", hue="survived", col="sex", bins=20,
                multiple="stack", palette={0: "#8172B2", 1: "#55A868"}, height=4.5,
                aspect=1.2)
g.figure.suptitle("Age by survival and sex (passengers with recorded age)", y=1.03)
g.figure.savefig(FIG / "06_story2_age_survival.png", dpi=150, bbox_inches="tight")
plt.close(g.figure)
say("*(figure saved: `figures/06_story2_age_survival.png`)*")
say()
child = known_age["age"] < 13
child_rate = known_age.loc[child, "survived"].mean()
adult_rate = known_age.loc[~child, "survived"].mean()
boy_rate = known_age.loc[child & (known_age.sex == "male"), "survived"].mean()
man_rate = known_age.loc[~child & (known_age.sex == "male"), "survived"].mean()
say(f"**Interpretation.** Children under 13 survived at {pct(child_rate)} versus "
    f"{pct(adult_rate)} for everyone else. The effect is strongest for males: boys "
    f"under 13 survived at {pct(boy_rate)} against {pct(man_rate)} for adult men "
    f"(13+). Women and girls of every age band survived at well over half "
    f"(see Chart 5), so age matters mainly as a modifier for males. Only passengers with a recorded age are "
    f"plotted ({len(known_age)} of {len(df)}) so the imputed median does not create "
    f"a fake spike.")

# ---- Chart 3: fare by class and survival ----------------------------------
heading("Chart 3 - Fare paid by class and survival", 3)
plt.figure(figsize=(9, 5))
sns.boxplot(data=df, x="pclass", y="fare", hue="survived",
            palette={0: "#8172B2", 1: "#55A868"})
plt.yscale("symlog", linthresh=10)  # symlog handles the zero fares
plt.title("Fare by class and survival (symlog scale)")
plt.ylabel("Fare (symlog)")
save_fig("07_story3_fare_class_survival.png")
med = df.groupby("survived")["fare"].median()
med_c = df.groupby(["pclass", "survived"])["fare"].median().unstack()
say(f"**Interpretation.** Overall, survivors paid a median fare of {med[1]:.1f} versus "
    f"{med[0]:.1f} for non-survivors, which looks like a wealth effect. But the boxes "
    f"show most of that gap is really a class effect: within 1st class the medians "
    f"are {med_c.loc[1, 1]:.1f} (survived) vs {med_c.loc[1, 0]:.1f} (died) and in 3rd "
    f"class {med_c.loc[3, 1]:.1f} vs {med_c.loc[3, 0]:.1f}: the gap shrinks sharply "
    f"once class is held fixed, and is almost gone in 3rd class. Ticket price acts "
    f"largely as a proxy for cabin location and status rather than an independent cause.")

# ---- Chart 4: survival vs family size ------
heading("Chart 4 - Survival rate by family size", 3)
df["family_size"] = df["sibsp"] + df["parch"] + 1
fam = df.groupby("family_size")["survived"].agg(["mean", "count"]).reset_index()
plt.figure(figsize=(9, 5))
ax = sns.barplot(data=fam, x="family_size", y="mean", color="#4C72B0")
for i, (_, row) in enumerate(fam.iterrows()):
    ax.text(i, row["mean"] + 0.02, f"n={int(row['count'])}", ha="center", fontsize=9)
plt.ylim(0, 1.05)
plt.ylabel("Survival rate")
plt.xlabel("Family size (self + siblings/spouse + parents/children)")
plt.title("Survival rate by family size")
save_fig("08_story4_family_size.png")
solo = df.loc[df.family_size == 1, "survived"].mean()
small = df.loc[df.family_size.between(2, 4), "survived"].mean()
large = df.loc[df.family_size >= 5, "survived"].mean()
say(f"**Interpretation.** Passengers travelling alone survived at {pct(solo)}, small "
    f"families of 2-4 at {pct(small)} and families of 5+ at only {pct(large)}. Small "
    f"groups could look after one another and reach lifeboats together, whereas "
    f"large families (mostly 3rd class) struggled to keep everyone together. Several "
    f"family-size groups are small (see the n labels), so those bars are noisy.")

# ---- Chart 5: survival heatmap age group x sex ---
heading("Chart 5 - Survival heatmap: age group x sex", 3)
bins = [0, 12, 19, 39, 59, 120]
labels = ["Child (0-12)", "Teen (13-19)", "Adult (20-39)", "Middle (40-59)", "Senior (60+)"]
known_age["age_group"] = pd.cut(known_age["age"], bins=bins, labels=labels)
pivot = known_age.pivot_table(index="age_group", columns="sex", values="survived",
                              aggfunc="mean", observed=True)
cnt = known_age.pivot_table(index="age_group", columns="sex", values="survived",
                            aggfunc="count", observed=True)
annot = pivot.round(2).astype(str) + "\n(n=" + cnt.astype(int).astype(str) + ")"
plt.figure(figsize=(7, 5.5))
sns.heatmap(pivot, annot=annot, fmt="", cmap="RdYlGn", vmin=0, vmax=1,
            linewidths=0.5)
plt.title("Survival rate by age group and sex")
plt.ylabel("")
save_fig("09_story5_agegroup_sex_heatmap.png")
stack = pivot.stack()
best_idx, worst_idx = stack.idxmax(), stack.idxmin()
say(f"**Interpretation.** The best-off cell is {best_idx[1]} / {best_idx[0]} at "
    f"{pct(stack.max())} survival, and the worst is {worst_idx[1]} / {worst_idx[0]} at "
    f"{pct(stack.min())}. Across nearly every age band the female column is far greener "
    f"than the male one; the child row is the exception, where girls and boys are almost "
    f"level and boys do far better than adult men. Cell sizes (n) are shown because the oldest "
    f"bands hold few passengers.")

heading("Data story - putting it together", 3)
say(f"Being **female** was the strongest single predictor of survival "
    f"({pct(rate(female)[0])} vs {pct(rate(male)[0])} for men), **class** modulated it "
    f"(1st {pct(rate(c1)[0])}, 3rd {pct(rate(c3)[0])}), **age** helped mainly young boys, and "
    f"**family size** showed a sweet spot at 2-4 people. Fare mostly echoes class. "
    f"Together this describes a 'women and children first' evacuation that was "
    f"filtered through class-based access to the boat deck.")
say()

#TASK 6: EXPLORATORY STANDARDISATION CHECK (z-score)

heading("Task 6 - Exploratory z-score standardisation check (age, fare)", 1)
say("*EDA-stage sanity check only - fitted on the full cleaned DataFrame and NOT used "
    "by the modeling pipeline, which performs its own train-only scaling.*")
say()

scaler = StandardScaler()
z = pd.DataFrame(scaler.fit_transform(df[["age", "fare"]]),
                 columns=["age_z", "fare_z"], index=df.index)
# StandardScaler uses population std (ddof=0), so use ddof=0 to confirm std == 1
comp = pd.DataFrame({
    "mean (before)": df[["age", "fare"]].mean().values,
    "std (before)": df[["age", "fare"]].std(ddof=0).values,
    "mean (after)": z.mean().values,
    "std (after)": z.std(ddof=0).values,
}, index=["age", "fare"])
say("```\n" + comp.round(6).to_string() + "\n```")
say()
say("Manual check `z = (x - mean) / std` matches the scaler: "
    f"max |difference| for age = {np.abs(((df['age'] - df['age'].mean()) / df['age'].std(ddof=0)) - z['age_z']).max():.2e}, "
    f"fare = {np.abs(((df['fare'] - df['fare'].mean()) / df['fare'].std(ddof=0)) - z['fare_z']).max():.2e}.")
say()

fig, axes = plt.subplots(2, 2, figsize=(11, 7))
for j, (col, zc) in enumerate([("age", "age_z"), ("fare", "fare_z")]):
    sns.histplot(df[col], bins=30, ax=axes[0, j], color="#4C72B0")
    axes[0, j].set_title(f"{col} - before (mean={df[col].mean():.2f}, std={df[col].std(ddof=0):.2f})")
    sns.histplot(z[zc], bins=30, ax=axes[1, j], color="#55A868")
    axes[1, j].set_title(f"{col} - after z-score (mean={z[zc].mean():.2f}, std={z[zc].std(ddof=0):.2f})")
save_fig("10_standardisation_before_after.png")
say("The shape of each distribution is unchanged (z-scoring is a linear rescale); only "
    "the centre moves to 0 and the spread to 1.")

(REP / "eda_report.md").write_text("\n".join(_report), encoding="utf-8")
print(f"\nDone. Report: {REP / 'eda_report.md'}  |  Figures: {FIG}")
