from pathlib import Path
import sys
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import joblib

from IPython.display import display

from sklearn.base import clone
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    RandomizedSearchCV,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)

warnings.filterwarnings("default")

# ============================================================
# 1. CONFIGURATION
# ============================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_SPLITS = 5

# Start with 5 to reduce runtime.
# Increase to 10-20 if your computer has sufficient resources.
N_ITER = 5

OUTPUT_DIR = Path(__file__).resolve().parent / "project_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print("BANK MARKETING CLASSIFICATION PROJECT")
print("=" * 65)
print("Python:", sys.version)
print("pandas:", pd.__version__)
print("scikit-learn:", sklearn.__version__)
print("Output folder:", OUTPUT_DIR)


# ============================================================
# 2. LOAD LOCAL DATASET
# ============================================================

DATA_PATH = Path(__file__).resolve().parent / "bank-full.csv"

if not DATA_PATH.is_file():
    raise FileNotFoundError(
        "\nDataset file not found.\n"
        f"Expected location:\n{DATA_PATH}\n\n"
        "Download the UCI Bank Marketing dataset from:\n"
        "https://archive.ics.uci.edu/dataset/222/bank+marketing\n\n"
        "Extract bank-full.csv and put it beside this Python script."
    )

# The original bank-full.csv uses semicolons as separators.
df = pd.read_csv(DATA_PATH, sep=";")

# Also handle an accidental comma-separated import.
if len(df.columns) == 1:
    raise ValueError(
        "The CSV appears to have been read as one column. "
        "Check that bank-full.csv is the original semicolon-separated file."
    )

df.columns = [str(column).strip() for column in df.columns]

if "y" not in df.columns:
    raise ValueError(
        "Target column 'y' was not found. "
        f"Available columns: {df.columns.tolist()}"
    )

print("\nDataset loaded successfully.")
print("Dataset dimensions:", df.shape)
print("Column names:", df.columns.tolist())

# Normalise the target labels.
target = df["y"].astype(str).str.strip().str.lower()
valid_target = target.isin(["yes", "no"])

if not valid_target.all():
    print("Removing rows with invalid target labels:",
          int((~valid_target).sum()))
    df = df.loc[valid_target].copy()
    target = target.loc[valid_target]

y = target.map({"no": 0, "yes": 1}).astype(int).reset_index(drop=True)

# Exclude target and duration.
# Duration may reveal information only available after the call.
X = df.drop(columns=["y", "duration"], errors="ignore").reset_index(drop=True)

if X.empty or len(X) != len(y):
    raise ValueError("Feature and target data are empty or inconsistent.")

print("Feature matrix:", X.shape)
print("Target vector:", y.shape)
print("\nTarget counts:")
print(y.map({0: "no", 1: "yes"}).value_counts())
print("\nTarget proportions:")
print(y.value_counts(normalize=True).round(4))


# ============================================================
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================

print("\n" + "=" * 65)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 65)

print("\nData types:")
display(X.dtypes.astype(str).value_counts().to_frame("count"))

print("\nDuplicate feature rows:", int(X.duplicated().sum()))

print("\nMissing values by feature:")
display(
    X.isna().sum()
    .sort_values(ascending=False)
    .to_frame("missing_count")
)

print("\nDescriptive statistics:")
display(X.describe(include="all").T)

# Count literal "unknown" category values.
unknown_counts = {}

for column in X.select_dtypes(
    include=["object", "category"]
).columns:
    unknown_counts[column] = int(
        X[column]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("unknown")
        .sum()
    )

print("\nUnknown categorical values:")
display(
    pd.Series(unknown_counts, name="unknown_count")
    .sort_values(ascending=False)
    .to_frame()
)

# Target distribution.
fig, ax = plt.subplots(figsize=(7, 5))
y.map({0: "No", 1: "Yes"}).value_counts().plot(
    kind="bar", ax=ax
)
ax.set_title("Bank Marketing Target Distribution")
ax.set_xlabel("Term deposit subscription")
ax.set_ylabel("Number of records")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "target_distribution.png", dpi=150)
plt.show()

# Numeric feature distributions.
numeric_eda = X.select_dtypes(include=np.number).columns.tolist()

if numeric_eda:
    X[numeric_eda].hist(
        figsize=(14, 10),
        bins=30,
        edgecolor="black"
    )
    plt.suptitle("Numeric Feature Distributions")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "numeric_distributions.png", dpi=150)
    plt.show()

    plt.figure(figsize=(10, 7))
    sns.heatmap(
        X[numeric_eda].corr(),
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0
    )
    plt.title("Correlation Between Numeric Features")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "numeric_correlations.png", dpi=150)
    plt.show()

# Target subscription rates for selected categorical features.
categorical_eda = X.select_dtypes(
    exclude=np.number
).columns.tolist()

for column in categorical_eda[:6]:
    rates = (
        pd.DataFrame({
            "category": X[column].astype(str),
            "subscribed": y
        })
        .groupby("category")["subscribed"]
        .agg(["count", "mean"])
        .sort_values("count", ascending=False)
        .head(15)
    )

    print(f"\nSubscription rates by {column}:")
    display(rates)


# ============================================================
# 4. TRAIN/TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

numeric_features = X_train.select_dtypes(
    include=np.number
).columns.tolist()

categorical_features = [
    column for column in X_train.columns
    if column not in numeric_features
]

print("\n" + "=" * 65)
print("DATA PREPARATION")
print("=" * 65)
print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("Numeric features:", numeric_features)
print("Categorical features:", categorical_features)

# Preprocessing is fitted inside each pipeline to avoid data leakage.
numeric_scaled = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

numeric_tree = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

# Dense output is compatible with HistGradientBoostingClassifier.
categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    ))
])

preprocess_lr = ColumnTransformer([
    ("numeric", numeric_scaled, numeric_features),
    ("categorical", categorical_pipe, categorical_features)
])

preprocess_tree = ColumnTransformer([
    ("numeric", numeric_tree, numeric_features),
    ("categorical", categorical_pipe, categorical_features)
])


# ============================================================
# 5. DEFINE THREE BASE CLASSIFICATION MODELS
# ============================================================

models_and_spaces = {
    "Logistic Regression": (
        Pipeline([
            ("preprocess", preprocess_lr),
            ("model", LogisticRegression(
                max_iter=2000,
                random_state=RANDOM_STATE
            ))
        ]),
        {
            "model__C": [0.01, 0.1, 1.0, 10.0, 100.0],
            "model__class_weight": [None, "balanced"]
        }
    ),

    "Random Forest": (
        Pipeline([
            ("preprocess", preprocess_tree),
            ("model", RandomForestClassifier(
                random_state=RANDOM_STATE,
                n_jobs=1
            ))
        ]),
        {
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [None, 10, 20],
            "model__min_samples_leaf": [1, 2, 5],
            "model__max_features": ["sqrt", 0.5],
            "model__class_weight": [None, "balanced"]
        }
    ),

    "Histogram Gradient Boosting": (
        Pipeline([
            ("preprocess", preprocess_tree),
            ("model", HistGradientBoostingClassifier(
                random_state=RANDOM_STATE
            ))
        ]),
        {
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__max_iter": [100, 150, 200],
            "model__max_leaf_nodes": [15, 31],
            "model__min_samples_leaf": [10, 20, 30],
            "model__l2_regularization": [0.0, 0.1, 1.0]
        }
    )
}


# ============================================================
# 6. HYPERPARAMETER OPTIMISATION
# ============================================================

print("\n" + "=" * 65)
print("HYPERPARAMETER OPTIMISATION")
print("=" * 65)

cv = StratifiedKFold(
    n_splits=CV_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

scoring = {
    "f1": "f1",
    "roc_auc": "roc_auc",
    "precision": "precision",
    "recall": "recall",
    "accuracy": "accuracy"
}

searches = {}
best_estimators = {}
search_rows = []

for name, (pipeline, parameter_space) in models_and_spaces.items():
    print(f"\nTuning model: {name}")

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=parameter_space,
        n_iter=N_ITER,
        scoring=scoring,
        refit="f1",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
        return_train_score=False,
        error_score="raise"
    )

    start_time = time.perf_counter()
    search.fit(X_train, y_train)
    elapsed = time.perf_counter() - start_time

    searches[name] = search
    best_estimators[name] = search.best_estimator_

    best_index = search.best_index_

    search_rows.append({
        "model": name,
        "best_cv_f1": search.best_score_,
        "best_cv_roc_auc":
            search.cv_results_["mean_test_roc_auc"][best_index],
        "best_cv_precision":
            search.cv_results_["mean_test_precision"][best_index],
        "best_cv_recall":
            search.cv_results_["mean_test_recall"][best_index],
        "search_seconds": elapsed,
        "best_parameters": str(search.best_params_)
    })

    print("Best cross-validation F1:", round(search.best_score_, 4))
    print("Best parameters:", search.best_params_)

tuning_results = pd.DataFrame(search_rows).sort_values(
    "best_cv_f1", ascending=False
)

print("\nHyperparameter search summary:")
display(tuning_results.round(4))

tuning_results.to_csv(
    OUTPUT_DIR / "hyperparameter_search_results.csv",
    index=False
)


# ============================================================
# 7. SOFT-VOTING ENSEMBLE
# ============================================================

print("\n" + "=" * 65)
print("ENSEMBLE MODEL")
print("=" * 65)

# Select the two models with the strongest training CV F1.
# The test set is not used for this selection.
ranked_names = sorted(
    best_estimators.keys(),
    key=lambda model_name: searches[model_name].best_score_,
    reverse=True
)

selected_names = ranked_names[:2]
print("Selected models based on CV F1:", selected_names)

voting_model = VotingClassifier(
    estimators=[
        (f"model_{index + 1}", clone(best_estimators[name]))
        for index, name in enumerate(selected_names)
    ],
    voting="soft",
    n_jobs=-1
)

ensemble_start = time.perf_counter()
voting_model.fit(X_train, y_train)
print(
    "Ensemble training time:",
    round(time.perf_counter() - ensemble_start, 2),
    "seconds"
)


# ============================================================
# 8. EVALUATE MODELS ON THE HELD-OUT TEST SET
# ============================================================

print("\n" + "=" * 65)
print("MODEL EVALUATION")
print("=" * 65)

final_models = dict(best_estimators)
final_models["Soft Voting Ensemble"] = voting_model

evaluation_rows = []
predictions = {}
probabilities = {}
confusion_matrices = {}

for name, model in final_models.items():
    start_time = time.perf_counter()

    predicted = model.predict(X_test)
    predict_seconds = time.perf_counter() - start_time

    predicted_probability = model.predict_proba(X_test)[:, 1]

    predictions[name] = predicted
    probabilities[name] = predicted_probability

    cm = confusion_matrix(
        y_test, predicted, labels=[0, 1]
    )
    confusion_matrices[name] = cm

    evaluation_rows.append({
        "model": name,
        "accuracy": accuracy_score(y_test, predicted),
        "precision_yes": precision_score(
            y_test, predicted, zero_division=0
        ),
        "recall_yes": recall_score(
            y_test, predicted, zero_division=0
        ),
        "f1_yes": f1_score(
            y_test, predicted, zero_division=0
        ),
        "roc_auc": roc_auc_score(
            y_test, predicted_probability
        ),
        "predict_seconds": predict_seconds,
        "true_negative": int(cm[0, 0]),
        "false_positive": int(cm[0, 1]),
        "false_negative": int(cm[1, 0]),
        "true_positive": int(cm[1, 1])
    })

evaluation = pd.DataFrame(evaluation_rows).sort_values(
    "f1_yes", ascending=False
)

print("\nTest-set model comparison:")
display(evaluation.round(4))

evaluation.to_csv(
    OUTPUT_DIR / "test_set_model_comparison.csv",
    index=False
)


# ============================================================
# 9. CONFUSION MATRICES
# ============================================================

for name, cm in confusion_matrices.items():
    print(f"\nConfusion matrix: {name}")
    print(cm)

    display_object = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No", "Yes"]
    )

    display_object.plot(
        values_format="d",
        colorbar=False
    )
    plt.title(f"Confusion Matrix - {name}")
    plt.tight_layout()

    safe_name = (
        name.lower()
        .replace(" ", "_")
        .replace("-", "_")
    )

    plt.savefig(
        OUTPUT_DIR / f"confusion_matrix_{safe_name}.png",
        dpi=150
    )
    plt.show()


# ============================================================
# 10. ROC CURVE COMPARISON
# ============================================================

fig, ax = plt.subplots(figsize=(8, 6))

for name in final_models:
    RocCurveDisplay.from_predictions(
        y_test,
        probabilities[name],
        name=name,
        ax=ax
    )

ax.set_title("ROC Curves - Held-Out Test Set")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "roc_curve_comparison.png", dpi=150)
plt.show()


# ============================================================
# 11. CLASSIFICATION REPORTS
# ============================================================

for name in final_models:
    print("\n" + "-" * 60)
    print("Classification report:", name)
    print("-" * 60)

    print(classification_report(
        y_test,
        predictions[name],
        labels=[0, 1],
        target_names=["No subscription", "Yes subscription"],
        zero_division=0
    ))


# ============================================================
# 12. SAVE THE MODEL SELECTED BY CROSS-VALIDATION
# ============================================================

# Choose based on training cross-validation F1, not test-set results.
# This avoids selecting the final model based on test performance.
selected_cv_name = ranked_names[0]
selected_model = best_estimators[selected_cv_name]

model_path = OUTPUT_DIR / "bank_marketing_selected_model.joblib"
joblib.dump(selected_model, model_path)

print("\nSelected model based on CV F1:", selected_cv_name)
print("Saved model to:", model_path)


# ============================================================
# 13. SAVE RUN SUMMARY
# ============================================================

summary = {
    "dataset": "UCI Bank Marketing - bank-full.csv",
    "rows_after_target_validation": int(len(X)),
    "number_of_features": int(X.shape[1]),
    "training_rows": int(len(X_train)),
    "testing_rows": int(len(X_test)),
    "positive_target_count": int(y.sum()),
    "negative_target_count": int((y == 0).sum()),
    "cross_validation_f1_selected_model":
        float(searches[selected_cv_name].best_score_),
    "selected_model": selected_cv_name,
    "excluded_duration": True,
    "random_state": RANDOM_STATE
}

pd.Series(summary, name="value").to_csv(
    OUTPUT_DIR / "run_summary.csv",
    header=True
)

print("\n" + "=" * 65)
print("PROJECT RUN COMPLETED")
print("=" * 65)
print("Output folder:", OUTPUT_DIR)
print("Generated files:")
for path in sorted(OUTPUT_DIR.iterdir()):
    print(" -", path.name)

print(
    "\nImportant: report the actual metrics generated by this run. "
    "Do not invent or assume model performance values."
)
