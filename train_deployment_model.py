"""
=====================================================================
 TRAIN THE DEPLOYMENT MODEL
=====================================================================
This script trains the model that the Streamlit app actually uses.

IMPORTANT - why this is a SEPARATE model from modeling_pipeline.py:

The research pipeline (modeling_pipeline.py) used every available
column, including `Status` (did the patient die / get a transplant)
and `N_Days` (how long the patient was followed until that outcome).
Those are fine for a retrospective study, but they are NOT available
for a NEW patient walking into a clinic today - you cannot know a
patient's survival outcome before predicting their disease stage.

Using them in a deployed app would be textbook DATA LEAKAGE: the app
would look accurate on paper while being unusable in reality.

So this script deliberately DROPS `Status` and `N_Days`, and trains
only on information a clinician genuinely has at diagnosis time:
demographics, physical signs, and lab results.

Run this ONCE before launching the app:
    python train_deployment_model.py
=====================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.dummy import DummyClassifier
from xgboost import XGBClassifier


# ---------------------------------------------------------------------
# 1) Load + clean (same logic as the research pipeline)
# ---------------------------------------------------------------------
df = pd.read_csv("cirrhosis.csv")
df = df.dropna(subset=["Stage"])          # Stage is the target: can't impute a label
df = df.drop(columns=["ID"])              # row counter, no medical meaning

# ---------------------------------------------------------------------
# 2) DROP THE LEAKY / UNAVAILABLE-AT-DIAGNOSIS COLUMNS
# ---------------------------------------------------------------------
# This is the single most important difference vs. the research script.
df = df.drop(columns=["Status", "N_Days"])

categorical_col = df.select_dtypes(include="object").columns
numerical_col = df.select_dtypes(exclude="object").columns

for col in numerical_col:
    df[col] = df[col].fillna(df[col].mean())
for col in categorical_col:
    df[col] = df[col].fillna(df[col].mode()[0])

# ---------------------------------------------------------------------
# 3) Feature engineering (identical to the research pipeline)
# ---------------------------------------------------------------------
df["Age_Years"] = (df["Age"] / 365.25).astype(int)

bili_threshold = df["Bilirubin"].quantile(0.75)
df["High_Severity"] = ((df["Bilirubin"] > bili_threshold) & (df["Ascites"] == "Y")).astype(int)

signs = ["Ascites", "Hepatomegaly", "Spiders", "Edema"]
df["Signs_Count"] = df[signs].apply(lambda row: (row == "Y").sum(), axis=1)

df["Low_Albumin"] = (df["Albumin"] < 2.8).astype(int)

# ---------------------------------------------------------------------
# 4) Encoding (identical strategy to the research pipeline)
# ---------------------------------------------------------------------
binary_map = {"Y": 1, "N": 0}
for col in ["Ascites", "Hepatomegaly", "Spiders"]:
    df[col] = df[col].map(binary_map)

df["Sex"] = df["Sex"].map({"M": 1, "F": 0})
df["Edema"] = df["Edema"].map({"N": 0, "S": 1, "Y": 2})   # ordinal: real severity order
df = pd.get_dummies(df, columns=["Drug"], drop_first=True)  # nominal: no natural order

# ---------------------------------------------------------------------
# 5) Split, scale, train
# ---------------------------------------------------------------------
X = df.drop(columns=["Stage"])
y = df["Stage"].astype(int) - 1     # XGBoost needs labels starting at 0

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

numeric_features = [
    "Age", "Age_Years", "Bilirubin", "Cholesterol", "Albumin",
    "Copper", "Alk_Phos", "SGOT", "Tryglicerides", "Platelets", "Prothrombin",
]

scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[numeric_features] = scaler.fit_transform(X_train[numeric_features])
X_test_scaled[numeric_features] = scaler.transform(X_test[numeric_features])

# Tune XGBoost (the winner from the research comparison) on this reduced feature set
param_grid = {
    "n_estimators": [100, 200, 300],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [3, 4, 5],
}
grid = GridSearchCV(
    XGBClassifier(random_state=42, eval_metric="mlogloss"),
    param_grid, cv=5, scoring="f1_macro", n_jobs=-1,
)
grid.fit(X_train_scaled, y_train)
model = grid.best_estimator_

# ---------------------------------------------------------------------
# 6) Evaluate honestly and save everything the app needs
# ---------------------------------------------------------------------
preds = model.predict(X_test_scaled)

dummy = DummyClassifier(strategy="most_frequent", random_state=42).fit(X_train_scaled, y_train)
baseline_acc = accuracy_score(y_test, dummy.predict(X_test_scaled))

metrics = {
    "accuracy": float(accuracy_score(y_test, preds)),
    "f1_macro": float(f1_score(y_test, preds, average="macro")),
    "cv_f1_macro": float(grid.best_score_),
    "baseline_accuracy": float(baseline_acc),
    "best_params": grid.best_params_,
    "n_train": int(len(X_train)),
    "n_test": int(len(X_test)),
}

print("Best params:", grid.best_params_)
print(f"Test accuracy      : {metrics['accuracy']:.3f}")
print(f"Test F1-macro      : {metrics['f1_macro']:.3f}")
print(f"Baseline accuracy  : {metrics['baseline_accuracy']:.3f}")
print("\n", classification_report(y_test, preds, zero_division=0))

joblib.dump(model, "cirrhosis_model.pkl")
joblib.dump(scaler, "scaler.pkl")

# The app must build its input row with EXACTLY these columns, in this
# exact order - otherwise the model silently receives shuffled features.
joblib.dump(list(X.columns), "feature_columns.pkl")
joblib.dump(numeric_features, "numeric_features.pkl")

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("\nSaved: cirrhosis_model.pkl, scaler.pkl, feature_columns.pkl, numeric_features.pkl, metrics.json")
