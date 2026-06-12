import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import time

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold
)

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve
)

# =====================================
# SETTINGS
# =====================================

TOP_FEATURES = 30
RANDOM_STATE = 42   # FIXED for reproducibility

# =====================================
# LOAD DATA
# =====================================

df = pd.read_csv("dataset.csv")

print("\nDataset shape:", df.shape)

X = df.drop("label", axis=1)
y = df["label"]

feature_names = X.columns

# =====================================
# TRAIN TEST SPLIT
# =====================================

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.5,
    stratify=y,
    random_state=RANDOM_STATE
)

print("\nTrain samples:", len(X_train_raw))
print("Test samples:", len(X_test_raw))

# =====================================
# 🔥 FIXED FEATURE SELECTION (CV BASED)
# =====================================

print("\nSelecting stable top features (CV-based)...")

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)

importance_matrix = np.zeros(X_train_raw.shape[1])

rf_selector = RandomForestClassifier(
    n_estimators=300,
    random_state=RANDOM_STATE
)

for train_idx, _ in cv.split(X_train_raw, y_train):

    X_fold = X_train_raw.iloc[train_idx]
    y_fold = y_train.iloc[train_idx]

    rf_selector.fit(X_fold, y_fold)

    importance_matrix += rf_selector.feature_importances_

importance_matrix /= cv.get_n_splits()

feature_importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importance_matrix
}).sort_values(by="importance", ascending=False)

feature_importance_df.to_csv("feature_importance.csv", index=False)

selected_features = feature_importance_df.head(TOP_FEATURES)["feature"].tolist()

pd.DataFrame({
    "selected_feature": selected_features
}).to_csv("selected_features.csv", index=False)

print(f"\nTop {TOP_FEATURES} Stable Features:\n")
for f in selected_features:
    print(f)

# =====================================
# REDUCE FEATURE SPACE
# =====================================

X_train = X_train_raw[selected_features]
X_test = X_test_raw[selected_features]

# =====================================
# SCALING
# =====================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =====================================
# MODELS
# =====================================

models = {
    "RandomForest": RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE
    ),

    "LogisticRegression": LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_STATE
    ),

    "SVM": SVC(
        kernel="linear",
        probability=True,
        random_state=RANDOM_STATE
    )
}

results = {}
overhead_results = {}

best_model = None
best_model_name = None
best_accuracy = 0

# =====================================
# TRAIN + EVALUATE
# =====================================

for name, model in models.items():

    print(f"\n================ {name} ================")

    train_start = time.time()

    model.fit(X_train_scaled, y_train)

    training_time = time.time() - train_start

    predict_start = time.time()

    y_pred = model.predict(X_test_scaled)

    prediction_time = time.time() - predict_start

    acc = accuracy_score(y_test, y_pred)

    print("Accuracy:", round(acc, 4))

    print(classification_report(
        y_test,
        y_pred,
        target_names=["benign", "malware"]
    ))

    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    cv_scores = cross_val_score(
        model,
        X_train_scaled,
        y_train,
        cv=cv
    )

    print("CV Mean Accuracy:", round(cv_scores.mean(), 4))
    print("Training Time:", round(training_time, 6))
    print("Prediction Time:", round(prediction_time, 6))

    overhead_results[name] = {
        "accuracy": acc,
        "cv_accuracy": cv_scores.mean(),
        "training_time_sec": training_time,
        "prediction_time_sec": prediction_time
    }

    results[name] = acc

    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model
        best_model_name = name

# =====================================
# SAVE OVERHEAD
# =====================================

pd.DataFrame(overhead_results).T.to_csv("overhead_results.csv")

# =====================================
# BEST MODEL
# =====================================

print("\n==============================")
print("BEST MODEL:", best_model_name)
print("==============================")

# =====================================
# ROC CURVE
# =====================================

y_prob = best_model.predict_proba(X_test_scaled)[:, 1]

fpr, tpr, _ = roc_curve(y_test, y_prob)

roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC Curve - {best_model_name}")
plt.legend()
plt.savefig("roc_curve.png", bbox_inches="tight")

# =====================================
# PRECISION RECALL
# =====================================

precision, recall, _ = precision_recall_curve(y_test, y_prob)

plt.figure()
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title(f"Precision-Recall Curve - {best_model_name}")
plt.savefig("precision_recall_curve.png", bbox_inches="tight")

# =====================================
# FEATURE PLOT
# =====================================

top15 = feature_importance_df.head(15)

plt.figure(figsize=(8, 6))
plt.barh(top15["feature"][::-1], top15["importance"][::-1])
plt.title("Top 15 Most Important Features")
plt.tight_layout()
plt.savefig("feature_importance.png", bbox_inches="tight")

# =====================================
# SAVE MODEL
# =====================================

joblib.dump(best_model, "malware_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nSaved files:")
print("- malware_model.pkl")
print("- scaler.pkl")
print("- feature_importance.csv")
print("- selected_features.csv")
print("- overhead_results.csv")
print("- roc_curve.png")
print("- precision_recall_curve.png")
print("- feature_importance.png")
