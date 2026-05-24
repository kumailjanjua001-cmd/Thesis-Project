import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import time

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
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

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("dataset.csv")

print("\nDataset shape:", df.shape)

X = df.drop("label", axis=1)
y = df["label"]

# =========================
# RANDOM SPLIT (CHANGES EVERY RUN)
# =========================
random_seed = int(time.time()) % 100000  # dynamic randomness

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.5,
    random_state=random_seed,
    stratify=y
)

print("\nTrain samples:", len(X_train))
print("Test samples:", len(X_test))

# =========================
# SCALING
# =========================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# MODELS
# =========================
models = {
    "RandomForest": RandomForestClassifier(
        n_estimators=200,
        random_state=random_seed,
        class_weight="balanced"
    ),
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "SVM": SVC(kernel="linear", probability=True, random_state=random_seed)
}

results = {}

# =========================
# TRAIN + EVALUATE
# =========================
for name, model in models.items():
    print(f"\n================ {name} ================")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    results[name] = acc

    print("Accuracy:", round(acc, 4))
    print(classification_report(y_test, y_pred, target_names=["benign", "malware"]))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Cross-validation (MOST IMPORTANT METRIC)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv)

    print("CV Mean Accuracy:", round(cv_scores.mean(), 4))

# =========================
# BEST MODEL
# =========================
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]

print("\nBEST MODEL:", best_model_name)

# =========================
# ROC CURVE
# =========================
y_prob = best_model.predict_proba(X_test)[:, 1]

fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC Curve - {best_model_name}")
plt.legend()
plt.show()

# =========================
# PRECISION-RECALL CURVE
# =========================
precision, recall, _ = precision_recall_curve(y_test, y_prob)

plt.figure()
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title(f"Precision-Recall Curve - {best_model_name}")
plt.show()

# =========================
# FEATURE IMPORTANCE (RandomForest only)
# =========================
if best_model_name == "RandomForest":
    importances = best_model.feature_importances_
    features = X.columns

    top_idx = np.argsort(importances)[-15:]

    plt.figure()
    plt.barh(range(len(top_idx)), importances[top_idx])
    plt.yticks(range(len(top_idx)), features[top_idx])
    plt.title("Top 15 Important Syscall Features")
    plt.show()

# =========================
# SAVE MODEL
# =========================
joblib.dump(best_model, "malware_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\n[+] Model and scaler saved successfully")
