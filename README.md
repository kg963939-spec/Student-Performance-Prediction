
# Student Performance Prediction Using Machine Learning
# Author: Khushi Gupta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# 1. Create sample student dataset
np.random.seed(42)
n = 300

study_hours = np.round(np.random.uniform(1, 10, n), 1)
attendance = np.round(np.random.uniform(50, 100, n), 1)
assignment_score = np.round(np.random.uniform(35, 100, n), 1)
internal_marks = np.round(np.random.uniform(30, 100, n), 1)
previous_score = np.round(np.random.uniform(30, 100, n), 1)

score = (
    0.25 * study_hours * 10
    + 0.20 * attendance
    + 0.20 * assignment_score
    + 0.20 * internal_marks
    + 0.15 * previous_score
)

result = np.where(score >= 60, "Pass", "Need Improvement")

df = pd.DataFrame({
    "Study_Hours": study_hours,
    "Attendance": attendance,
    "Assignment_Score": assignment_score,
    "Internal_Marks": internal_marks,
    "Previous_Score": previous_score,
    "Result": result
})

print("First five records:")
print(df.head())

print("\nDataset shape:", df.shape)
print("\nMissing values:")
print(df.isnull().sum())

print("\nResult distribution:")
print(df["Result"].value_counts())


# 2. Visualize result distribution
plt.figure(figsize=(7, 4))
sns.countplot(data=df, x="Result")
plt.title("Student Result Distribution")
plt.xlabel("Result")
plt.ylabel("Number of Students")
plt.show()


# 3. Prepare data
X = df.drop("Result", axis=1)
y = df["Result"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# 4. Train Random Forest model
model = RandomForestClassifier(
    n_estimators=150,
    random_state=42
)

model.fit(X_train, y_train)


# 5. Make predictions
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", f"{accuracy:.2%}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# 6. Confusion matrix
cm = confusion_matrix(
    y_test,
    y_pred,
    labels=["Pass", "Need Improvement"]
)

plt.figure(figsize=(6, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=["Pass", "Need Improvement"],
    yticklabels=["Pass", "Need Improvement"]
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


# 7. Feature importance
importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nFeature Importance:")
print(importance)

plt.figure(figsize=(8, 4))
importance.plot(kind="bar")
plt.title("Feature Importance")
plt.ylabel("Importance")
plt.xlabel("Features")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()


# 8. Predict a new student's result
new_student = pd.DataFrame({
    "Study_Hours": [6.0],
    "Attendance": [85.0],
    "Assignment_Score": [78.0],
    "Internal_Marks": [75.0],
    "Previous_Score": [72.0]
})

prediction = model.predict(new_student)[0]
probability = model.predict_proba(new_student).max()

print("\nNew Student Prediction:")
print("Predicted Result:", prediction)
print(f"Prediction Confidence: {probability:.2%}")


# 9. Conclusion
print("\nConclusion:")
print(
    "The Random Forest model uses study hours, attendance, assignment score, "
    "internal marks and previous score to classify student performance."
)
