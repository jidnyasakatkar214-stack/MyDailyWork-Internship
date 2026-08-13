import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


try:
    df = pd.read_csv("IRIS.csv", encoding="utf-8")
except UnicodeDecodeError:
    df = pd.read_csv("IRIS.csv", encoding="latin1")

print("--- First 5 Rows of Dataset ---")
print(df.head())


feature_columns = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

X = df[feature_columns]
y = df["species"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train_scaled, y_train)


y_pred = model.predict(X_test_scaled)

print("\n--- Accuracy Score ---")
print(f"Model Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")

print("--- Classification Report ---")
print(classification_report(y_test, y_pred))


sample_data = pd.DataFrame(
    [[5.1, 3.5, 1.4, 0.2]],
    columns=feature_columns,
)
sample_scaled = scaler.transform(sample_data)
prediction = model.predict(sample_scaled)[0]

print("--- Sample Prediction ---")
print(f"Prediction for input [5.1, 3.5, 1.4, 0.2]: {prediction}")
