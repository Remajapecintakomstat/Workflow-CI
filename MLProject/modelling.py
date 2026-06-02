import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# =========================
# MLflow Configuration
# =========================

mlflow.set_tracking_uri("http://127.0.0.1:5000")

mlflow.set_experiment(
    "Telco Customer Churn"
)

mlflow.sklearn.autolog()

# =========================
# Load Dataset
# =========================

df = pd.read_csv(
    r"C:\Users\hp\Downloads\Telco Customer Churn_preprocessing.csv"
)

print("Dataset Shape:", df.shape)

# =========================
# Features & Target
# =========================

X = df.drop(
    "Churn",
    axis=1
)

y = df["Churn"]

# =========================
# Train Test Split
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nFinal Dataset")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)

print("\nTarget Distribution")
print(y_train.value_counts())

# =========================
# Training
# =========================

with mlflow.start_run():

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    # Prediction
    y_pred = model.predict(
        X_test
    )

    # Metrics
    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred
    )

    recall = recall_score(
        y_test,
        y_pred
    )

    f1 = f1_score(
        y_test,
        y_pred
    )

    print("\nModel Performance")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

print("\nTraining selesai.")
print("Cek MLflow UI di:")
print("http://127.0.0.1:5000")
