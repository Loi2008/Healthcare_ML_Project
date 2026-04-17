import os
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


def train_model(df_cleaned, save_dir="Models"):
    print("\n Starting Model Training...")

    os.makedirs(save_dir, exist_ok=True)

    X = df_cleaned.drop("test_results", axis=1)
    y = df_cleaned["test_results"]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    categorical_cols = X.select_dtypes(include=["object", "string", "str"]).columns
    numeric_cols = X.select_dtypes(exclude=["object", "string", "str"]).columns

    print("Categorical columns:", list(categorical_cols))
    print("Numeric columns:", list(numeric_cols))

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", StandardScaler(), numeric_cols)
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=3000),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            objective="multi:softprob",
            num_class=len(label_encoder.classes_),
            eval_metric="mlogloss",
            random_state=42
        ),
        "LightGBM": LGBMClassifier(
            objective="multiclass",
            random_state=42
        ),
        "CatBoost": CatBoostClassifier(
            loss_function="MultiClass",
            verbose=0,
            random_state=42
        )
    }

    results = {}
    trained_pipelines = {}

    for name, model in models.items():
        print(f"\n{'=' * 60}")
        print(f" Training {name}...")
        print(f"{'=' * 60}")

        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", model)
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        y_test_labels = label_encoder.inverse_transform(y_test)
        y_pred_labels = label_encoder.inverse_transform(y_pred)

        accuracy = accuracy_score(y_test_labels, y_pred_labels)
        precision = precision_score(y_test_labels, y_pred_labels, average="weighted", zero_division=0)
        recall = recall_score(y_test_labels, y_pred_labels, average="weighted", zero_division=0)
        f1 = f1_score(y_test_labels, y_pred_labels, average="weighted", zero_division=0)
        cm = confusion_matrix(y_test_labels, y_pred_labels)

        results[name] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
        trained_pipelines[name] = pipeline

        print(f"\n {name} Performance:")
        print(f"Accuracy:   {accuracy:.4f}")
        print(f"Precision:  {precision:.4f}")
        print(f"Recall:     {recall:.4f}")
        print(f"F1-score:   {f1:.4f}")

        print("\nClassification Report:")
        print(classification_report(y_test_labels, y_pred_labels, zero_division=0))

        print("Confusion Matrix:")
        print(cm)

    print(f"\n{'=' * 60}")
    print(" MODEL COMPARISON SUMMARY")
    print(f"{'=' * 60}")
    for model_name, metrics in results.items():
        print(
            f"{model_name}: "
            f"Accuracy={metrics['accuracy']:.4f}, "
            f"Precision={metrics['precision']:.4f}, "
            f"Recall={metrics['recall']:.4f}, "
            f"F1-score={metrics['f1_score']:.4f}"
        )

    best_model_name = max(results, key=lambda x: results[x]["f1_score"])
    best_model = trained_pipelines[best_model_name]

    print(f"\n Best Model: {best_model_name}")
    print(f"Best Accuracy:  {results[best_model_name]['accuracy']:.4f}")
    print(f"Best Precision: {results[best_model_name]['precision']:.4f}")
    print(f"Best Recall:    {results[best_model_name]['recall']:.4f}")
    print(f"Best F1-score:  {results[best_model_name]['f1_score']:.4f}")

    model_path = os.path.join(save_dir, "best_healthcare_model.pkl")
    encoder_path = os.path.join(save_dir, "label_encoder.pkl")

    joblib.dump(best_model, model_path)
    joblib.dump(label_encoder, encoder_path)

    print(f"\n Best model saved as {model_path}")
    print(f" Label encoder saved as {encoder_path}")

    return best_model, label_encoder