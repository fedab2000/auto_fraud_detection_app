import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# Load dataset
df = pd.read_csv("synthetic_auto_insurance_claims.csv")

# Drop ID column
df = df.drop(columns=["claim_id"])

# Target and features
y = df["fraud_flag"]
X = df.drop(columns=["fraud_flag"])

# Identify feature types
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
numeric_features = X.select_dtypes(exclude=["object"]).columns.tolist()

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=5000),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced"
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        random_state=42
    )
}

results = []
trained_pipelines = {}

for model_name, model in models.items():

    print(f"\nTraining {model_name}...")

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba)

    cm = confusion_matrix(y_test, y_pred)

    results.append({
        "Model": model_name,
        "Accuracy": round(accuracy, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1 Score": round(f1, 4),
        "ROC-AUC": round(roc_auc, 4),
        "True Negatives": cm[0][0],
        "False Positives": cm[0][1],
        "False Negatives": cm[1][0],
        "True Positives": cm[1][1]
    })

    trained_pipelines[model_name] = pipeline

    print(classification_report(y_test, y_pred, zero_division=0))
    print("Confusion Matrix:")
    print(cm)

    # Save confusion matrix image
    plt.figure(figsize=(6, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Non-Fraud", "Fraud"],
        yticklabels=["Non-Fraud", "Fraud"]
    )

    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    filename = model_name.lower().replace(" ", "_") + "_confusion_matrix.png"

    plt.savefig(filename, bbox_inches="tight")
    plt.close()

# Results table
results_df = pd.DataFrame(results)

print("\nMODEL COMPARISON RESULTS")
print(results_df)

# Save results
results_df.to_csv("model_comparison_results.csv", index=False)

with open("model_comparison_results.txt", "w") as f:
    f.write("AUTO INSURANCE FRAUD DETECTION - MODEL COMPARISON\n")
    f.write("=" * 65 + "\n\n")
    f.write(results_df.to_string(index=False))
    f.write("\n\nInterpretation:\n")
    f.write("- Accuracy shows overall correct predictions.\n")
    f.write("- Precision shows how many predicted fraud cases were actually fraud.\n")
    f.write("- Recall shows how many actual fraud cases the model found.\n")
    f.write("- F1 Score balances precision and recall.\n")
    f.write("- ROC-AUC measures the model's ability to separate fraud from non-fraud.\n")
    f.write("- False Negatives are especially important because they represent fraud cases missed by the model.\n")

# Select best model by ROC-AUC
best_model_name = results_df.sort_values(
    by="ROC-AUC",
    ascending=False
).iloc[0]["Model"]

best_model = trained_pipelines[best_model_name]

# Save best model
with open("trained_fraud_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

print("\nFiles created:")
print("- model_comparison_results.csv")
print("- model_comparison_results.txt")
print("- trained_fraud_model.pkl")
print("- logistic_regression_confusion_matrix.png")
print("- random_forest_confusion_matrix.png")
print("- gradient_boosting_confusion_matrix.png")

print(f"\nBest model saved: {best_model_name}")