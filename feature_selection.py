import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, RidgeClassifier

# Load Data
df = pd.read_csv("synthetic_auto_insurance_claims.csv")

# Remove ID column
df = df.drop(columns=["claim_id"])

# Target and Features
y = df["fraud_flag"]
X = df.drop(columns=["fraud_flag"])

# Identify column types
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
numeric_features = X.select_dtypes(exclude=["object"]).columns.tolist()

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

X_processed = preprocessor.fit_transform(X)

cat_names = preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_features)
feature_names = numeric_features + list(cat_names)

X_processed_df = pd.DataFrame(
    X_processed.toarray() if hasattr(X_processed, "toarray") else X_processed,
    columns=feature_names
)

def get_ranked_features(model, model_name):
    model.fit(X_processed_df, y)

    results = pd.DataFrame({
        "Feature": feature_names,
        f"{model_name}_Coefficient": model.coef_[0]
    })

    results[f"{model_name}_AbsCoeff"] = results[f"{model_name}_Coefficient"].abs()

    results = results.sort_values(
        by=f"{model_name}_AbsCoeff",
        ascending=False
    ).reset_index(drop=True)

    results[f"{model_name}_Rank"] = results.index + 1

    return results[[
        "Feature",
        f"{model_name}_Coefficient",
        f"{model_name}_AbsCoeff",
        f"{model_name}_Rank"
    ]]

# LASSO
lasso = LogisticRegression(
    penalty="l1",
    solver="liblinear",
    max_iter=5000
)

lasso_results = get_ranked_features(lasso, "LASSO")

# RIDGE
ridge = RidgeClassifier(alpha=1.0)

ridge_results = get_ranked_features(ridge, "Ridge")

# ELASTIC NET
elastic = LogisticRegression(
    penalty="elasticnet",
    solver="saga",
    l1_ratio=0.5,
    max_iter=5000
)

elastic_results = get_ranked_features(elastic, "ElasticNet")

# Merge rankings
ranking_comparison = lasso_results.merge(
    ridge_results,
    on="Feature",
    how="outer"
).merge(
    elastic_results,
    on="Feature",
    how="outer"
)

# Average rank
ranking_comparison["Average_Rank"] = ranking_comparison[
    ["LASSO_Rank", "Ridge_Rank", "ElasticNet_Rank"]
].mean(axis=1)

# Rank difference range
ranking_comparison["Rank_Difference"] = ranking_comparison[
    ["LASSO_Rank", "Ridge_Rank", "ElasticNet_Rank"]
].max(axis=1) - ranking_comparison[
    ["LASSO_Rank", "Ridge_Rank", "ElasticNet_Rank"]
].min(axis=1)

# Sort table
ranking_comparison = ranking_comparison.sort_values(
    by="Average_Rank",
    ascending=True
)

# Save detailed results
lasso_results.to_csv("lasso_results.csv", index=False)
ridge_results.to_csv("ridge_results.csv", index=False)
elastic_results.to_csv("elastic_results.csv", index=False)

# Save comparison as CSV
ranking_comparison.to_csv("feature_ranking_comparison.csv", index=False)

# Save comparison as TXT
with open("feature_ranking_comparison.txt", "w") as f:
    f.write("FEATURE RANKING COMPARISON: LASSO vs RIDGE vs ELASTIC NET\n")
    f.write("=" * 75 + "\n\n")

    f.write("Top Features Ranked by Average Rank\n\n")

    f.write(
        ranking_comparison[
            [
                "Feature",
                "LASSO_Rank",
                "Ridge_Rank",
                "ElasticNet_Rank",
                "Average_Rank",
                "Rank_Difference"
            ]
        ].head(30).to_string(index=False)
    )

    f.write("\n\n")
    f.write("Interpretation:\n")
    f.write("- Average_Rank shows the overall importance of the feature across all three methods.\n")
    f.write("- Rank_Difference shows how much the feature ranking changes between methods.\n")
    f.write("- A low Average_Rank and low Rank_Difference means the feature is consistently important.\n")

print("Feature ranking comparison created successfully.")
print("Files saved:")
print("- feature_ranking_comparison.txt")
print("- feature_ranking_comparison.csv")
print("- lasso_results.csv")
print("- ridge_results.csv")
print("- elastic_results.csv")

print("\nTop 20 Features by Average Rank:")
print(
    ranking_comparison[
        [
            "Feature",
            "LASSO_Rank",
            "Ridge_Rank",
            "ElasticNet_Rank",
            "Average_Rank",
            "Rank_Difference"
        ]
    ].head(20)
)