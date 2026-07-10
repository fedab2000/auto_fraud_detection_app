import os

import pandas as pd
import plotly.express as px
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from ai_investigator import generate_rule_based_report
#from model_explainer import get_shap_explanation


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Auto Insurance Fraud Detection",
    page_icon="🚗",
    layout="wide"
)


# =====================================================
# DATA LOADING
# =====================================================

@st.cache_data
def load_data():
    claims = pd.read_csv("synthetic_auto_insurance_claims.csv")
    feature_rankings = pd.read_csv("feature_ranking_comparison.csv")
    model_results = pd.read_csv("model_comparison_results.csv")

    return claims, feature_rankings, model_results


# =====================================================
# MODEL TRAINING
# =====================================================

@st.cache_resource
def train_model_for_app(claims):
    df_model = claims.drop(columns=["claim_id"])

    y = df_model["fraud_flag"]
    X = df_model.drop(columns=["fraud_flag"])

    categorical_features = (
        X.select_dtypes(include=["object"])
        .columns
        .tolist()
    )

    numeric_features = (
        X.select_dtypes(exclude=["object"])
        .columns
        .tolist()
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                numeric_features
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features
            )
        ]
    )

    negative_count = (y == 0).sum()
    positive_count = (y == 1).sum()

    scale_pos_weight = (
        negative_count / positive_count
        if positive_count > 0
        else 1.0
    )

    xgb_model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        scale_pos_weight=scale_pos_weight
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", xgb_model)
        ]
    )

    pipeline.fit(X, y)

    return pipeline


# =====================================================
# LOAD DATA AND MODEL
# =====================================================

claims, feature_rankings, model_results = load_data()
model = train_model_for_app(claims)


# =====================================================
# APPLICATION HEADER
# =====================================================

st.title("🚗 Auto Insurance Fraud Detection Dashboard")

st.write(
    "Synthetic auto insurance claims analytics, feature selection, "
    "model comparison, fraud risk scoring, confusion matrix evaluation, "
    "user-controlled fraud threshold, explainable AI, and structured "
    "claim investigation reports."
)


tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Executive Dashboard",
    "Fraud Analytics",
    "Feature Selection",
    "Model Performance",
    "Claim Risk Scoring Tool",
    "Confusion Matrices"
])


# =====================================================
# TAB 1: EXECUTIVE DASHBOARD
# =====================================================

with tab1:
    st.header("Executive Dashboard")

    total_claims = len(claims)
    fraud_cases = claims["fraud_flag"].sum()
    fraud_rate = claims["fraud_flag"].mean() * 100
    avg_claim_amount = claims["claim_amount"].mean()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Claims",
        f"{total_claims:,}"
    )

    col2.metric(
        "Fraud Cases",
        f"{fraud_cases:,}"
    )

    col3.metric(
        "Fraud Rate",
        f"{fraud_rate:.2f}%"
    )

    col4.metric(
        "Average Claim Amount",
        f"${avg_claim_amount:,.0f}"
    )

    st.subheader("Fraud vs Non-Fraud Claims")

    fraud_counts = (
        claims["fraud_flag"]
        .map({
            0: "Non-Fraud",
            1: "Fraud"
        })
        .value_counts()
        .reset_index()
    )

    fraud_counts.columns = [
        "Claim Type",
        "Count"
    ]

    fig = px.pie(
        fraud_counts,
        names="Claim Type",
        values="Count",
        title="Fraud vs Non-Fraud Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Claim Amount Distribution")

    claims_copy = claims.copy()

    claims_copy["Fraud Label"] = (
        claims_copy["fraud_flag"]
        .map({
            0: "Non-Fraud",
            1: "Fraud"
        })
    )

    fig = px.histogram(
        claims_copy,
        x="claim_amount",
        color="Fraud Label",
        nbins=40,
        title="Claim Amount Distribution by Fraud Flag"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# =====================================================
# TAB 2: FRAUD ANALYTICS
# =====================================================

with tab2:
    st.header("Fraud Analytics")

    st.subheader("Fraud Rate by Province")

    province_fraud = (
        claims
        .groupby("province")["fraud_flag"]
        .mean()
        .reset_index()
    )

    province_fraud["fraud_rate_percent"] = (
        province_fraud["fraud_flag"] * 100
    )

    fig = px.bar(
        province_fraud.sort_values(
            "fraud_rate_percent",
            ascending=False
        ),
        x="province",
        y="fraud_rate_percent",
        title="Fraud Rate by Province",
        labels={
            "fraud_rate_percent": "Fraud Rate (%)",
            "province": "Province"
        }
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Fraud Rate by Claim Type")

    claim_type_fraud = (
        claims
        .groupby("claim_type")["fraud_flag"]
        .mean()
        .reset_index()
    )

    claim_type_fraud["fraud_rate_percent"] = (
        claim_type_fraud["fraud_flag"] * 100
    )

    fig = px.bar(
        claim_type_fraud.sort_values(
            "fraud_rate_percent",
            ascending=False
        ),
        x="claim_type",
        y="fraud_rate_percent",
        title="Fraud Rate by Claim Type",
        labels={
            "fraud_rate_percent": "Fraud Rate (%)",
            "claim_type": "Claim Type"
        }
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Fraud Rate by Vehicle Brand")

    brand_fraud = (
        claims
        .groupby("vehicle_brand")["fraud_flag"]
        .mean()
        .reset_index()
    )

    brand_fraud["fraud_rate_percent"] = (
        brand_fraud["fraud_flag"] * 100
    )

    fig = px.bar(
        brand_fraud.sort_values(
            "fraud_rate_percent",
            ascending=False
        ),
        x="vehicle_brand",
        y="fraud_rate_percent",
        title="Fraud Rate by Vehicle Brand",
        labels={
            "fraud_rate_percent": "Fraud Rate (%)",
            "vehicle_brand": "Vehicle Brand"
        }
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Fraud Rate by Gender")

    gender_fraud = (
        claims
        .groupby("gender")["fraud_flag"]
        .mean()
        .reset_index()
    )

    gender_fraud["fraud_rate_percent"] = (
        gender_fraud["fraud_flag"] * 100
    )

    fig = px.bar(
        gender_fraud,
        x="gender",
        y="fraud_rate_percent",
        title="Fraud Rate by Gender",
        labels={
            "fraud_rate_percent": "Fraud Rate (%)",
            "gender": "Gender"
        }
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# =====================================================
# TAB 3: FEATURE SELECTION
# =====================================================

with tab3:
    st.header("Feature Selection")

    st.write(
        "This section compares feature rankings from "
        "LASSO, Ridge, and Elastic Net."
    )

    ranking_columns = [
        "Feature",
        "LASSO_Rank",
        "Ridge_Rank",
        "ElasticNet_Rank",
        "Average_Rank",
        "Rank_Difference"
    ]

    st.dataframe(
        feature_rankings[
            ranking_columns
        ].head(30),
        use_container_width=True
    )

    st.subheader("Top Features by Average Rank")

    top_features = (
        feature_rankings
        .sort_values("Average_Rank")
        .head(15)
    )

    fig = px.bar(
        top_features.sort_values(
            "Average_Rank",
            ascending=True
        ),
        x="Average_Rank",
        y="Feature",
        orientation="h",
        title="Top 15 Features by Average Rank"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.info(
        "Lower Average Rank means the feature was consistently "
        "ranked as more important across LASSO, Ridge, and "
        "Elastic Net."
    )


# =====================================================
# TAB 4: MODEL PERFORMANCE
# =====================================================

with tab4:
    st.header("Model Performance")

    st.subheader("Model Comparison Table")

    st.dataframe(
        model_results,
        use_container_width=True
    )

    st.subheader("ROC-AUC by Model")

    fig = px.bar(
        model_results.sort_values(
            "ROC-AUC",
            ascending=False
        ),
        x="Model",
        y="ROC-AUC",
        title="Model Comparison by ROC-AUC",
        text="ROC-AUC"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Precision, Recall, and F1 Score")

    metric_df = model_results.melt(
        id_vars="Model",
        value_vars=[
            "Precision",
            "Recall",
            "F1 Score"
        ],
        var_name="Metric",
        value_name="Score"
    )

    fig = px.bar(
        metric_df,
        x="Model",
        y="Score",
        color="Metric",
        barmode="group",
        title="Precision, Recall, and F1 Score by Model"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# =====================================================
# TAB 5: CLAIM RISK SCORING TOOL
# =====================================================

with tab5:
    st.header("Claim Risk Scoring Tool")

    st.write(
        "Enter claim details below to estimate fraud probability "
        "and explain the model prediction."
    )

    threshold = st.slider(
        "Fraud Classification Threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.50,
        step=0.05
    )

    st.caption(
        "Lower threshold = more claims flagged as fraud. "
        "Higher threshold = fewer claims flagged as fraud."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider(
            "Age",
            min_value=18,
            max_value=85,
            value=40
        )

        gender = st.selectbox(
            "Gender",
            [
                "Male",
                "Female",
                "Other"
            ]
        )

        province = st.selectbox(
            "Province",
            sorted(
                claims["province"].unique()
            )
        )

        postal_code = st.text_input(
            "Postal Code Prefix",
            "M5A"
        )

        years_licensed = st.slider(
            "Years Licensed",
            min_value=0,
            max_value=65,
            value=15
        )

        employment_status = st.selectbox(
            "Employment Status",
            [
                "Employed",
                "Unemployed"
            ]
        )

    with col2:
        vehicle_brand = st.selectbox(
            "Vehicle Brand",
            sorted(
                claims["vehicle_brand"].unique()
            )
        )

        vehicle_age = st.slider(
            "Vehicle Age",
            min_value=0,
            max_value=25,
            value=6
        )

        vehicle_use = st.selectbox(
            "Vehicle Use",
            [
                "Personal",
                "Business"
            ]
        )

        vehicle_value = st.number_input(
            "Vehicle Value",
            min_value=3000,
            max_value=100000,
            value=26000,
            step=1000
        )

        claim_type = st.selectbox(
            "Claim Type",
            sorted(
                claims["claim_type"].unique()
            )
        )

    with col3:
        claim_amount = st.number_input(
            "Claim Amount",
            min_value=300,
            max_value=120000,
            value=8000,
            step=500
        )

        previous_claims = st.slider(
            "Previous Claims",
            min_value=0,
            max_value=10,
            value=1
        )

        police_report_filed = st.selectbox(
            "Police Report Filed",
            [
                "Yes",
                "No"
            ]
        )

        witness_present = st.selectbox(
            "Witness Present",
            [
                "Yes",
                "No"
            ]
        )

        policy_tenure_months = st.slider(
            "Policy Tenure Months",
            min_value=1,
            max_value=240,
            value=36
        )

    claim_to_vehicle_value_ratio = (
        claim_amount / vehicle_value
    )

    input_data = pd.DataFrame([{
        "age": age,
        "gender": gender,
        "province": province,
        "postal_code": postal_code,
        "years_licensed": years_licensed,
        "employment_status": employment_status,
        "vehicle_brand": vehicle_brand,
        "vehicle_age": vehicle_age,
        "vehicle_use": vehicle_use,
        "vehicle_value": vehicle_value,
        "claim_type": claim_type,
        "claim_amount": claim_amount,
        "previous_claims": previous_claims,
        "police_report_filed": police_report_filed,
        "witness_present": witness_present,
        "policy_tenure_months": policy_tenure_months,
        "claim_to_vehicle_value_ratio":
            claim_to_vehicle_value_ratio
    }])

    st.subheader(
        "Calculated Claim-to-Vehicle-Value Ratio"
    )

    st.write(
        f"{claim_to_vehicle_value_ratio:.2f}"
    )

    if st.button(
        "Score Claim",
        type="primary"
    ):
        fraud_probability = (
            model.predict_proba(
                input_data
            )[0][1]
        )

        prediction = (
            1
            if fraud_probability >= threshold
            else 0
        )

        if fraud_probability < 0.30:
            risk_level = "Low Risk"

        elif fraud_probability < 0.60:
            risk_level = "Medium Risk"

        else:
            risk_level = "High Risk"

        st.subheader("Fraud Risk Result")

        result_col1, result_col2, result_col3, result_col4 = (
            st.columns(4)
        )

        result_col1.metric(
            "Fraud Probability",
            f"{fraud_probability * 100:.2f}%"
        )

        result_col2.metric(
            "Threshold",
            f"{threshold:.2f}"
        )

        result_col3.metric(
            "Predicted Class",
            (
                "Fraud"
                if prediction == 1
                else "Non-Fraud"
            )
        )

        result_col4.metric(
            "Risk Level",
            risk_level
        )

        st.warning(
            "This is a synthetic demonstration model. "
            "It should not be used for real claim decisions "
            "without validation, governance, explainability, "
            "fairness testing, and human review."
        )

        # =================================================
        # SHAP MODEL EXPLANATION
        # =================================================

       
        # =================================================
        # RULE-BASED INVESTIGATION REPORT
        # =================================================

        claim_data = (
            input_data
            .iloc[0]
            .to_dict()
        )

        claim_data["claim_id"] = (
            "Manual Entry"
        )

        report = generate_rule_based_report(
            claim_data=claim_data,
            fraud_probability=fraud_probability,
            risk_level=risk_level
        )

        st.divider()

        st.subheader(
            "AI Claim Investigation Report"
        )

        st.write(
            "**Investigation Summary**"
        )

        st.write(
            report.investigation_summary
        )

        st.write(
            "**Risk Factor Assessment**"
        )

        risk_factor_df = pd.DataFrame([
            {
                "Risk Factor":
                    risk_factor.factor,

                "Severity":
                    risk_factor.severity,

                "Explanation":
                    risk_factor.explanation
            }
            for risk_factor
            in report.risk_factors
        ])

        st.dataframe(
            risk_factor_df,
            use_container_width=True,
            hide_index=True
        )

        st.write(
            "**Recommended Action**"
        )

        if "SIU" in report.recommended_action:
            st.error(
                report.recommended_action
            )

        elif "Review" in report.recommended_action:
            st.warning(
                report.recommended_action
            )

        else:
            st.success(
                report.recommended_action
            )

        if shap_df is not None:
            st.info(
                "The Model Explanation section reflects the "
                "actual XGBoost prediction. The current "
                "investigation report still uses separate "
                "business rules. The next enhancement will "
                "connect the SHAP evidence directly to the "
                "investigation report."
            )


# =====================================================
# TAB 6: CONFUSION MATRICES
# =====================================================

with tab6:
    st.header("Confusion Matrices")

    st.markdown("""
    A confusion matrix shows how each model classified claims:

    - **True Positives:** Fraud claims correctly detected
    - **False Positives:** Legitimate claims incorrectly flagged as fraud
    - **False Negatives:** Fraud claims missed by the model
    - **True Negatives:** Legitimate claims correctly classified

    In fraud detection, **False Negatives** are especially important
    because they represent fraud cases that were missed.
    """)

    selected_model = st.selectbox(
        "Select Model",
        [
            "Logistic Regression",
            "Random Forest",
            "Gradient Boosting",
            "XGBoost"
        ]
    )

    image_lookup = {
        "Logistic Regression":
            "logistic_regression_confusion_matrix.png",

        "Random Forest":
            "random_forest_confusion_matrix.png",

        "Gradient Boosting":
            "gradient_boosting_confusion_matrix.png",

        "XGBoost":
            "xgboost_confusion_matrix.png"
    }

    image_file = image_lookup[
        selected_model
    ]

    if os.path.exists(image_file):
        st.image(
            image_file,
            caption=(
                f"{selected_model} Confusion Matrix"
            ),
            use_container_width=True
        )

    else:
        st.error(
            f"Confusion matrix image not found: "
            f"{image_file}. Please run train_models.py "
            "again and upload the PNG files to GitHub."
        )

    st.subheader(
        "Model Performance Summary"
    )

    summary_columns = [
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC"
    ]

    optional_columns = [
        "True Negatives",
        "False Positives",
        "False Negatives",
        "True Positives"
    ]

    available_columns = (
        summary_columns
        + [
            column
            for column in optional_columns
            if column in model_results.columns
        ]
    )

    st.dataframe(
        model_results[
            available_columns
        ],
        use_container_width=True,
        hide_index=True
    )

    st.subheader(
        "Business Interpretation"
    )

    if selected_model == "Logistic Regression":
        st.info(
            "Logistic Regression is a simple and explainable "
            "baseline model. Use it to compare more complex "
            "models against a transparent benchmark."
        )

    elif selected_model == "Random Forest":
        st.success(
            "Random Forest performs well on structured insurance "
            "data and provides useful feature importance."
        )

    elif selected_model == "Gradient Boosting":
        st.warning(
            "Gradient Boosting can detect complex fraud patterns. "
            "Compare its False Positives and False Negatives "
            "against XGBoost."
        )

    elif selected_model == "XGBoost":
        st.success(
            "XGBoost is often highly effective for structured "
            "fraud detection problems. It is usually a strong "
            "candidate when optimizing ROC-AUC and recall."
        )