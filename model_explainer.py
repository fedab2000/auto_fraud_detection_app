from typing import Any

import numpy as np
import pandas as pd
import shap


def clean_feature_name(feature_name: str) -> str:
    """
    Convert transformed pipeline feature names into readable labels.

    Examples:
        num__claim_amount -> Claim Amount
        cat__police_report_filed_No -> Police Report Filed: No
    """
    name = feature_name

    if name.startswith("num__"):
        name = name.replace("num__", "", 1)

    elif name.startswith("cat__"):
        name = name.replace("cat__", "", 1)

    name = name.replace("_", " ")

    return name.title()


def get_original_value(
    technical_feature: str,
    input_data: pd.DataFrame
):
    """
    Return the original business value for a transformed feature.
    """

    if technical_feature.startswith("num__"):
        original_column = technical_feature.replace("num__", "", 1)
        return input_data.iloc[0][original_column]

    if technical_feature.startswith("cat__"):
        encoded_name = technical_feature.replace("cat__", "", 1)

        matching_columns = [
            column
            for column in input_data.columns
            if encoded_name.startswith(f"{column}_")
        ]

        if matching_columns:
            original_column = max(
                matching_columns,
                key=len
            )
            return input_data.iloc[0][original_column]

    return None


def format_original_value(
    technical_feature: str,
    original_value
) -> str:
    """
    Format original claim values for business users.
    """

    if original_value is None:
        return "Not available"

    feature_lower = technical_feature.lower()

    if any(
        term in feature_lower
        for term in [
            "claim_amount",
            "vehicle_value"
        ]
    ):
        return f"${float(original_value):,.0f}"

    if "claim_to_vehicle_value_ratio" in feature_lower:
        return f"{float(original_value):.2f}"

    if "policy_tenure_months" in feature_lower:
        return f"{int(original_value)} months"

    if "previous_claims" in feature_lower:
        return str(int(original_value))

    if "years_licensed" in feature_lower:
        return f"{int(original_value)} years"

    if "vehicle_age" in feature_lower:
        return f"{int(original_value)} years"

    if "age" in feature_lower:
        return str(int(original_value))

    return str(original_value)


def extract_claim_shap_values(
    explainer,
    transformed_input: np.ndarray
) -> np.ndarray:
    """
    Extract SHAP values for the positive fraud class across
    different SHAP output formats.
    """

    explanation = explainer(transformed_input)

    values = np.asarray(explanation.values)

    # Common binary-class format:
    # rows x features
    if values.ndim == 2:
        return values[0]

    # Possible format:
    # rows x features x classes
    if values.ndim == 3:
        return values[0, :, -1]

    # Possible single-row format:
    # features
    if values.ndim == 1:
        return values

    raise ValueError(
        f"Unexpected SHAP output shape: {values.shape}"
    )


def get_shap_explanation(
    pipeline: Any,
    input_data: pd.DataFrame,
    top_n: int = 10
) -> pd.DataFrame:
    """
    Generate a local SHAP explanation for one claim.

    Returns a business-friendly DataFrame with:
    - feature name
    - original claim value
    - SHAP contribution
    - direction
    """

    if len(input_data) != 1:
        raise ValueError(
            "SHAP explanation requires exactly one claim."
        )

    if "preprocessor" not in pipeline.named_steps:
        raise ValueError(
            "Pipeline does not contain a 'preprocessor' step."
        )

    if "model" not in pipeline.named_steps:
        raise ValueError(
            "Pipeline does not contain a 'model' step."
        )

    preprocessor = pipeline.named_steps["preprocessor"]
    xgb_model = pipeline.named_steps["model"]

    transformed_input = preprocessor.transform(input_data)

    if hasattr(transformed_input, "toarray"):
        transformed_input = transformed_input.toarray()

    transformed_input = np.asarray(transformed_input)

    feature_names = preprocessor.get_feature_names_out()

    if transformed_input.shape[1] != len(feature_names):
        raise ValueError(
            "The number of transformed columns does not match "
            "the number of feature names."
        )

    explainer = shap.TreeExplainer(xgb_model)

    claim_shap_values = extract_claim_shap_values(
        explainer=explainer,
        transformed_input=transformed_input
    )

    if len(claim_shap_values) != len(feature_names):
        raise ValueError(
            "The number of SHAP values does not match "
            "the number of transformed features."
        )

    rows = []

    for index, technical_feature in enumerate(feature_names):
        transformed_value = transformed_input[0][index]
        shap_value = float(claim_shap_values[index])

        # Ignore inactive one-hot encoded categories.
        if (
            technical_feature.startswith("cat__")
            and transformed_value == 0
        ):
            continue

        original_value = get_original_value(
            technical_feature=technical_feature,
            input_data=input_data
        )

        rows.append({
            "Technical Feature": technical_feature,
            "Feature": clean_feature_name(technical_feature),
            "Original Value": format_original_value(
                technical_feature,
                original_value
            ),
            "SHAP Value": shap_value,
            "Absolute Contribution": abs(shap_value),
            "Direction": (
                "Increases Fraud Risk"
                if shap_value > 0
                else (
                    "Decreases Fraud Risk"
                    if shap_value < 0
                    else "Neutral"
                )
            )
        })

    explanation_df = pd.DataFrame(rows)

    if explanation_df.empty:
        return explanation_df

    explanation_df = explanation_df.sort_values(
        "Absolute Contribution",
        ascending=False
    )

    return explanation_df.head(top_n).reset_index(drop=True)