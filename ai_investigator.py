from pydantic import BaseModel
from typing import List


class RiskFactor(BaseModel):
    factor: str
    severity: str
    explanation: str


class FraudInvestigationReport(BaseModel):
    claim_id: str
    fraud_probability: float
    risk_level: str
    risk_factors: List[RiskFactor]
    investigation_summary: str
    recommended_action: str


def generate_rule_based_report(
    claim_data: dict,
    fraud_probability: float,
    risk_level: str
) -> FraudInvestigationReport:

    risk_factors = []

    claim_amount = claim_data["claim_amount"]
    vehicle_value = claim_data["vehicle_value"]
    ratio = claim_data["claim_to_vehicle_value_ratio"]
    previous_claims = claim_data["previous_claims"]
    policy_tenure = claim_data["policy_tenure_months"]
    police_report = claim_data["police_report_filed"]
    witness = claim_data["witness_present"]
    claim_type = claim_data["claim_type"]

    if claim_amount > 25000:
        risk_factors.append(
            RiskFactor(
                factor="High claim amount",
                severity="High",
                explanation=f"The claim amount is ${claim_amount:,.0f}, which is unusually high for many auto claims."
            )
        )

    if ratio > 0.85:
        risk_factors.append(
            RiskFactor(
                factor="High claim-to-vehicle-value ratio",
                severity="High",
                explanation=f"The claim amount is {ratio:.2f} times the vehicle value of ${vehicle_value:,.0f}."
            )
        )

    if previous_claims >= 3:
        risk_factors.append(
            RiskFactor(
                factor="Multiple previous claims",
                severity="High",
                explanation=f"The claimant has {previous_claims} previous claims, indicating elevated repeat-claim risk."
            )
        )

    if policy_tenure <= 6 and claim_amount > 10000:
        risk_factors.append(
            RiskFactor(
                factor="Claim shortly after policy inception",
                severity="High",
                explanation=f"The policy has been active for only {policy_tenure} months and the claim amount is above $10,000."
            )
        )

    if police_report == "No" and claim_amount > 12000:
        risk_factors.append(
            RiskFactor(
                factor="No police report for high-value claim",
                severity="Medium",
                explanation="No police report was filed even though the claim amount is above $12,000."
            )
        )

    if witness == "No" and claim_type in ["Collision", "Injury"]:
        risk_factors.append(
            RiskFactor(
                factor="No witness for collision or injury claim",
                severity="Medium",
                explanation=f"The claim type is {claim_type}, but no witness was reported."
            )
        )

    if not risk_factors:
        risk_factors.append(
            RiskFactor(
                factor="No major rule-based risk indicators",
                severity="Low",
                explanation="The claim does not trigger the major fraud indicators currently defined in the rule-based assessment."
            )
        )

    if fraud_probability >= 0.60:
        action = "Refer to SIU for investigation"
    elif fraud_probability >= 0.30:
        action = "Review manually before settlement"
    else:
        action = "Proceed with standard claim handling"

    factor_text = ", ".join([rf.factor for rf in risk_factors])

    summary = (
        f"This claim has a fraud probability of {fraud_probability:.2%}. "
        f"The model classified it as {risk_level}. "
        f"The main risk indicators are: {factor_text}."
    )

    return FraudInvestigationReport(
        claim_id=claim_data.get("claim_id", "Manual Entry"),
        fraud_probability=fraud_probability,
        risk_level=risk_level,
        risk_factors=risk_factors,
        investigation_summary=summary,
        recommended_action=action
    )