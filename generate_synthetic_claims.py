import numpy as np
import pandas as pd

np.random.seed(42)

N = 15000

provinces = ["ON", "QC", "BC", "AB", "MB", "SK", "NS", "NB", "NL", "PE"]
genders = ["Male", "Female", "Other"]
employment_statuses = ["Employed", "Unemployed"]
vehicle_brands = ["Toyota", "Honda", "Ford", "Chevrolet", "BMW", "Mercedes", "Hyundai", "Kia", "Nissan", "Tesla"]
vehicle_uses = ["Personal", "Business"]
claim_types = ["Collision", "Theft", "Vandalism", "Weather", "Injury", "Glass"]

postal_prefixes = {
    "ON": ["M", "L", "K", "N", "P"],
    "QC": ["H", "G", "J"],
    "BC": ["V"],
    "AB": ["T"],
    "MB": ["R"],
    "SK": ["S"],
    "NS": ["B"],
    "NB": ["E"],
    "NL": ["A"],
    "PE": ["C"]
}

data = []

for i in range(N):
    province = np.random.choice(provinces, p=[0.38, 0.22, 0.13, 0.12, 0.04, 0.03, 0.03, 0.02, 0.02, 0.01])
    postal_code = np.random.choice(postal_prefixes[province]) + str(np.random.randint(1, 10)) + chr(np.random.randint(65, 91))

    age = int(np.clip(np.random.normal(42, 14), 18, 85))
    gender = np.random.choice(genders, p=[0.49, 0.49, 0.02])
    years_licensed = int(np.clip(age - np.random.randint(16, 25), 0, 65))

    employment_status = np.random.choice(employment_statuses, p=[0.82, 0.18])
    vehicle_brand = np.random.choice(vehicle_brands)
    vehicle_age = int(np.clip(np.random.exponential(6), 0, 25))
    vehicle_use = np.random.choice(vehicle_uses, p=[0.84, 0.16])

    vehicle_value = int(np.clip(np.random.normal(26000, 12000), 3000, 90000))
    claim_type = np.random.choice(claim_types, p=[0.42, 0.13, 0.09, 0.14, 0.12, 0.10])

    previous_claims = np.random.poisson(0.7)

    policy_tenure_months = int(np.clip(np.random.exponential(42), 1, 240))

    base_claim = {
        "Collision": 6500,
        "Theft": 11000,
        "Vandalism": 3000,
        "Weather": 4500,
        "Injury": 16000,
        "Glass": 1200
    }[claim_type]

    claim_amount = int(np.clip(np.random.normal(base_claim, base_claim * 0.55), 300, 120000))

    police_report_filed = np.random.choice(["Yes", "No"], p=[0.68, 0.32])
    witness_present = np.random.choice(["Yes", "No"], p=[0.46, 0.54])

    claim_to_vehicle_value_ratio = claim_amount / vehicle_value

    fraud_score = 0

    if claim_amount > 25000:
        fraud_score += 1
    if claim_to_vehicle_value_ratio > 0.85:
        fraud_score += 2
    if previous_claims >= 3:
        fraud_score += 2
    if policy_tenure_months <= 6 and claim_amount > 10000:
        fraud_score += 2
    if police_report_filed == "No" and claim_amount > 12000:
        fraud_score += 1
    if witness_present == "No" and claim_type in ["Collision", "Injury"]:
        fraud_score += 1
    if years_licensed < 3 and claim_amount > 8000:
        fraud_score += 1
    if vehicle_use == "Business" and previous_claims >= 2:
        fraud_score += 1
    if employment_status == "Unemployed" and claim_amount > 15000:
        fraud_score += 1
    if province in ["ON", "QC", "BC"] and claim_type in ["Theft", "Injury"]:
        fraud_score += 1

    fraud_probability = min(0.04 + fraud_score * 0.11, 0.85)
    fraud_flag = np.random.choice([0, 1], p=[1 - fraud_probability, fraud_probability])

    data.append({
        "claim_id": f"CLM{i+1:06d}",
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
        "claim_to_vehicle_value_ratio": round(claim_to_vehicle_value_ratio, 3),
        "fraud_flag": fraud_flag
    })

df = pd.DataFrame(data)

df.to_csv("synthetic_auto_insurance_claims.csv", index=False)

print("Dataset created successfully.")
print(df.head())
print()
print("Shape:", df.shape)
print("Fraud rate:", round(df["fraud_flag"].mean() * 100, 2), "%")