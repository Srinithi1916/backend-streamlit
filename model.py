import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from joblib import dump

# File paths
policy_file = "Policy_Driven_Carbon_Reduction_1640.csv"
eco_file = "EcoBuild_Sustainability_Records.csv"
model_file = "mcdm_model.joblib"

# Criteria and metadata
criteria = [
    "Embodied_Emissions_tCO2e", "Operational_Emissions_tCO2e", "Material_Reuse_%",
    "Renewable_Energy_%", "Waste_Minimization_%", "Urban_Sustainability_Score",
    "Total_Lifecycle_Emissions_tCO2e", "Energy_Efficiency_Score", "Carbon_Footprint_Percent"
]
benefit_flags = [False, False, True, True, True, True, False, True, False]
weights = np.array([0.2, 0.15, 0.1, 0.1, 0.1, 0.1, 0.15, 0.07, 0.03])
weights /= weights.sum()

# Load and merge
policy_df = pd.read_csv(policy_file)
eco_df = pd.read_csv(eco_file)
merged = pd.merge(policy_df, eco_df, on="Project_ID", how="outer")

# Fill missing columns
defaults = {
    "Embodied_Emissions_tCO2e": 500.0, "Operational_Emissions_tCO2e": 600.0,
    "Material_Reuse_%": 30.0, "Renewable_Energy_%": 40.0, "Waste_Minimization_%": 20.0,
    "Urban_Sustainability_Score": 70.0, "Energy_Efficiency_Score": 85.0,
    "Carbon_Footprint_Percent": 30.0
}
for col, val in defaults.items():
    if col not in merged.columns:
        merged[col] = val
    else:
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(val)

# Compute lifecycle emissions
merged["Total_Lifecycle_Emissions_tCO2e"] = merged.get("Total_Lifecycle_Emissions_tCO2e")
if merged["Total_Lifecycle_Emissions_tCO2e"].isnull().any():
    merged["Total_Lifecycle_Emissions_tCO2e"] = (
        merged["Embodied_Emissions_tCO2e"] + merged["Operational_Emissions_tCO2e"]
    )

# Final training frame
train_df = merged[criteria].dropna()
if train_df.empty:
    raise ValueError("No valid rows for training.")

# Normalize and score
scaler = MinMaxScaler()
X_norm = scaler.fit_transform(train_df)
weighted = X_norm * weights
ideal = np.where(benefit_flags, weighted.max(axis=0), weighted.min(axis=0))
anti_ideal = np.where(benefit_flags, weighted.min(axis=0), weighted.max(axis=0))
d_pos = np.linalg.norm(weighted - ideal, axis=1)
d_neg = np.linalg.norm(weighted - anti_ideal, axis=1)
scores = d_neg / (d_pos + d_neg + 1e-12)

# Quantiles and classification
q33, q66 = np.quantile(scores, 0.33), np.quantile(scores, 0.66)
classes = ["Low" if s <= q33 else "Medium" if s <= q66 else "High" for s in scores]

# Save model
model = {
    "criteria": criteria,
    "weights": weights,
    "benefit_flags": benefit_flags,
    "scaler": scaler,
    "ideal": ideal,
    "anti_ideal": anti_ideal,
    "quantiles": {"q33": q33, "q66": q66},
    "class_labels": ["Low", "Medium", "High"]
}
dump(model, model_file)
print(f"✅ Model saved to {model_file}")
print(f"Quantiles: q33 = {q33:.4f}, q66 = {q66:.4f}")
