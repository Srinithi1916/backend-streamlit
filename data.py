import pandas as pd
import numpy as np
import random

# Constants
NUM_RECORDS = 100000
locations = ['Mumbai', 'Delhi', 'Bengaluru', 'Chennai', 'Hyderabad', 'Pune', 'Ahmedabad', 'Jaipur', 'Kochi', 'Kolkata']
project_types = ['Residential', 'Commercial', 'Industrial', 'Institutional', 'Mixed-Use']
cert_levels = ['None', 'Bronze', 'Silver', 'Gold', 'Platinum']
policies = ['None', 'Tax Incentives', 'Emission Caps', 'Material Subsidies', 'Renewable Mandates']
phases = ['Design', 'Construction', 'Operation', 'Renovation', 'Demolition']
materials = ['Wood', 'Concrete', 'Steel', 'Glass']
cert_progress_levels = ['Bronze', 'Silver', 'Gold']
waste_strategies = ['Reduction', 'Segregation', 'Disposal', 'Standards']

# Helper functions
def simulate_emission(area, cert, phase, policy):
    base = 0.5
    phase_mult = {'Design': 0.6, 'Construction': 1.2, 'Operation': 0.8, 'Renovation': 0.9, 'Demolition': 1.0}
    cert_disc = {'None': 1.0, 'Bronze': 0.9, 'Silver': 0.8, 'Gold': 0.7, 'Platinum': 0.6}
    policy_mult = {'None': 1.0, 'Tax Incentives': 0.95, 'Emission Caps': 0.85, 'Material Subsidies': 0.9, 'Renewable Mandates': 0.8}
    return round(base * area * phase_mult[phase] * cert_disc[cert] * policy_mult[policy], 2)

# Generate data
data = []
for i in range(NUM_RECORDS):
    pid = f"P{i+1:05d}"
    loc = random.choice(locations)
    ptype = random.choice(project_types)
    area = random.randint(50, 10000)
    cert = random.choice(cert_levels)
    policy = random.choice(policies)
    phase = random.choice(phases)
    emission = simulate_emission(area, cert, phase, policy)
    material = random.choice(materials)
    cf_percent = random.choice([10, 30, 50])
    energy = round(area * random.uniform(5, 15), 2)
    year = random.choice([2020, 2021, 2022, 2023, 2024, 2025])
    cert_prog = random.choice(cert_progress_levels)
    eff_score = random.randint(50, 100)
    waste_gen = round(area * random.uniform(0.2, 1.0), 2)
    recycled = round(waste_gen * random.uniform(0.3, 0.9), 2)
    forecast = round(energy * random.uniform(1.1, 1.5), 2)
    bio_pct = random.randint(40, 70)
    nonbio_pct = 100 - bio_pct
    strategy = random.choice(waste_strategies)

    data.append([
        pid, loc, ptype, area, cert, policy, phase, emission, material, cf_percent,
        energy, year, cert_prog, eff_score, waste_gen, recycled, forecast,
        bio_pct, nonbio_pct, strategy
    ])

# Create DataFrame and export
columns = [
    "Project_ID", "Project_Location", "Project_Type", "Building_Area_m2",
    "Green_Certification_Level", "Policy_Intervention", "Lifecycle_Phase",
    "Predicted_Emission_kgCO2", "Material_Type", "Carbon_Footprint_Percent",
    "Energy_Consumption_kWh", "Certification_Progress_Year", "Certification_Level",
    "Energy_Efficiency_Score", "Waste_Generation_kg", "Recycled_Waste_kg",
    "Energy_Demand_Forecast_kWh", "Biodegradable_Waste_Percent",
    "NonBiodegradable_Waste_Percent", "Waste_Strategy"
]

df = pd.DataFrame(data, columns=columns)
df.to_csv("EcoBuild_Sustainability_Records.csv", index=False)
