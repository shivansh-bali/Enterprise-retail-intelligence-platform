import pandas as pd
import os

print("Running Ranking Optimization Engine...")

ab_path = "data/experiments/ab_results.csv"

if not os.path.exists(ab_path):
    raise FileNotFoundError(
    "A/B results not found"
    )

ab = pd.read_csv(ab_path)

best_group = ab.sort_values(
"conversion_rate",
ascending=False
).iloc[0]["group"]

if best_group == "A":
    affinity_weight = 0.7
    forecast_weight = 0.3
else:
    affinity_weight = 0.5
    forecast_weight = 0.5

config = pd.DataFrame([{
"affinity_weight": affinity_weight,
"forecast_weight": forecast_weight
}])

config_folder = "config"
os.makedirs(config_folder, exist_ok=True)

config.to_csv(
f"{config_folder}/ranking_weights.csv",
index=False
)

print("Ranking weights updated")
