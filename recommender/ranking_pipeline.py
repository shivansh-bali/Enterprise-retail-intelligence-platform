import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import os
import re

print("Running Final Ranking Pipeline...")

data_folder = "data/processed"
model_folder = "recommender"

forecast_files = [
f for f in os.listdir(data_folder)
if f.startswith("forecast_tft_") and f.endswith(".csv")
]

forecast_versions = []

for f in forecast_files:
    match = re.search(r"forecast_tft_(\d+)\.csv", f)
    if match:
        forecast_versions.append(int(match.group(1)))

latest_forecast_v = max(forecast_versions)
forecast_file = f"forecast_tft_{latest_forecast_v}.csv"
forecast_path = os.path.join(data_folder, forecast_file)
forecast = pd.read_csv(forecast_path)

print(f"Loaded forecast → v{latest_forecast_v}")

model_files = [
f for f in os.listdir(model_folder)
if f.startswith("two_tower_") and f.endswith(".ckpt")
]

model_versions = []

for f in model_files:
    match = re.search(r"two_tower_(\d+)\.ckpt", f)
    if match:
        model_versions.append(int(match.group(1)))

latest_model_v = max(model_versions)
model_file = f"two_tower_{latest_model_v}.ckpt"
model_path = os.path.join(model_folder, model_file)

print(f"Loaded Two-Tower model → v{latest_model_v}")

candidates = pd.read_csv(
"data/processed/cf_candidates.csv"
)

interactions = pd.read_csv(
"data/processed/user_product_interactions.csv"
)

user_ids = interactions["user_id"].unique()
product_ids = interactions["product_id"].unique()

user_map = {u:i for i,u in enumerate(user_ids)}
product_map = {p:i for i,p in enumerate(product_ids)}

class TwoTower(nn.Module):
    
    def __init__(self, n_users, n_products):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, 64)
        self.prod_emb = nn.Embedding(n_products, 64)

    def forward(self, u, p):
        u_vec = self.user_emb(u)
        p_vec = self.prod_emb(p)
        return torch.sigmoid((u_vec * p_vec).sum(dim=1))


model = TwoTower(
len(user_ids),
len(product_ids)
)

model.load_state_dict(
torch.load(model_path)
)

model.eval()

user_id = 10
user_idx = user_map[user_id]

candidate_products = candidates["product_id"]

candidate_idx = [
product_map[p]
for p in candidate_products
if p in product_map
]

user_tensor = torch.tensor(
[user_idx] * len(candidate_idx)
)

product_tensor = torch.tensor(
candidate_idx
)

scores = model(
user_tensor,
product_tensor
).detach().numpy()

valid_products = [
    p for p in candidate_products
    if p in product_map
]

ranking_df = pd.DataFrame({
    "product_id": valid_products,
    "affinity_score": scores
})


demand = (
forecast.groupby("product_id")[
"forecast_qty"
]
.mean()
.reset_index()
)
ranking_df["product_id"] = (
    ranking_df["product_id"]
    .astype(str)
)

demand["product_id"] = (
    demand["product_id"]
    .astype(str)
)


final_df = ranking_df.merge(
demand,
on="product_id",
how="left"
)

final_df["forecast_qty"] = (
    final_df["forecast_qty"].fillna(0)
)

weights_path = "config/ranking_weights.csv"

if os.path.exists(weights_path):
    weights = pd.read_csv(weights_path)
    aw = weights["affinity_weight"][0]
    fw = weights["forecast_weight"][0]
else:
    aw = 0.7
    fw = 0.3

final_df["forecast_norm"] = (
final_df["forecast_qty"]
- final_df["forecast_qty"].min()
) / (
final_df["forecast_qty"].max()
- final_df["forecast_qty"].min()
)

final_df["final_score"] = (
aw * final_df["affinity_score"]
+ fw * final_df["forecast_norm"]
)




final_df = final_df.sort_values(
"final_score",
ascending=False
)

rec_files = [
f for f in os.listdir(data_folder)
if f.startswith("recommendations_")
and f.endswith(".csv")
]

rec_versions = []

for f in rec_files:
    match = re.search(r"recommendations_(\d+)\.csv", f)
    if match:
        rec_versions.append(int(match.group(1)))

next_version = max(rec_versions) + 1 if rec_versions else 1

save_path = os.path.join(
data_folder,
f"recommendations_{next_version}.csv"
)

final_df.to_csv(
save_path,
index=False
)

print(f"Recommendations saved → v{next_version}")
