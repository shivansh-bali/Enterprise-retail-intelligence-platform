import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import os
import re

print("Running Final Ranking Pipeline...")

data_folder = "data/processed"
model_folder = "recommender"


# LOAD LATEST FORECAST


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
forecast_path = os.path.join(
    data_folder, f"forecast_tft_{latest_forecast_v}.csv"
)
forecast = pd.read_csv(forecast_path)

print(f"Loaded forecast → v{latest_forecast_v}")


# LOAD LATEST MODEL


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
model_path = os.path.join(
    model_folder, f"two_tower_{latest_model_v}.ckpt"
)

print(f"Loaded Two-Tower model → v{latest_model_v}")


# LOAD CANDIDATES, INTERACTIONS, AND MAPS


candidates = pd.read_csv(
    os.path.join(data_folder, "cf_candidates.csv")
)

interactions = pd.read_csv(
    os.path.join(data_folder, "user_product_interactions.csv")
)

user_map_df = pd.read_csv(
    os.path.join(data_folder, "user_map.csv")
)
product_map_df = pd.read_csv(
    os.path.join(data_folder, "product_map.csv")
)

user_map = dict(zip(user_map_df["user_id"], user_map_df["user_idx"]))
product_map = dict(zip(product_map_df["product_id"], product_map_df["product_idx"]))

n_users = len(user_map)
n_products = len(product_map)


# LOAD MODEL


class TwoTower(nn.Module):

    def __init__(self, n_users, n_products):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, 64)
        self.prod_emb = nn.Embedding(n_products, 64)

    def forward(self, u, p):
        u_vec = self.user_emb(u)
        p_vec = self.prod_emb(p)
        return torch.sigmoid(
            (u_vec * p_vec).sum(dim=1)
        )


model = TwoTower(n_users, n_products)
model.load_state_dict(torch.load(model_path))
model.eval()


# PRECOMPUTE DEMAND


demand = (
    forecast
    .groupby("product_id")["forecast_qty"]
    .mean()
    .reset_index()
)
demand["product_id"] = demand["product_id"].astype(str)

forecast_min = demand["forecast_qty"].min()
forecast_max = demand["forecast_qty"].max()
forecast_range = forecast_max - forecast_min + 1e-8

demand["forecast_norm"] = (
    (demand["forecast_qty"] - forecast_min) / forecast_range
)


# LOAD RANKING WEIGHTS


weights_path = "config/ranking_weights.csv"\

if os.path.exists(weights_path):
    weights = pd.read_csv(weights_path)
    aw = weights["affinity_weight"][0]
    fw = weights["forecast_weight"][0]
else:
    aw = 0.7
    fw = 0.3


# SCORE ALL USERS


all_user_ids = candidates["user_id"].unique()
all_rankings = []

print(f"Scoring {len(all_user_ids)} users...")

with torch.no_grad():
    for user_id in all_user_ids:
        user_candidates = candidates[
            candidates["user_id"] == user_id
        ]["product_id"]

        if user_id not in user_map:
            continue

        user_idx = user_map[user_id]

        valid_products = [
            p for p in user_candidates
            if p in product_map
        ]

        if not valid_products:
            continue

        candidate_idx = [product_map[p] for p in valid_products]

        user_tensor = torch.tensor(
            [user_idx] * len(candidate_idx)
        )
        product_tensor = torch.tensor(candidate_idx)

        scores = model(
            user_tensor, product_tensor
        ).numpy()

        ranking_df = pd.DataFrame({
            "user_id": user_id,
            "product_id": [str(p) for p in valid_products],
            "affinity_score": scores
        })

        all_rankings.append(ranking_df)

print("Merging demand signal...")

final_df = pd.concat(all_rankings, ignore_index=True)

final_df = final_df.merge(
    demand[["product_id", "forecast_norm"]],
    on="product_id",
    how="left"
)

final_df["forecast_norm"] = final_df["forecast_norm"].fillna(0)

final_df["final_score"] = (
    aw * final_df["affinity_score"]
    + fw * final_df["forecast_norm"]
)

final_df = (
    final_df
    .sort_values(["user_id", "final_score"], ascending=[True, False])
)

print(f"Total recommendation rows: {len(final_df)}")


# SAVE VERSIONED OUTPUT


rec_files = [
    f for f in os.listdir(data_folder)
    if f.startswith("recommendations_") and f.endswith(".csv")
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

final_df.to_csv(save_path, index=False)

print(f"Recommendations saved → v{next_version}")
