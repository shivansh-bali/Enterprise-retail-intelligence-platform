import pandas as pd
import torch
import torch.nn as nn
import os
import re

print("🚀 Generating global recommendations...")

# -------------------------
# PATHS
# -------------------------
data_folder = "data/processed"
model_folder = "recommender"

# -------------------------
# LOAD MAPS
# -------------------------
user_map_df = pd.read_csv(os.path.join(data_folder, "user_map.csv"))
product_map_df = pd.read_csv(os.path.join(data_folder, "product_map.csv"))

user_map = dict(zip(user_map_df["user_id"], user_map_df["user_idx"]))
product_map = dict(zip(product_map_df["product_id"], product_map_df["product_idx"]))

all_users = list(user_map.keys())

# -------------------------
# LOAD FORECAST
# -------------------------
forecast_files = [
    f for f in os.listdir(data_folder)
    if f.startswith("forecast_tft_") and f.endswith(".csv")
]

latest_v = max([
    int(re.search(r"forecast_tft_(\d+).csv", f).group(1))
    for f in forecast_files
])

forecast_path = os.path.join(data_folder, f"forecast_tft_{latest_v}.csv")
forecast = pd.read_csv(forecast_path)

demand = (
    forecast.groupby("product_id")["forecast_qty"]
    .mean()
    .reset_index()
)

demand["product_id"] = demand["product_id"].astype(str)

# -------------------------
# LOAD CANDIDATES
# -------------------------
candidates = pd.read_csv(os.path.join(data_folder, "cf_candidates.csv"))

user_candidate_map = {
    uid: group["product_id"].tolist()
    for uid, group in candidates.groupby("user_id")
}

# -------------------------
# MODEL
# -------------------------
class TwoTower(nn.Module):
    def __init__(self, n_users, n_products):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, 64)
        self.prod_emb = nn.Embedding(n_products, 64)

    def forward(self, u, p):
        u_vec = self.user_emb(u)
        p_vec = self.prod_emb(p)
        return torch.sigmoid((u_vec * p_vec).sum(dim=1))


model_files = [
    f for f in os.listdir(model_folder)
    if f.startswith("two_tower_") and f.endswith(".ckpt")
]

latest_model_v = max([
    int(re.search(r"two_tower_(\d+).ckpt", f).group(1))
    for f in model_files
])

model_path = os.path.join(model_folder, f"two_tower_{latest_model_v}.ckpt")

model = TwoTower(len(user_map), len(product_map))
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

# -------------------------
# RECOMMENDATION FUNCTION
# -------------------------
def compute_recommendations(user_id, top_k=20):

    if user_id not in user_map:
        return []

    user_idx = user_map[user_id]

    user_candidates = user_candidate_map.get(user_id, [])[:200]

    valid_products = [p for p in user_candidates if p in product_map]

    if not valid_products:
        return []

    candidate_idx = [product_map[p] for p in valid_products]

    with torch.no_grad():
        scores = model(
            torch.tensor([user_idx] * len(candidate_idx)),
            torch.tensor(candidate_idx)
        ).numpy()

    ranking_df = pd.DataFrame({
        "product_id": [str(p) for p in valid_products],
        "affinity_score": scores
    })

    final_df = ranking_df.merge(demand, on="product_id", how="left")
    final_df["forecast_qty"] = final_df["forecast_qty"].fillna(0)

    denom = final_df["forecast_qty"].max() - final_df["forecast_qty"].min() + 1e-8

    final_df["forecast_norm"] = (
        (final_df["forecast_qty"] - final_df["forecast_qty"].min()) / denom
    )

    final_df["final_score"] = (
        0.7 * final_df["affinity_score"]
        + 0.3 * final_df["forecast_norm"]
    )

    return final_df.sort_values(
        "final_score", ascending=False
    ).head(top_k).to_dict(orient="records")


# -------------------------
# GENERATE GLOBAL RECS
# -------------------------
all_recs = []

for user in all_users:
    recs = compute_recommendations(user, top_k=20)

    for r in recs:
        r["user_id"] = user
        all_recs.append(r)

df = pd.DataFrame(all_recs)

# -------------------------
# SAVE WITH VERSIONING
# -------------------------
rec_files = [
    f for f in os.listdir(data_folder)
    if f.startswith("global_recommendations_")
]

versions = [
    int(re.search(r"global_recommendations_(\d+).csv", f).group(1))
    for f in rec_files
    if re.search(r"global_recommendations_(\d+).csv", f)
]

next_version = max(versions) + 1 if versions else 1

save_path = os.path.join(
    data_folder,
    f"global_recommendations_{next_version}.csv"
)

df.to_csv(save_path, index=False)

print(f" Done! Saved → global_recommendations_{next_version}.csv")