from fastapi import FastAPI
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import os
import re
from pydantic import BaseModel
from datetime import datetime
app = FastAPI()

data_folder = "data/processed"
model_folder = "recommender"

forecast_files = [
f for f in os.listdir(data_folder)
if f.startswith("forecast_tft_") and f.endswith(".csv")
]

forecast_versions = []

for f in forecast_files:
    match = re.search(r"forecast_tft_(\d+).csv", f)
    if match:
        forecast_versions.append(int(match.group(1)))

latest_forecast_v = max(forecast_versions)

forecast_path = os.path.join(
data_folder,
f"forecast_tft_{latest_forecast_v}.csv"
)

forecast = pd.read_csv(forecast_path)

demand = (
forecast.groupby("product_id")[
"forecast_qty"
]
.mean()
.reset_index()
)

demand["product_id"] = demand["product_id"].astype(str)

model_files = [
f for f in os.listdir(model_folder)
if f.startswith("two_tower_") and f.endswith(".ckpt")
]

model_versions = []

for f in model_files:
    match = re.search(r"two_tower_(\d+).ckpt", f)
    if match:
        model_versions.append(int(match.group(1)))

latest_model_v = max(model_versions)

model_path = os.path.join(
model_folder,
f"two_tower_{latest_model_v}.ckpt"
)

interactions = pd.read_csv(
"data/processed/user_product_interactions.csv"
)

candidates = pd.read_csv(
"data/processed/cf_candidates.csv"
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

@app.get("/recommend")
def recommend(user_id: int):


    if user_id not in user_map:
        return {"error": "User not found"}

    user_idx = user_map[user_id]

    valid_products = [
        p for p in candidates["product_id"]
        if p in product_map
    ]

    candidate_idx = [
        product_map[p]
        for p in valid_products
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

    ranking_df = pd.DataFrame({
        "product_id": valid_products,
        "affinity_score": scores
    })

    ranking_df["product_id"] = ranking_df["product_id"].astype(str)

    final_df = ranking_df.merge(
        demand,
        on="product_id",
        how="left"
    )

    final_df["forecast_qty"] = final_df["forecast_qty"].fillna(0)

    final_df["final_score"] = (
        0.7 * final_df["affinity_score"]
        + 0.3 * final_df["forecast_qty"]
    )

    final_df = final_df.sort_values(
        "final_score",
        ascending=False
    ).head(10)

    return final_df.to_dict(orient="records")

feedback_folder = "data/feedback"
os.makedirs(feedback_folder, exist_ok=True)

feedback_file = os.path.join(
feedback_folder,
"user_feedback.csv"
)

class Feedback(BaseModel):
    user_id: int
    product_id: str
    event: str

@app.post("/feedback")
def log_feedback(data: Feedback):

    record = {
        "user_id": data.user_id,
        "product_id": data.product_id,
        "event": data.event,
        "timestamp": datetime.now()
    }

    df = pd.DataFrame([record])

    if os.path.exists(feedback_file):
        df.to_csv(
            feedback_file,
            mode="a",
            header=False,
            index=False
        )
    else:
        df.to_csv(
            feedback_file,
            index=False
        )

    return {"status": "logged"}