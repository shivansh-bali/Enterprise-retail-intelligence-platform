from fastapi import FastAPI
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware  # ✅ ADD THIS
import pandas as pd
import torch
import torch.nn as nn
import os
import re
from pydantic import BaseModel
from datetime import datetime
from functools import lru_cache


# PATHS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

data_folder = os.path.join(ROOT_DIR, "data", "processed")
model_folder = os.path.join(ROOT_DIR, "recommender")


# GLOBAL OBJECTS

forecast = None
demand = None
candidates = None
user_map = None
product_map = None
model = None
user_candidate_map = None
global_recs = None


# LOAD GLOBAL RECOMMENDATIONS

def load_latest_global_recs():
    files = [
        f for f in os.listdir(data_folder)
        if f.startswith("global_recommendations_")
    ]

    if not files:
        print(" No global recommendations found")
        return None

    latest_v = max([
        int(re.search(r"global_recommendations_(\d+).csv", f).group(1))
        for f in files
    ])

    path = os.path.join(
        data_folder,
        f"global_recommendations_{latest_v}.csv"
    )

    print(f" Loaded global recs v{latest_v}")
    return pd.read_csv(path)


# MODEL

class TwoTower(nn.Module):
    def __init__(self, n_users, n_products):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, 64)
        self.prod_emb = nn.Embedding(n_products, 64)

    def forward(self, u, p):
        u_vec = self.user_emb(u)
        p_vec = self.prod_emb(p)
        return torch.sigmoid((u_vec * p_vec).sum(dim=1))


# LIFESPAN

@asynccontextmanager
async def lifespan(app: FastAPI):
    global forecast, demand, candidates, user_map, product_map, model, user_candidate_map, global_recs

    print(" Loading resources...")

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

    candidates = pd.read_csv(os.path.join(data_folder, "cf_candidates.csv"))

    user_candidate_map = {
        uid: group["product_id"].tolist()
        for uid, group in candidates.groupby("user_id")
    }

    user_map_df = pd.read_csv(os.path.join(data_folder, "user_map.csv"))
    product_map_df = pd.read_csv(os.path.join(data_folder, "product_map.csv"))

    user_map = dict(zip(user_map_df["user_id"], user_map_df["user_idx"]))
    product_map = dict(zip(product_map_df["product_id"], product_map_df["product_idx"]))

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

    global_recs = load_latest_global_recs()

    print(" API READY")
    yield


# CREATE APP

app = FastAPI(lifespan=lifespan)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# RECOMMEND

@lru_cache(maxsize=1000)
def compute_recommendations(user_id, top_k):

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
   
    # LOAD RANKING WEIGHTS

    config_path = "config/ranking_weights.csv"

    if os.path.exists(config_path):
        weights = pd.read_csv(config_path).iloc[0]

        affinity_w = weights["affinity_weight"]
        forecast_w = weights["forecast_weight"]

        print(f"Using learned weights → A:{affinity_w:.3f}, F:{forecast_w:.3f}")

    else:
        affinity_w = 0.7
        forecast_w = 0.3

    print("Using default weights → A:0.7, F:0.3")
    final_df["final_score"] = (
        affinity_w * final_df["affinity_score"] +
        forecast_w * ((final_df["forecast_qty"] - final_df["forecast_qty"].min()) / denom)
    )

    return final_df.sort_values("final_score", ascending=False).head(top_k).to_dict(orient="records")

@app.get("/recommend")
def recommend(user_id: int, top_k: int = 10):
    results = compute_recommendations(user_id, top_k)
    return results if results else {"error": "No recommendations"}

# ANALYTICS APIs

@app.get("/analytics/products")
def product_analytics():
    if global_recs is None:
        return {"error": "No global recommendations"}

    df = global_recs.copy()

    df["impact"] = df["final_score"] * df["forecast_qty"]

    return {
        "top_products": df.groupby("product_id")["final_score"].mean().nlargest(10).reset_index().to_dict("records"),
        "top_demand": df.groupby("product_id")["forecast_qty"].mean().nlargest(10).reset_index().to_dict("records"),
        "high_impact": df.groupby("product_id")["impact"].mean().nlargest(10).reset_index().to_dict("records")
    }

@app.get("/analytics/stock")
def stock():
    if global_recs is None:
        return {"error": "No global recommendations"}

    df = global_recs.copy()
    df["impact"] = df["final_score"] * df["forecast_qty"]

    stock_df = df.groupby("product_id").mean().reset_index().sort_values("impact", ascending=False).head(20)

    stock_df["priority"] = pd.cut(stock_df["impact"], bins=3, labels=["LOW", "MEDIUM", "HIGH"])

    return stock_df.to_dict("records")

@app.get("/analytics/summary")
def summary():
    if global_recs is None:
        return {"error": "No global recommendations"}

    df = global_recs.copy()
    df["impact"] = df["final_score"] * df["forecast_qty"]

    return {
        "total_products": int(df["product_id"].nunique()),
        "avg_score": float(df["final_score"].mean()),
        "avg_demand": float(df["forecast_qty"].mean()),
        "avg_impact": float(df["impact"].mean())
    }



# FEEDBACK

feedback_folder = os.path.join(ROOT_DIR, "data", "feedback")
os.makedirs(feedback_folder, exist_ok=True)

feedback_file = os.path.join(feedback_folder, "user_feedback.csv")

class Feedback(BaseModel):
    user_id: int
    product_id: str
    event: str  # view / click / purchase


@app.post("/feedback")
def log_feedback(data: Feedback):

    record = {
        "user_id": data.user_id,
        "product_id": data.product_id,
        "event": data.event,
        "timestamp": datetime.now().isoformat()
    }

    pd.DataFrame([record]).to_csv(
        feedback_file,
        mode="a" if os.path.exists(feedback_file) else "w",
        header=not os.path.exists(feedback_file),
        index=False
    )

    return {"status": "logged"}