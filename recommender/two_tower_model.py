import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import os
import re

print("Running Two-Tower Training...")


# LOAD DATA


interactions = pd.read_csv(
    "data/processed/user_product_interactions.csv"
)

# Vocabulary is scoped to CF candidates only.
# This keeps the embedding tables to a manageable size —
# no point learning embeddings for products that will
# never appear in the ranking stage anyway.
candidates = pd.read_csv(
    "data/processed/cf_candidates.csv"
)

candidate_products = set(candidates["product_id"].unique())
candidate_users = set(candidates["user_id"].unique())

# Filter interactions to only users and products in the candidate set
interactions = interactions[
    interactions["user_id"].isin(candidate_users) &
    interactions["product_id"].isin(candidate_products)
].copy()

print(
    f"Interactions after vocab filter: {len(interactions)} rows | "
    f"{interactions['user_id'].nunique()} users | "
    f"{interactions['product_id'].nunique()} products"
)


# NEGATIVE SAMPLING (scoped to candidate products only)


interactions["label"] = 1

# Only sample negatives from the candidate product pool —
# not the full product catalogue
neg_pool = list(candidate_products)

user_bought_map = (
    interactions
    .groupby("user_id")["product_id"]
    .apply(set)
    .to_dict()
)

neg_samples = []

for user, bought in user_bought_map.items():
    not_bought = [p for p in neg_pool if p not in bought]

    sampled = np.random.choice(
        not_bought,
        size=min(5, len(not_bought)),
        replace=False
    )

    for p in sampled:
        neg_samples.append([user, p, 0])

neg_df = pd.DataFrame(
    neg_samples,
    columns=["user_id", "product_id", "label"]
)

train_df = pd.concat([interactions, neg_df], ignore_index=True)

print(f"Training rows: {len(train_df)}")


# ENCODE IDS


user_ids = train_df["user_id"].unique()
product_ids = train_df["product_id"].unique()

user_map = {u: i for i, u in enumerate(user_ids)}
product_map = {p: i for i, p in enumerate(product_ids)}

train_df["user_id_enc"] = train_df["user_id"].map(user_map)
train_df["product_id_enc"] = train_df["product_id"].map(product_map)

print(
    f"Vocab size → users: {len(user_map)} | "
    f"products: {len(product_map)}"
)

# Estimate embedding memory before allocating
emb_bytes = (len(user_map) + len(product_map)) * 64 * 4
print(f"Estimated embedding memory: {emb_bytes / 1e6:.1f} MB")


# MODEL


class TwoTower(nn.Module):

    def __init__(self, n_users, n_products, emb_dim=64):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.prod_emb = nn.Embedding(n_products, emb_dim)

    def forward(self, u, p):
        u_vec = self.user_emb(u)
        p_vec = self.prod_emb(p)
        return torch.sigmoid(
            (u_vec * p_vec).sum(dim=1)
        )


model = TwoTower(len(user_map), len(product_map))

print("Training Two-Tower model...")


# TRAIN WITH MINI-BATCHES
# Full-dataset forward pass also risks OOM at scale —
# batching keeps peak memory flat regardless of dataset size.


optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.BCELoss()

u_all = torch.tensor(train_df["user_id_enc"].values)
p_all = torch.tensor(train_df["product_id_enc"].values)
y_all = torch.tensor(train_df["label"].values).float()

n = len(train_df)
batch_size = 4096
n_epochs = 5

for epoch in range(n_epochs):

    # Shuffle each epoch
    perm = torch.randperm(n)
    u_all = u_all[perm]
    p_all = p_all[perm]
    y_all = y_all[perm]

    epoch_loss = 0.0
    n_batches = 0

    for start in range(0, n, batch_size):
        u = u_all[start: start + batch_size]
        p = p_all[start: start + batch_size]
        y = y_all[start: start + batch_size]

        preds = model(u, p)
        loss = loss_fn(preds, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        n_batches += 1

    print(
        f"Epoch {epoch + 1}/{n_epochs}  "
        f"Loss {epoch_loss / n_batches:.4f}"
    )


# SAVE MAPS


os.makedirs("data/processed", exist_ok=True)

pd.DataFrame({
    "user_id": list(user_map.keys()),
    "user_idx": list(user_map.values())
}).to_csv("data/processed/user_map.csv", index=False)

pd.DataFrame({
    "product_id": list(product_map.keys()),
    "product_idx": list(product_map.values())
}).to_csv("data/processed/product_map.csv", index=False)

print("Maps saved")


# SAVE MODEL


folder = "recommender"
os.makedirs(folder, exist_ok=True)

files = [
    f for f in os.listdir(folder)
    if f.startswith("two_tower_") and f.endswith(".ckpt")
]

versions = []
for f in files:
    match = re.search(r"two_tower_(\d+)\.ckpt", f)
    if match:
        versions.append(int(match.group(1)))

v = max(versions) + 1 if versions else 1
path = f"{folder}/two_tower_{v}.ckpt"

torch.save(model.state_dict(), path)

print(f"Model saved → two_tower_{v}.ckpt")