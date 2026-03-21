import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import os

print(" Running Two-Tower Ranking...")



#  LOAD DATA



transactions = pd.read_csv(
"data/processed/cf_candidates.csv"
)

interactions = pd.read_csv(
"data/processed/user_product_interactions.csv"
)



#  NEGATIVE SAMPLING


interactions["label"] = 1

users = interactions["user_id"].unique()
all_products = interactions["product_id"].unique()

neg_samples = []

user_bought_map = (
interactions
.groupby("user_id")["product_id"]
.apply(set)
.to_dict()
)

all_products_set = set(all_products)

for user in users:

    bought = user_bought_map[user]

    not_bought = list(
        all_products_set - bought
    )

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

train_df = pd.concat([
interactions,
neg_df
])


print("now encode ids")
#  ENCODE IDS



user_ids = train_df["user_id"].unique()
product_ids = train_df["product_id"].unique()

user_map = {
u:i for i,u in enumerate(user_ids)
}

product_map = {
p:i for i,p in enumerate(product_ids)
}

train_df["user_id_enc"] = train_df["user_id"].map(user_map)
train_df["product_id_enc"] = train_df["product_id"].map(product_map)


print("now model")
# MODEL



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


model = TwoTower(
len(user_ids),
len(product_ids)
)


print("now training")
# TRAIN


optimizer = torch.optim.Adam(
model.parameters(),
lr=0.001
)

loss_fn = nn.BCELoss()

u = torch.tensor(train_df["user_id_enc"].values)
p = torch.tensor(train_df["product_id_enc"].values)
y = torch.tensor(train_df["label"].values).float()

for epoch in range(5):


    preds = model(u, p)

    loss = loss_fn(preds, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1} Loss {loss.item():.4f}")




#  SAVE MODEL

import re

folder = "recommender"

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
