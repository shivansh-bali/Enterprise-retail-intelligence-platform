import pandas as pd
import numpy as np
import os
from sklearn.utils import shuffle

print(" Training BPR Ranking Model...")


feedback_path = "data/feedback/user_feedback.csv"
rec_path = "data/processed"

feedback = pd.read_csv(feedback_path)

rec_files = [
    f for f in os.listdir(rec_path)
    if f.startswith("global_recommendations_")
]

latest = sorted(rec_files)[-1]
recs = pd.read_csv(f"{rec_path}/{latest}")


event_weight = {
    "view": 0.1,
    "click": 3,
    "purchase": 8
}

feedback["score"] = feedback["event"].map(event_weight)


feedback_agg = (
    feedback.groupby(["user_id", "product_id"])["score"]
    .max()
    .reset_index()
)


df = recs.merge(
    feedback_agg,
    on=["user_id", "product_id"],
    how="left"
)

df["score"] = df["score"].fillna(0)


valid = df["forecast_qty"] > 0

df["forecast_norm"] = 0

if valid.sum() > 0:
    min_f = df.loc[valid, "forecast_qty"].min()
    max_f = df.loc[valid, "forecast_qty"].max()

    df.loc[valid, "forecast_norm"] = (
        (df.loc[valid, "forecast_qty"] - min_f) /
        (max_f - min_f + 1e-8)
    )


pairs = []

for user in df["user_id"].unique():

    user_df = df[df["user_id"] == user]

    pos_items = user_df[user_df["score"] > 0]
    neg_items = user_df[user_df["score"] == 0]

    if len(pos_items) == 0 or len(neg_items) == 0:
        continue

    neg_sample = neg_items.sample(
        n=min(len(pos_items) * 2, len(neg_items)),
        random_state=42
    )

    for _, pos in pos_items.iterrows():
        for _, neg in neg_sample.iterrows():

            pairs.append([
                pos["affinity_score"],
                pos["forecast_norm"],
                neg["affinity_score"],
                neg["forecast_norm"]
            ])

pairs = np.array(pairs)

print(f"Generated {len(pairs)} training pairs")


w = np.random.randn(2)  
lr = 0.01
epochs = 5

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

for epoch in range(epochs):

    pairs = shuffle(pairs)

    total_loss = 0

    for row in pairs:

        pos = row[:2]
        neg = row[2:]

        score_pos = np.dot(w, pos)
        score_neg = np.dot(w, neg)

        diff = score_pos - score_neg
        prob = sigmoid(diff)

        loss = -np.log(prob + 1e-8)
        total_loss += loss

        grad = (1 - prob)

        w += lr * grad * (pos - neg)

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")


w = np.abs(w)
w = w / (w.sum() + 1e-8)

affinity_weight = float(w[0])
forecast_weight = float(w[1])

print(f"\n Learned Weights:")
print(f"Affinity: {affinity_weight:.3f}")
print(f"Forecast: {forecast_weight:.3f}")



config_folder = "config"
os.makedirs(config_folder, exist_ok=True)

pd.DataFrame([{
    "affinity_weight": affinity_weight,
    "forecast_weight": forecast_weight
}]).to_csv(
    f"{config_folder}/ranking_weights.csv",
    index=False
)

print("BPR training complete & weights saved")