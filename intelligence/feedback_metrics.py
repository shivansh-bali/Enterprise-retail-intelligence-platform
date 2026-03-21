import pandas as pd
import os

print("Running Feedback Metrics Engine...")

feedback_path = "data/feedback/user_feedback.csv"

if not os.path.exists(feedback_path):
    raise FileNotFoundError(
    "Feedback file not found"
    )

df = pd.read_csv(feedback_path)

total_views = len(
df[df["event"] == "view"]
)

total_clicks = len(
df[df["event"] == "click"]
)

total_purchases = len(
df[df["event"] == "purchase"]
)

ctr = (
total_clicks / total_views
if total_views > 0 else 0
)

conversion_rate = (
total_purchases / total_clicks
if total_clicks > 0 else 0
)

product_engagement = (
df.groupby("product_id")["event"]
.count()
.reset_index()
.sort_values("event", ascending=False)
)

top_products = product_engagement.head(10)

metrics = {
"total_views": total_views,
"total_clicks": total_clicks,
"total_purchases": total_purchases,
"ctr": round(ctr, 4),
"conversion_rate": round(conversion_rate, 4)
}

metrics_df = pd.DataFrame([metrics])

output_folder = "data/metrics"
os.makedirs(output_folder, exist_ok=True)

metrics_df.to_csv(
f"{output_folder}/feedback_summary.csv",
index=False
)

top_products.to_csv(
f"{output_folder}/top_engaged_products.csv",
index=False
)

print("Feedback metrics computed")
