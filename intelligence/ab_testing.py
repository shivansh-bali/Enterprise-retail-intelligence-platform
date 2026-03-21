import pandas as pd
import os

print("Running A/B Testing Engine...")

feedback_path = "data/feedback/user_feedback.csv"
rec_path = "data/processed"

if not os.path.exists(feedback_path):
    raise FileNotFoundError(
    "Feedback file not found"
    )

rec_files = [
f for f in os.listdir(rec_path)
if f.startswith("recommendations_")
]

versions = [
int(f.split("_")[1].split(".")[0])
for f in rec_files
]

latest = f"recommendations_{max(versions)}.csv"

recommendations = pd.read_csv(
f"{rec_path}/{latest}"
)

feedback = pd.read_csv(feedback_path)

users = feedback["user_id"].unique()

group_a = users[:len(users)//2]
group_b = users[len(users)//2:]

feedback["group"] = feedback["user_id"].apply(
lambda x: "A" if x in group_a else "B"
)

views_a = len(
feedback[
(feedback["group"] == "A")
& (feedback["event"] == "view")
]
)

clicks_a = len(
feedback[
(feedback["group"] == "A")
& (feedback["event"] == "click")
]
)

purchases_a = len(
feedback[
(feedback["group"] == "A")
& (feedback["event"] == "purchase")
]
)

ctr_a = clicks_a / views_a if views_a > 0 else 0
conv_a = purchases_a / clicks_a if clicks_a > 0 else 0

views_b = len(
feedback[
(feedback["group"] == "B")
& (feedback["event"] == "view")
]
)

clicks_b = len(
feedback[
(feedback["group"] == "B")
& (feedback["event"] == "click")
]
)

purchases_b = len(
feedback[
(feedback["group"] == "B")
& (feedback["event"] == "purchase")
]
)

ctr_b = clicks_b / views_b if views_b > 0 else 0
conv_b = purchases_b / clicks_b if clicks_b > 0 else 0

results = pd.DataFrame([
{
"group": "A",
"ctr": round(ctr_a, 4),
"conversion_rate": round(conv_a, 4)
},
{
"group": "B",
"ctr": round(ctr_b, 4),
"conversion_rate": round(conv_b, 4)
}
])

output_folder = "data/experiments"
os.makedirs(output_folder, exist_ok=True)

results.to_csv(
f"{output_folder}/ab_results.csv",
index=False
)

print("A/B testing completed")
