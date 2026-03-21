import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import LabelEncoder
from scipy.sparse import coo_matrix
from sklearn.decomposition import TruncatedSVD

print("Running Collaborative Filtering...")



# LOAD DATA 



orders = pd.read_csv(
"data/raw/orders.csv",
usecols=["order_id", "user_id"]
)

order_products = pd.read_csv(
"data/raw/order_products__prior.csv",
usecols=["order_id", "product_id"]
)

transactions = order_products.merge(
orders,
on="order_id"
)





#  BUILD INTERACTIONS



user_product = (
transactions
.groupby(["user_id", "product_id"])
.size()
.reset_index(name="interaction")
)





# ENCODE IDS



user_enc = LabelEncoder()
prod_enc = LabelEncoder()

user_product["user_id_enc"] = user_enc.fit_transform(
user_product["user_id"].values
)

user_product["product_id_enc"] = prod_enc.fit_transform(
user_product["product_id"].values
)





# BUILD SPARSE MATRIX



matrix = coo_matrix(
(
user_product["interaction"],
(
user_product["user_id_enc"],
user_product["product_id_enc"]
)
)
)

print("Sparse matrix built")

# --------------------------------------------------

# 5️⃣ TRAIN CF MODEL

# --------------------------------------------------

svd = TruncatedSVD(n_components=50)

user_emb = svd.fit_transform(matrix)
product_emb = svd.components_.T

print("CF embeddings trained")

# --------------------------------------------------

# 6️⃣ GENERATE CANDIDATES

# --------------------------------------------------

user_id = 10

user_idx = user_enc.transform([user_id])[0]

scores = product_emb @ user_emb[user_idx]

top_k = 200

top_products_idx = np.argsort(scores)[-top_k:]

candidate_products = prod_enc.inverse_transform(
top_products_idx
)

candidates_df = pd.DataFrame({
"product_id": candidate_products,
"cf_score": scores[top_products_idx]
})



# SAVE CANDIDATES



os.makedirs("data/processed", exist_ok=True)

candidates_df.to_csv(
"data/processed/cf_candidates.csv",
index=False
)

print(" Candidates saved")

interactions_path = (
"data/processed/user_product_interactions.csv"
)

user_product[
["user_id", "product_id", "interaction"]
].to_csv(
interactions_path,
index=False
)

print(" Interactions saved")