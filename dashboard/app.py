import streamlit as st
import pandas as pd
import os
import re
import numpy as np

st.set_page_config(
page_title="Enterprise Retail AI Dashboard",
layout="wide"
)

st.title("Enterprise Recommendation + Forecast Intelligence")


data_folder = "data/processed"


forecast_files = [
f for f in os.listdir(data_folder)
if f.startswith("forecast_tft_")
]


forecast_versions = [
int(f.split("_")[2].split(".")[0])
for f in forecast_files
]

latest_forecast = f"forecast_tft_{max(forecast_versions)}.csv"

forecast = pd.read_csv(
f"{data_folder}/{latest_forecast}"
)

rec_files = [
f for f in os.listdir(data_folder)
if f.startswith("recommendations_")
]

rec_versions = [
int(f.split("_")[1].split(".")[0])
for f in rec_files
]

latest_rec = f"recommendations_{max(rec_versions)}.csv"

recommendations = pd.read_csv(
f"{data_folder}/{latest_rec}"
)

st.header("Demand Forecast Intelligence")

demand = (
forecast.groupby("product_id")[
"forecast_qty"
]
.mean()
.sort_values(ascending=False)
.head(10)
)

st.bar_chart(demand)

st.dataframe(demand)

st.header("Top Recommended Products")

top_recs = recommendations.head(10)

st.dataframe(top_recs)

st.bar_chart(
top_recs.set_index("product_id")[
"final_score"
]
)



import requests

if "viewed_products" not in st.session_state:
    st.session_state.viewed_products = set()
    
API_URL = "http://127.0.0.1:8000"

def get_recommendations(user_id):
    try:
        res = requests.get(
            f"{API_URL}/recommend",
            params={"user_id": user_id}
        )
        return res.json()
    except:
        return []

def log_event(user_id, product_id, event):
    payload = {
        "user_id": user_id,
        "product_id": product_id,
        "event": event
    }

    try:
        requests.post(f"{API_URL}/feedback", json=payload)
    except:
        pass

st.header("User Recommendation Interface")

user_list = recommendations["user_id"].unique() if "user_id" in recommendations.columns else [10]

selected_user = st.selectbox(
    "Select User",
    user_list
)

if st.button("Get Personalized Recommendations"):

    recs = get_recommendations(selected_user)

    if len(recs) == 0:
        st.warning("No recommendations found")
    else:
        st.subheader(f"Top Recommendations for User {selected_user}")

        for i, product in enumerate(recs):

            product_id = product["product_id"]

            
            if product_id not in st.session_state.viewed_products:
                log_event(selected_user, product_id, "view")
                st.session_state.viewed_products.add(product_id)

            col1, col2, col3 = st.columns([3, 1, 1])

            with col1:
                st.write(f"Product: {product_id}")
                st.write(f"Score: {round(product['final_score'], 4)}")

            with col2:
                if st.button(
                    "Click",
                    key=f"click_{selected_user}_{product_id}_{i}"
                ):
                    log_event(selected_user, product_id, "click")
                    st.success("Clicked")

            with col3:
                if st.button(
                    "Buy",
                    key=f"buy_{selected_user}_{product_id}_{i}"
                ):
                    log_event(selected_user, product_id, "purchase")
                    st.success("Purchased")

            st.markdown("---")    







st.header("Forecast vs Recommendation Fusion")

recommendations["product_id"] = (
recommendations["product_id"]
.astype(str)
)

forecast["product_id"] = (
forecast["product_id"]
.astype(str)
)

fusion = recommendations.merge(
forecast,
on="product_id",
how="left"
)
fusion.rename(
columns={
"forecast_qty_y": "forecast_qty"
},
inplace=True
)



fusion["forecast_qty"] = fusion["forecast_qty"].fillna(0)


fusion["forecast_log"] = np.log1p(
fusion["forecast_qty"]
)

fusion_sample = fusion.sample(n=2000, random_state=42)

st.scatter_chart(
    fusion_sample,
    x="forecast_log",
    y="final_score"
)


st.header("LLM Business Copilot")

questions = [
"Top demand products next week",
"Restock alerts",
"High affinity + high demand",
"Over-recommended products"
]

selected_question = st.selectbox(
"Select Business Question",
questions
)

forecast["product_id"] = forecast["product_id"].astype(str)
recommendations["product_id"] = recommendations["product_id"].astype(str)

if st.button("Generate Insight"):

    if selected_question == "Top demand products next week":

        result = (
            forecast.groupby("product_id")[
                "forecast_qty"
            ]
            .mean()
            .sort_values(ascending=False)
            .head(10)
        )

        st.dataframe(result)

    elif selected_question == "Restock alerts":

        result = (
            forecast.groupby("product_id")[
                "forecast_qty"
            ]
            .mean()
            .sort_values(ascending=False)
            .head(10)
        )

        st.dataframe(result)

    elif selected_question == "High affinity + high demand":
       
        forecast_agg = (
        forecast.groupby("product_id")[
        "forecast_qty"
        ]
        .mean()
        .reset_index()
        )


        merged = recommendations.merge(
            forecast_agg,
            on="product_id",
            how="left"
        )
        merged.rename(
        columns={
        "forecast_qty_y": "forecast_qty"
        },
        inplace=True
        )

        merged["score"] = (
            merged["final_score"]
            * merged["forecast_qty"]
        )

        result =( merged.sort_values(
            "score",
            ascending=False
        )
        .head(10)
        .reset_index(drop=True)
        )
        st.dataframe(result)

    elif selected_question == "Over-recommended products":
        forecast_agg = (
        forecast.groupby("product_id")[
        "forecast_qty"
        ]
        .mean()
        .reset_index()
        )

        merged = recommendations.merge(
            forecast_agg,
            on="product_id",
            how="left"
        )
        merged.rename(
        columns={
        "forecast_qty_y": "forecast_qty"
        },
        inplace=True
        )
        merged["forecast_norm"] = (
        merged["forecast_qty"] - merged["forecast_qty"].min()
        ) / (
        merged["forecast_qty"].max() - merged["forecast_qty"].min()
        )

        merged["gap"] = (
        merged["final_score"] - merged["forecast_norm"]
        )

        result =( merged.sort_values(
            "gap",
            ascending=False
        )
        .head(10)
        .reset_index(drop=True)
        )

        st.dataframe(result)

