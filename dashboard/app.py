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
metrics_folder = "data/metrics"
exp_folder = "data/experiments"
config_folder = "config"

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

metrics_path = f"{metrics_folder}/feedback_summary.csv"

if os.path.exists(metrics_path):


    st.header("Feedback Performance Metrics")

    metrics = pd.read_csv(metrics_path)

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "CTR",
        metrics["ctr"][0]
    )

    col2.metric(
        "Conversion Rate",
        metrics["conversion_rate"][0]
    )

    col3.metric(
        "Total Purchases",
        metrics["total_purchases"][0]
    )


exp_path = f"{exp_folder}/ab_results.csv"

if os.path.exists(exp_path):


    st.header("A/B Experiment Results")

    ab = pd.read_csv(exp_path)

    st.dataframe(ab)

    st.bar_chart(
        ab.set_index("group")[
            "conversion_rate"
        ]
    )


weights_path = f"{config_folder}/ranking_weights.csv"

if os.path.exists(weights_path):


    st.header("Ranking Weight Configuration")

    weights = pd.read_csv(weights_path)

    st.dataframe(weights)


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

st.scatter_chart(
fusion,
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
        merged["gap"] = (
            merged["final_score"]
            - merged["forecast_qty"]
        )

        result =( merged.sort_values(
            "gap",
            ascending=False
        )
        .head(10)
        .reset_index(drop=True)
        )

        st.dataframe(result)

