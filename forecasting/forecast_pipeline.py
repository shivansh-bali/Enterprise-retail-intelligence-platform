import os
import pandas as pd
import numpy as np
import torch

from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting import TemporalFusionTransformer




def load_data():


    data = pd.read_csv(
        "../data/processed/daily_sales.csv"
    )

    data["date"] = pd.to_datetime(data["date"])

  
    data.rename(
        columns={
            "StockCode": "product_id",
            "Quantity": "sales"
        },
        inplace=True
    )

    
    data = data.sort_values("date")

    data["time_idx"] = (
        data["date"] - data["date"].min()
    ).dt.days


    data["day_of_week"] = data["date"].dt.dayofweek
    data["month"] = data["date"].dt.month

    
    data["product_id"] = data["product_id"].astype(str)
    data["time_idx"] = data["time_idx"].astype(int)

    return data




def build_dataset(data):


    max_encoder_length = 30
    max_prediction_length = 7

    training_cutoff = (
        data["time_idx"].max()
        - max_prediction_length
    )

    training = TimeSeriesDataSet(
        data[data.time_idx <= training_cutoff],

        time_idx="time_idx",
        target="sales",
        group_ids=["product_id"],

        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,

        time_varying_known_reals=[
            "time_idx",
            "day_of_week",
            "month"
        ],

        time_varying_unknown_reals=["sales"],

        allow_missing_timesteps=True
    )

    validation = TimeSeriesDataSet.from_dataset(
        training,
        data,
        predict=True,
        stop_randomization=True
    )

    return training, validation



def load_model():

    model = TemporalFusionTransformer.load_from_checkpoint(
        "../forecasting/tft_model.ckpt"
    )

    print(" TFT model loaded")

    return model




def generate_forecasts(model, validation):


    val_loader = validation.to_dataloader(
        train=False,
        batch_size=64
    )

    raw_output = model.predict(
        val_loader,
        mode="prediction"
    )

    preds = raw_output.output.prediction

    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()

   
    index_df = validation.to_dataframe()

    forecast_list = []

    for i in range(len(preds)):

        product = index_df.iloc[i]["product_id"]
        start_time = index_df.iloc[i]["time_idx"]

        for step in range(preds.shape[1]):

            forecast_list.append({
                "product_id": product,
                "time_idx": start_time + step,
                "forecast_qty": float(preds[i, step])
            })

    forecast_df = pd.DataFrame(forecast_list)

    return forecast_df



def map_dates(forecast_df, data):


    date_map = data[
        ["time_idx", "date"]
    ].drop_duplicates()

    forecast_df = forecast_df.merge(
        date_map,
        on="time_idx",
        how="left"
    )

    forecast_df = forecast_df[
        ["product_id", "date", "forecast_qty"]
    ]

    return forecast_df



def save_forecast(df):

    output_dir = "../data/processed"
    os.makedirs(output_dir, exist_ok=True)

  
    existing_files = [
        f for f in os.listdir(output_dir)
        if f.startswith("forecast_tft_")
        and f.endswith(".csv")
    ]

   
    versions = []

    for file in existing_files:
        try:
            v = int(
                file.replace("forecast_tft_", "")
                    .replace(".csv", "")
            )
            versions.append(v)
        except:
            pass

    
    next_version = max(versions, default=0) + 1

    filename = f"forecast_tft_{next_version}.csv"

    filepath = os.path.join(output_dir, filename)

    df.to_csv(filepath, index=False)

    print(f" Forecast saved → {filename}")




def main():


    print("Running Forecast Pipeline...")

    data = load_data()

    training, validation = build_dataset(data)

    model = load_model()

    forecast_df = generate_forecasts(
        model,
        validation
    )

    forecast_df = map_dates(
        forecast_df,
        data
    )

    save_forecast(forecast_df)

    print("Forecast pipeline completed")



