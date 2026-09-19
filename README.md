# PJMW Hourly Energy Consumption Forecast

This project forecasts hourly electricity consumption using PJM historical energy data.

## Objectives

Analyze hourly, weekly, monthly, seasonal, and holiday demand patterns.

Use the final year as the test set.

Compare baseline, statistical, machine-learning, and deep-learning models.

Forecast electricity demand for the next 30 days.

Deploy the forecasting system with Streamlit.

## Models Compared

Previous Hour Baseline

Previous Day Baseline

Previous Week Baseline

ARIMA

Random Forest

XGBoost

LightGBM

LSTM

## Features

The ML models use calendar, lag, and rolling features such as:

Hour, day of week, month, weekend, holiday

lag_1, lag_24, lag_168

24-hour and 168-hour rolling statistics

## Evaluation

Models are evaluated using:

MAE

RMSE

MAPE

R²

Random Forest produced the strongest result in the current evaluation, while LSTM also performed well.

Streamlit App

The deployed dashboard includes:

# Project overview

EDA visualizations

Model comparison

Historical data explorer

30-day recursive forecast

## Run Locally

pip install -r requirements.txt
streamlit run app.py

# Project Structure

PJMW-Energy-Forecasting/
├── app.py
├── PJMW_hourly.csv
├── best_energy_forecast_model.pkl
├── model_comparison.csv
├── requirements.txt
├── README.md
└── .streamlit/
    └── config.toml
