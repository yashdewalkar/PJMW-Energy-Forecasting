# PJMW Hourly Energy Consumption Forecast

An end-to-end time-series forecasting project for predicting PJM West (PJMW) hourly electricity consumption using statistical, machine-learning, gradient-boosting, and deep-learning approaches.

The project covers exploratory data analysis (EDA), time-series feature engineering, baseline forecasting, model comparison, error analysis, 30-day forecasting, and Streamlit deployment.

### Project Objective

PJM Interconnection is a regional transmission organization (RTO) in the United States. The dataset contains historical hourly electricity consumption values measured in megawatts (MW).

### The main objectives of this project are to:

analyze hourly, daily, weekly, monthly, holiday, and long-term consumption patterns;

understand how electricity demand changes across seasons;

use the last year of observations as the test set;

build and compare several forecasting approaches;

evaluate models using the same forecasting metrics;

select the strongest forecasting model;

generate a 30-day future electricity-demand forecast;

deploy the project as an interactive Streamlit dashboard.

Dataset

The project uses the PJMW_hourly dataset.

Item

Value

Observations

143,206

Start date

2002-04-01 01:00

End date

2018-08-03 00:00

Target

PJMW_MW

Frequency

Hourly

Unit

Megawatts (MW)

The raw data was checked for missing values, duplicated timestamps, ordering issues, and irregular hourly timestamps. A small number of timestamp irregularities are associated with daylight-saving-time behavior, so the modeling pipeline creates a regular hourly series before generating lag-based features.

### Exploratory Data Analysis

EDA was used to understand the temporal structure of electricity demand before model building.

The analysis includes:

long-term electricity-consumption trend;

hourly consumption pattern;

day-of-week behavior;

weekday vs weekend demand;

monthly and seasonal demand patterns;

winter vs summer hourly profiles;

holiday behavior;

yearly trend analysis;

consumption distribution and outlier inspection;

time-series decomposition;

lag correlation analysis.

Main EDA observations

Electricity consumption shows strong hour-of-day seasonality.

Demand is generally lower during early-morning hours and higher during evening peak periods.

Weekdays generally show higher demand than weekends.

Strong seasonal differences are visible between winter, summer, spring, and fall.

Winter and summer demand are generally higher than shoulder-season demand.

Historical consumption is strongly correlated with recent hourly values, which supports using lag and rolling-window features.

### Feature Engineering

Calendar features

The following calendar variables are used to represent time-dependent behavior:

Hour
DayOfWeek
Month
Year
DayOfYear
Is_Weekend
Is_Holiday

Cyclical encodings are also used for hour, weekday, and month so that values near the beginning and end of a cycle remain mathematically close.

#### Lag features

Lag features provide the model with specific historical demand values.

lag_1     -> previous hour
lag_2     -> two hours ago
lag_3     -> three hours ago
lag_24    -> same hour yesterday
lag_48    -> same hour two days ago
lag_72    -> same hour three days ago
lag_168   -> same hour one week ago
lag_336   -> same hour two weeks ago

#### Rolling features

Rolling statistics summarize recent demand behavior:

rolling_mean_24
rolling_std_24
rolling_mean_168
rolling_std_168

Rolling statistics are calculated after shifting the target by one hour to avoid target leakage.

Example:

df["rolling_mean_24"] = (
    df["PJMW_MW"]
    .shift(1)
    .rolling(24)
    .mean()
)

Train-Test Strategy

A random train-test split is not appropriate for forecasting because it can introduce future information into model training.

The project therefore uses a chronological split:

Training data : historical observations before 2017-08-03
Test data     : 2017-08-03 to 2018-08-03

The final one-year period contains 8,761 hourly observations because both boundary timestamps are included.

Baseline Forecasts

Three simple baseline forecasts are used before advanced modeling.

Previous-hour baseline

Prediction = lag_1

The current hour is predicted using the previous hour's electricity consumption.

Previous-day baseline

Prediction = lag_24

The current hour is predicted using the same hour from the previous day.

Previous-week baseline

Prediction = lag_168

The current hour is predicted using the same hour from the previous week.

These baselines are important because an advanced model should outperform simple historical rules before it can be considered useful.

### Models Implemented

The project compares multiple forecasting approaches.

Statistical model

ARIMA

An experimental SARIMA configuration was also tested, but the long-horizon configuration became unstable and was not used as a final candidate model.

Machine-learning models

Random Forest Regressor

XGBoost Regressor

LightGBM Regressor

Deep-learning model

LSTM (Long Short-Term Memory)

The LSTM uses the previous 168 hourly observations (7 days) as the input sequence for predicting the next hour.

### Evaluation Metrics

Every valid candidate model is evaluated using:

MAE — Mean Absolute Error

Average absolute difference between actual and predicted electricity consumption.

RMSE — Root Mean Squared Error

Similar to MAE, but gives a larger penalty to large forecast errors.

MAPE — Mean Absolute Percentage Error

Average percentage forecasting error.

R² — Coefficient of Determination

Measures how much of the variation in electricity demand is explained by the model.

For MAE, RMSE, and MAPE, lower is better. For R², higher is better.

### Model Comparison

The current experiment produced the following results on the chronological test period:

Model

Type

MAE (MW)

RMSE (MW)

MAPE (%)

R²

Random Forest

Machine Learning

53.89

70.99

0.946

0.99489

LightGBM

Gradient Boosting

55.06

73.41

0.957

0.99454

XGBoost

Gradient Boosting

55.25

76.53

0.955

0.99407

LSTM

Deep Learning

77.90

100.96

1.387

0.98967

Previous Hour

Baseline

163.87

207.61

2.933

0.95634

Previous Day

Baseline

382.63

501.35

6.663

0.74538

Previous Week

Baseline

612.08

817.53

10.505

0.32294

ARIMA

Statistical

896.09

1085.14

17.142

-0.19275

### Current leading model

Random Forest achieved the strongest overall result in the current one-step-ahead evaluation, with the lowest MAE, RMSE, and MAPE and the highest R² among the tested models.

Compared with the previous-hour baseline, Random Forest reduces RMSE substantially, demonstrating that the engineered temporal features add predictive value beyond simply copying the previous observation.

### 30-Day Forecast

The deployed application generates a 30-day / 720-hour forecast.

30 days × 24 hours = 720 hourly predictions

For recursive forecasting:

the first future hour uses historical observed values;

the predicted value is added to the forecasting history;

lag and rolling features are recalculated;

the next hour is predicted;

the process repeats until 720 hourly predictions are generated.

This approach allows the trained model to produce forecasts beyond the final observed timestamp.

### Limitations and Future Improvements

Possible improvements include:

incorporating temperature and weather variables;

adding richer holiday and event information;

multivariate LSTM/GRU modeling;

direct multi-horizon forecasting instead of recursive prediction;

walk-forward cross-validation;

more systematic hyperparameter optimization;

prediction intervals / uncertainty estimation;

model monitoring after deployment;

comparing additional architectures such as GRU, Temporal Convolutional Networks, or transformer-based forecasting models.

Weather data would be particularly valuable because electricity consumption is strongly influenced by heating and cooling demand.

### Conclusion

The project demonstrates a complete electricity-demand forecasting workflow, beginning with exploratory analysis and ending with an interactive deployed application.

The experiment shows that strong feature engineering combined with ensemble tree models can perform extremely well on structured hourly electricity-demand data. In the current evaluation, Random Forest provides the strongest overall test performance, while LSTM also performs strongly and substantially outperforms the naive baselines.

The final deployment extends the modeling work by producing a recursive 30-day hourly forecast that can be explored through a Streamlit dashboard.
## Local run

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

macOS/Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```


## Important evaluation note

Random Forest, XGBoost, LightGBM, LSTM, and lag baselines were evaluated in a rolling one-step setting using observed historical information. The ARIMA result came from a long multi-step forecast, so its evaluation protocol is not perfectly identical.

The 30-day dashboard forecast is recursive. After the last observed timestamp, lag and rolling features are generated from prior predictions, so long-horizon error may be larger than one-step holdout metrics.
