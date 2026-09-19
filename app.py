from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pandas.tseries.holiday import USFederalHolidayCalendar

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / 'PJMW_hourly.csv'
MODEL_PATH = BASE_DIR / 'best_energy_forecast_model.pkl'
COMPARISON_PATH = BASE_DIR / 'model_comparison.csv'
META_PATH = BASE_DIR / 'metadata.json'

st.set_page_config(
    page_title='PJMW Energy Forecasting',
    page_icon='⚡',
    layout='wide',
    initial_sidebar_state='expanded',
)

# -----------------------------
# Data/model loading
# -----------------------------
@st.cache_data
def load_raw_data():
    df = pd.read_csv(DATA_PATH, parse_dates=['Datetime'])
    return df.sort_values('Datetime').reset_index(drop=True)

@st.cache_data
def load_regular_series():
    df = load_raw_data()[['Datetime', 'PJMW_MW']].copy()
    df = (
        df.groupby('Datetime', as_index=False)['PJMW_MW']
          .mean()
          .sort_values('Datetime')
    )
    full_index = pd.date_range(df['Datetime'].min(), df['Datetime'].max(), freq='h')
    df = (
        df.set_index('Datetime')
          .reindex(full_index)
          .rename_axis('Datetime')
          .reset_index()
    )
    df['PJMW_MW'] = (
        df['PJMW_MW']
        .fillna(df['PJMW_MW'].shift(24))
        .fillna(df['PJMW_MW'].shift(168))
        .ffill()
    )
    return df

@st.cache_resource
def load_model_bundle():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_comparison():
    return pd.read_csv(COMPARISON_PATH)

@st.cache_data
def load_meta():
    if META_PATH.exists():
        return json.loads(META_PATH.read_text(encoding='utf-8'))
    return {}


def make_future_feature_row(ts, history, holiday_dates):
    """Create one recursive future feature row using only known/predicted history."""
    arr = np.asarray(history, dtype=float)
    hour = ts.hour
    dow = ts.dayofweek
    month = ts.month

    return {
        'Hour': hour,
        'DayOfWeek': dow,
        'Month': month,
        'Year': ts.year,
        'DayOfYear': ts.dayofyear,
        'Is_Weekend': int(dow >= 5),
        'Is_Holiday': int(ts.normalize() in holiday_dates),
        'hour_sin': np.sin(2 * np.pi * hour / 24),
        'hour_cos': np.cos(2 * np.pi * hour / 24),
        'dow_sin': np.sin(2 * np.pi * dow / 7),
        'dow_cos': np.cos(2 * np.pi * dow / 7),
        'month_sin': np.sin(2 * np.pi * month / 12),
        'month_cos': np.cos(2 * np.pi * month / 12),
        'lag_1': arr[-1],
        'lag_2': arr[-2],
        'lag_3': arr[-3],
        'lag_24': arr[-24],
        'lag_48': arr[-48],
        'lag_72': arr[-72],
        'lag_168': arr[-168],
        'lag_336': arr[-336],
        'rolling_mean_24': arr[-24:].mean(),
        'rolling_std_24': arr[-24:].std(ddof=1),
        'rolling_mean_168': arr[-168:].mean(),
        'rolling_std_168': arr[-168:].std(ddof=1),
    }


@st.cache_data(show_spinner=False)
def recursive_forecast(days):
    bundle = load_model_bundle()
    model = bundle['model']
    features = bundle['features']
    series = load_regular_series()

    history = series['PJMW_MW'].astype(float).tolist()
    last_ts = series['Datetime'].max()
    periods = int(days * 24)
    future_index = pd.date_range(last_ts + pd.Timedelta(hours=1), periods=periods, freq='h')
    holiday_dates = set(
        USFederalHolidayCalendar()
        .holidays(start=future_index.min().normalize(), end=future_index.max().normalize())
        .normalize()
    )

    preds = []
    for ts in future_index:
        row = make_future_feature_row(ts, history, holiday_dates)
        X = pd.DataFrame([row], columns=features)
        pred = float(model.predict(X)[0])
        preds.append(pred)
        history.append(pred)

    return pd.DataFrame({'Datetime': future_index, 'Forecast_MW': preds})


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title('⚡ PJMW Forecasting')
st.sidebar.caption('Hourly Energy Consumption Forecast')
page = st.sidebar.radio(
    'Navigate',
    ['Project Overview', 'EDA Dashboard', 'Model Comparison', '30-Day Forecast', 'Data Explorer']
)
st.sidebar.divider()
st.sidebar.caption('P-702 · PJM Hourly Energy Consumption')

raw = load_raw_data()
regular = load_regular_series()
comparison = load_comparison()
bundle = load_model_bundle()
meta = load_meta()

# -----------------------------
# Overview
# -----------------------------
if page == 'Project Overview':
    st.title('⚡ PJMW Hourly Energy Consumption Forecast')
    st.write(
        'An end-to-end time-series forecasting project using PJM hourly electricity demand, '
        'from exploratory analysis and baseline forecasting through machine learning, deep learning, '
        'model evaluation, and a recursive future forecast.'
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Observations', f'{len(raw):,}')
    c2.metric('Start', raw['Datetime'].min().strftime('%d %b %Y'))
    c3.metric('End', raw['Datetime'].max().strftime('%d %b %Y'))
    c4.metric('Average Load', f"{raw['PJMW_MW'].mean():,.0f} MW")

    st.subheader('Project workflow')
    st.markdown(
        '**EDA → preprocessing → lag & rolling features → baselines → '
        'Random Forest / XGBoost / LightGBM / ARIMA / LSTM → evaluation → 30-day forecast**'
    )

    st.subheader('Deployment model')
    m = bundle['evaluation']
    d1, d2, d3, d4 = st.columns(4)
    d1.metric('MAE', f"{m['MAE']:.2f} MW")
    d2.metric('RMSE', f"{m['RMSE']:.2f} MW")
    d3.metric('MAPE', f"{m['MAPE (%)']:.3f}%")
    d4.metric('R²', f"{m['R2']:.5f}")
    st.caption(
        'The deployed Random Forest is a compact hosting-optimized version (60 trees, max depth 16). '
        'The Model Comparison page shows the final notebook results for the full project comparison.'
    )

    st.info(
        'The future forecast is recursive: after the last observed hour, lag and rolling features '
        'are progressively built from earlier model predictions. Multi-step error can therefore '
        'be larger than one-step holdout error.'
    )

# -----------------------------
# EDA
# -----------------------------
elif page == 'EDA Dashboard':
    st.title('Exploratory Data Analysis')

    tab1, tab2, tab3, tab4 = st.tabs(['Long-term Trend', 'Hourly Pattern', 'Weekly Pattern', 'Monthly Pattern'])

    with tab1:
        daily = regular.set_index('Datetime')['PJMW_MW'].resample('D').mean().reset_index()
        fig = px.line(daily, x='Datetime', y='PJMW_MW', title='Daily Average PJMW Load')
        fig.update_layout(yaxis_title='MW', xaxis_title='Date')
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        temp = raw.assign(Hour=raw['Datetime'].dt.hour)
        hourly = temp.groupby('Hour', as_index=False)['PJMW_MW'].mean()
        fig = px.line(hourly, x='Hour', y='PJMW_MW', markers=True, title='Average Load by Hour of Day')
        fig.update_layout(yaxis_title='Average MW')
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        temp = raw.assign(
            DayOfWeek=raw['Datetime'].dt.dayofweek,
            DayName=raw['Datetime'].dt.day_name()
        )
        weekly = temp.groupby(['DayOfWeek','DayName'], as_index=False)['PJMW_MW'].mean().sort_values('DayOfWeek')
        fig = px.bar(weekly, x='DayName', y='PJMW_MW', title='Average Load by Day of Week')
        fig.update_layout(yaxis_title='Average MW', xaxis_title='Day')
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        temp = raw.assign(Month=raw['Datetime'].dt.month)
        monthly = temp.groupby('Month', as_index=False)['PJMW_MW'].mean()
        month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
        monthly['MonthName'] = monthly['Month'].apply(lambda x: month_names[x-1])
        fig = px.bar(monthly, x='MonthName', y='PJMW_MW', title='Average Load by Month')
        fig.update_layout(yaxis_title='Average MW', xaxis_title='Month')
        st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Model comparison
# -----------------------------
elif page == 'Model Comparison':
    st.title('Model Comparison')
    st.write('All reported models use MAE, RMSE, MAPE and R². Lower error metrics are better; higher R² is better.')

    display_cols = ['Model','Type','MAE','RMSE','MAPE (%)','R2','RMSE Improvement vs Prev Hour (%)','Evaluation Protocol']
    st.dataframe(
        comparison[display_cols].sort_values('RMSE').style.format({
            'MAE':'{:.2f}', 'RMSE':'{:.2f}', 'MAPE (%)':'{:.3f}', 'R2':'{:.5f}',
            'RMSE Improvement vs Prev Hour (%)':'{:.2f}'
        }),
        use_container_width=True,
        hide_index=True
    )

    metric = st.selectbox('Bar plot metric', ['RMSE','MAE','MAPE (%)','R2'])
    ascending = metric != 'R2'
    plot_df = comparison.sort_values(metric, ascending=ascending)
    fig = px.bar(
        plot_df,
        x=metric,
        y='Model',
        orientation='h',
        text=metric,
        title=f'Model Comparison by {metric}'
    )
    fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    fig.update_layout(yaxis={'categoryorder':'array', 'categoryarray':plot_df['Model'].tolist()[::-1]})
    st.plotly_chart(fig, use_container_width=True)

    best = comparison.sort_values('RMSE').iloc[0]
    b1, b2, b3, b4 = st.columns(4)
    b1.metric('Lowest RMSE Model', best['Model'])
    b2.metric('RMSE', f"{best['RMSE']:.2f} MW")
    b3.metric('MAPE', f"{best['MAPE (%)']:.3f}%")
    b4.metric('R²', f"{best['R2']:.5f}")

    st.warning(
        'ARIMA was evaluated as a long multi-step forecast, while the lag-based baselines, tree models, '
        'and LSTM were evaluated in rolling one-step form. Its result is useful as a statistical benchmark, '
        'but the evaluation protocol is not perfectly identical.'
    )

# -----------------------------
# 30-day forecast
# -----------------------------
elif page == '30-Day Forecast':
    st.title('Recursive Future Forecast')
    st.write(
        'The forecast starts immediately after the final observed timestamp. '
        'Future lag and rolling features are updated recursively using prior predictions.'
    )

    days = st.slider('Forecast horizon (days)', min_value=1, max_value=30, value=30, step=1)

    with st.spinner(f'Generating {days}-day recursive forecast...'):
        forecast = recursive_forecast(days)

    last_history = regular.tail(24 * 7)[['Datetime','PJMW_MW']].rename(columns={'PJMW_MW':'Value'})
    last_history['Series'] = 'Historical'
    future_plot = forecast.rename(columns={'Forecast_MW':'Value'}).copy()
    future_plot['Series'] = 'Forecast'
    plot = pd.concat([last_history, future_plot], ignore_index=True)

    fig = px.line(
        plot,
        x='Datetime',
        y='Value',
        color='Series',
        title=f'Historical Load + {days}-Day Recursive Forecast'
    )
    fig.update_layout(yaxis_title='MW', xaxis_title='Datetime')
    st.plotly_chart(fig, use_container_width=True)

    f1, f2, f3, f4 = st.columns(4)
    f1.metric('Forecast Hours', f'{len(forecast):,}')
    f2.metric('Average Forecast', f"{forecast['Forecast_MW'].mean():,.0f} MW")
    f3.metric('Peak Forecast', f"{forecast['Forecast_MW'].max():,.0f} MW")
    f4.metric('Minimum Forecast', f"{forecast['Forecast_MW'].min():,.0f} MW")

    st.subheader('Daily forecast summary')
    daily_f = (
        forecast.set_index('Datetime')['Forecast_MW']
        .resample('D')
        .agg(['mean','min','max'])
        .reset_index()
        .rename(columns={'mean':'Average_MW','min':'Minimum_MW','max':'Maximum_MW'})
    )
    st.dataframe(daily_f.round(2), use_container_width=True, hide_index=True)

    st.download_button(
        'Download forecast CSV',
        data=forecast.to_csv(index=False).encode('utf-8'),
        file_name=f'pjmw_{days}_day_forecast.csv',
        mime='text/csv'
    )

    st.info(
        'Interpretation note: this multi-step recursive forecast is harder than one-step test evaluation. '
        'Predictions further into the horizon increasingly depend on earlier predictions.'
    )

# -----------------------------
# Data explorer
# -----------------------------
else:
    st.title('Historical Data Explorer')
    min_date = raw['Datetime'].min().date()
    max_date = raw['Datetime'].max().date()

    c1, c2 = st.columns(2)
    start_date = c1.date_input('Start date', value=max(min_date, pd.Timestamp('2018-01-01').date()), min_value=min_date, max_value=max_date)
    end_date = c2.date_input('End date', value=max_date, min_value=min_date, max_value=max_date)

    if start_date > end_date:
        st.error('Start date must be before end date.')
    else:
        mask = (
            (raw['Datetime'].dt.date >= start_date) &
            (raw['Datetime'].dt.date <= end_date)
        )
        subset = raw.loc[mask].copy()
        fig = px.line(subset, x='Datetime', y='PJMW_MW', title='Selected Historical Load')
        fig.update_layout(yaxis_title='MW')
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(subset.tail(500), use_container_width=True, hide_index=True)
