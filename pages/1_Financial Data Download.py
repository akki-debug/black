import pandas as pd
import numpy as np
from numpy.linalg import multi_dot
import yfinance as yf
import scipy as stats
from scipy.stats import kurtosis, skew, norm
import matplotlib.pyplot as plt
import datetime as dt
import seaborn as sns
import streamlit as st
from fredapi import Fred
import requests
import plotly.graph_objects as go

sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

st.set_page_config(
    page_title="Financial Data Download",
    page_icon="mag"
)
st.title("Financial Data Download")
st.markdown("## Data Download & Currency Conversion :currency_exchange:")

st.markdown("The first step in this app is downloading your data. Enter the tickers to analyze, ensuring they are spelled correctly and separated by commas.")

def get_asset_data(tickers, start_date, end_date):
    temp_data = yf.download(tickers, start=start_date, end=end_date)["Close"].dropna()
    temp_data = temp_data.reset_index()
    temp_data['Date'] = pd.to_datetime(temp_data['Date']).dt.date
    temp_data = temp_data.set_index('Date')
    return temp_data

def convert_to_currency(data, start_date, end_date, target_currency="MXN"):
    conversion_rates = {}
    for ticker in data.columns:
        symbol = yf.Ticker(ticker)
        currency = symbol.info.get("currency", "USD")
        if currency != target_currency:
            fx_pair = f"{currency}{target_currency}=X"
            if fx_pair not in conversion_rates:
                fx_data = pd.DataFrame(get_asset_data(fx_pair, start_date, end_date)).rename(
                    columns={"Close": fx_pair})
                conversion_rates[fx_pair] = fx_data
            data[ticker] = data[ticker] * conversion_rates[fx_pair][fx_pair]
            data = data.ffill()
    return data

st.session_state.get_asset_data = get_asset_data
st.session_state.convert_to_currency = convert_to_currency

symbols_input = st.text_input(
    "Enter the tickers separated by commas:",
    placeholder="Example: AAPL, MSFT, GOOGL, TSLA"
).upper()

st.session_state.start_date = st.date_input("Enter the start date:", value=dt.date(2010, 1, 1))
st.session_state.end_date = st.date_input("Enter the end date:")

if "data" not in st.session_state:
    st.session_state.data = None
if "target_currency" not in st.session_state:
    st.session_state.target_currency = "MXN"

currency_options = ["USD", "MXN", "EUR", "JPY", "GBP", "CAD", "AUD"]
st.session_state.target_currency = st.selectbox("Select the target currency:", currency_options)

if st.button("Get data!"):
    if not symbols_input.strip():
        st.warning("No tickers detected. Please provide tickers separated by commas.")
    else:
        symbols = [ticker.strip() for ticker in symbols_input.split(",") if ticker.strip()]
        if st.session_state.start_date >= st.session_state.end_date:
            st.error("Invalid date range. Please ensure the start date is earlier than the end date.")
        else:
            try:
                st.session_state.data = get_asset_data(
                    symbols,
                    start_date=st.session_state.start_date,
                    end_date=st.session_state.end_date
                )
                if len(st.session_state.data) == 0:
                    st.warning("No data found for the provided tickers.")
                else:
                    st.session_state.data = convert_to_currency(
                        st.session_state.data,
                        start_date=st.session_state.start_date,
                        end_date=st.session_state.end_date,
                        target_currency=st.session_state.target_currency
                    )
                    st.success(f"Data downloaded and converted to {st.session_state.target_currency}.")
            except Exception as e:
                st.error(f"An error occurred during processing: {e}")

if st.session_state.data is not None and len(st.session_state.data) > 0:
    st.dataframe(st.session_state.data.tail())

st.markdown("*Note: Yahoo Finance data downloads may occasionally be unavailable. If some data is missing, please try again later.*")

if st.session_state.data is not None and len(st.session_state.data) > 0:
    st.markdown("## Closing Price Display :chart_with_upwards_trend:")
    colors = sns.color_palette("mako", n_colors=len(st.session_state.data.columns)).as_hex()
    st.session_state.colors = colors
    fig = go.Figure()
    for idx, column in enumerate(st.session_state.data.columns):
        fig.add_trace(
            go.Scatter(
                x=st.session_state.data.index,
                y=st.session_state.data[column],
                mode='lines',
                name=column,
                line=dict(width=2, color=colors[idx]),
            )
        )
    fig.update_layout(
        title="",
        xaxis_title="Date",
        yaxis_title=f"Closing Price ({st.session_state.target_currency})",
        template="plotly_white",
        hovermode="x unified",
        width=800,
        height=600,
        legend=dict(
            title="",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
    )
    st.plotly_chart(fig, use_container_width=True)

if "returns" not in st.session_state:
    st.session_state.returns = None

if st.session_state.data is not None:
    st.markdown("## Daily Returns :heavy_dollar_sign:")
    returns = st.session_state.data.copy()
    returns = returns.sort_index()
    for columna in returns.columns:
        returns[columna] = (returns[columna] - returns[columna].shift(1)) / returns[columna].shift(1)
    returns = returns.dropna()
    st.session_state.returns = returns
    st.success("Successfully calculated returns.")

if st.session_state.returns is not None:
    st.dataframe(st.session_state.returns.tail())

def get_rf_rate_us(plazo, key, start_date = dt.date(2010, 1, 1), end_date = dt.date.today(), today = False):
    fred = Fred(api_key=key)
    rf_rate_us = fred.get_series(plazo, start_date, end_date)
    rf_rate_us = rf_rate_us / 100
    if today == False:
        rf_rate_us = pd.DataFrame(rf_rate_us).rename(columns ={0: "Rate"})
        rf_rate_us.index.name = "Date"
        return rf_rate_us
    else:
        return rf_rate_us[-1].iloc[0]

def get_rf_rate_mx(plazo, key, start_date = dt.date(2010, 1, 1), end_date = dt.date.today(), today = False):
    if today:
        url = f"https://www.banxico.org.mx/SieAPIRest/service/v1/series/{plazo}/datos/oportuno?token={key}"
        response = requests.get(url)
        data = response.json()
        return float(data["bmx"]["series"][0]["datos"][0]["dato"]) / 100
    else:
        url = f"https://www.banxico.org.mx/SieAPIRest/service/v1/series/{plazo}/datos/{start_date}/{end_date}?token={key}"
        response = requests.get(url)
        data = response.json()
        dates = []
        values = []
        for entry in data["bmx"]["series"][0]["datos"]:
            dates.append(entry["fecha"])
            values.append(float(entry["dato"]) / 100)
        df = pd.DataFrame({"Date": dates, "Rate": values})
        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)
        return df

st.session_state.get_rf_rate_us = get_rf_rate_us
st.session_state.get_rf_rate_mx = get_rf_rate_mx

st.markdown("## Risk Free Rates :heavy_dollar_sign:")
st.text("To retrieve risk-free rates, this app uses the FRED and BANXICO APIs. These rates are essential for calculating key risk metrics. You can choose between Mexican and US rates and select different terms. We recommend using the 1-year risk-free rate.")

if "rf_rate_us" not in st.session_state:
    st.session_state.rf_rate_us = None
if "rf_rate_mx" not in st.session_state:
    st.session_state.rf_rate_mx = None

plazos = ["3m", "1y", "5y", "10y"]
plazo = st.selectbox("Select the period of the risk free rate:", plazos, index = 1)

if st.button("Get Rates!"):
    us_treasury = {"3m":"GS3M", "1y":"GS1", "5y":"GS5", "10y":"GS10"}
    key_fred = '3f2f344c22249ae2ed4577695e869bcd'
    rf_rate_us = get_rf_rate_us(plazo = us_treasury[plazo], key = key_fred)
    us_rf_rate_today = rf_rate_us.iloc[-1].iloc[0]
    mx_treasury = {"3m": "SF3338", "1y": "SF3367", "5y": "SF18608", "10y": "SF30001"}
    key_banxico = '8d5b5c9d7e8e4f05b2d49a6a44f56a35'
    rf_rate_mx = get_rf_rate_mx(plazo = mx_treasury[plazo], key = key_banxico)
    mx_rf_rate_today = rf_rate_mx.iloc[-1].iloc[0]
    st.success(f"Risk-free rate US: {round(us_rf_rate_today*100, 2)}%")
    st.success(f"Risk-free rate MX: {round(mx_rf_rate_today*100, 2)}%")
