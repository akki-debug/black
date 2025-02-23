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

# Nifty 50 tickers
nifty_50_tickers = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "HINDUNILVR.NS", "SBIN.NS", "HDFC.NS", "BHARTIARTL.NS", "ITC.NS",
    "KOTAKBANK.NS", "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "BAJFINANCE.NS", "MARUTI.NS", "SUNPHARMA.NS", "HCLTECH.NS", "TITAN.NS", "ULTRACEMCO.NS",
    "WIPRO.NS", "NESTLEIND.NS", "INDUSINDBK.NS", "POWERGRID.NS", "TATASTEEL.NS", "TECHM.NS", "ADANIENT.NS", "ONGC.NS", "JSWSTEEL.NS", "CIPLA.NS",
    "NTPC.NS", "BAJAJFINSV.NS", "DRREDDY.NS", "TATAMOTORS.NS", "HEROMOTOCO.NS", "GRASIM.NS", "BPCL.NS", "COALINDIA.NS", "DIVISLAB.NS", "HINDALCO.NS",
    "EICHERMOT.NS", "BAJAJ-AUTO.NS", "APOLLOHOSP.NS", "BRITANNIA.NS", "SBILIFE.NS", "UPL.NS", "ADANIPORTS.NS", "M&M.NS", "IOC.NS", "SHREECEM.NS"
]

# Function to fetch stock data
def get_asset_data(tickers, start_date, end_date):
    temp_data = yf.download(tickers, start=start_date, end=end_date)["Close"].dropna()
    temp_data = temp_data.reset_index()
    temp_data['Date'] = pd.to_datetime(temp_data['Date']).dt.date
    temp_data = temp_data.set_index('Date')
    return temp_data

# Currency conversion function
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

# User input for tickers
symbols_option = st.radio("Choose stock selection method:", ("Manual Entry", "Nifty 50"))

if symbols_option == "Manual Entry":
    symbols_input = st.text_input(
        "Enter the tickers separated by commas:",
        placeholder="Example: AAPL, MSFT, GOOGL, TSLA"
    ).upper()
    symbols = [ticker.strip() for ticker in symbols_input.split(",") if ticker.strip()]
else:
    symbols = nifty_50_tickers
    st.success("Nifty 50 tickers selected!")

# Date selection
st.session_state.start_date = st.date_input("Enter the start date:", value=dt.date(2010, 1, 1))
st.session_state.end_date = st.date_input("Enter the end date:")

# Currency selection
currency_options = ["USD", "MXN", "EUR", "JPY", "GBP", "CAD", "AUD"]
st.session_state.target_currency = st.selectbox("Select the target currency:", currency_options)

# Fetching data
if st.button("Get data!"):
    if not symbols:
        st.warning("No tickers detected. Please provide tickers or select Nifty 50.")
    elif st.session_state.start_date >= st.session_state.end_date:
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
