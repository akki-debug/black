import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

# List of Nifty 50 stock tickers
nifty_tickers = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "HINDUNILVR.NS", "ICICIBANK.NS", "SBIN.NS", "BAJFINANCE.NS", "KOTAKBANK.NS", "LT.NS"]

# Function to fetch stock data from Yahoo Finance
def fetch_stock_data(tickers):
    data = yf.download(tickers, period="1y", interval="1d")
    returns = data["Adj Close"].pct_change().dropna()
    expected_returns = returns.mean()
    cov_matrix = returns.cov()
    return expected_returns, cov_matrix

# Function to perform stress testing
def stress_test(portfolio, shocks):
    stressed_portfolio = portfolio.copy()
    stressed_portfolio["Expected Return"] += shocks
    return stressed_portfolio

# Streamlit UI
st.title("Stress Testing – Sensitivity Analysis on Nifty 50 Portfolio")

st.sidebar.header("Portfolio Inputs")
num_assets = st.sidebar.number_input("Number of Assets", min_value=2, max_value=len(nifty_tickers), value=4, step=1)

# User input for selecting Nifty stocks
tickers = st.sidebar.multiselect("Select Nifty 50 Stocks", nifty_tickers, nifty_tickers[:num_assets])

# Fetch stock data
expected_returns, cov_matrix = fetch_stock_data(tickers)

# User input for shocks
st.sidebar.subheader("Shocks to Expected Returns (%)")
shocks = np.array([st.sidebar.number_input(f"Shock for {tickers[i]}", value=0.0) for i in range(len(tickers))]) / 100

# Portfolio DataFrame
portfolio = pd.DataFrame({
    "Asset": tickers,
    "Expected Return": expected_returns.values,
})

# Perform Stress Testing
stressed_portfolio = stress_test(portfolio, shocks)

# Display results
st.subheader("Original Portfolio")
st.dataframe(portfolio)

st.subheader("Stressed Portfolio")
st.dataframe(stressed_portfolio)

# Plot the impact of shocks
fig, ax = plt.subplots()
ax.bar(portfolio["Asset"], portfolio["Expected Return"], label="Original", alpha=0.6)
ax.bar(stressed_portfolio["Asset"], stressed_portfolio["Expected Return"], label="Stressed", alpha=0.6)
ax.set_ylabel("Expected Return")
ax.set_title("Impact of Shocks on Expected Returns")
ax.legend()
st.pyplot(fig)
