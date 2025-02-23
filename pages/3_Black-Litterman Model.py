import pandas as pd
import numpy as np
import yfinance as yf
import scipy.optimize as sco
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from datetime import datetime

# Page Config
st.set_page_config(page_title="Black-Litterman Model", page_icon="mag")
st.title("Black-Litterman Model")

# Initialize session state variables
if "portafolios_bl" not in st.session_state:
    st.session_state.portafolios_bl = None

# Fetch stock data
def get_stock_data(tickers, start, end):
    data = yf.download(tickers, start=start, end=end)["Close"].dropna()
    return data

nifty50_stocks = [
    "RELIANCE.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", "TCS.NS",
    "HINDUNILVR.NS", "ITC.NS", "LT.NS", "AXISBANK.NS", "BHARTIARTL.NS"
]
selected_stocks = st.multiselect("Select stocks for portfolio:", nifty50_stocks, default=nifty50_stocks[:5])

start_date = st.date_input("Start Date", value=datetime(2023, 1, 1))
end_date = st.date_input("End Date", value=datetime(2023, 12, 31))

st.session_state.data = get_stock_data(selected_stocks, start_date, end_date)
st.session_state.returns = st.session_state.data.pct_change().dropna()

if st.session_state.data is not None and st.session_state.returns is not None:
    st.success("Stock data successfully fetched.")
    st.write("Closing prices:")
    st.dataframe(st.session_state.data.tail())
    st.write("Daily returns:")
    st.dataframe(st.session_state.returns.tail())
else:
    st.warning("No data available. Please select valid tickers and date range.")
    st.stop()

# Performance Statistics

def calculate_performance(returns):
    cumulative_returns = (1 + returns).cumprod()
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = drawdown.min()
    
    annual_return = returns.mean() * 252
    annual_volatility = returns.std() * np.sqrt(252)
    sharpe_ratio = annual_return / annual_volatility
    stability = returns.autocorr()
    omega_ratio = (returns[returns > 0].sum() / -returns[returns < 0].sum())
    sortino_ratio = annual_return / returns[returns < 0].std()
    skewness = returns.skew()
    kurtosis = returns.kurtosis()
    tail_ratio = returns.quantile(0.95) / abs(returns.quantile(0.05))
    daily_var = returns.quantile(0.05)
    
    return [
        f"{annual_return:.2%}", f"{annual_volatility:.2%}", f"{sharpe_ratio:.2f}", f"{max_drawdown:.2%}", 
        f"{stability:.2f}", f"{omega_ratio:.2f}", f"{sortino_ratio:.2f}", f"{skewness:.2f}", 
        f"{kurtosis:.2f}", f"{tail_ratio:.2f}", f"{daily_var:.2%}"
    ]

strategy_stats = calculate_performance(st.session_state.returns.mean(axis=1))
benchmark_ticker = "^NSEI"  # Nifty 50 Index
benchmark_data = get_stock_data([benchmark_ticker], start_date, end_date).pct_change().dropna()
benchmark_stats = calculate_performance(benchmark_data[benchmark_ticker])

stats_df = pd.DataFrame({
    "Metric": ["Annual Return", "Annual Volatility", "Sharpe Ratio", "Max Drawdown", "Stability",
               "Omega Ratio", "Sortino Ratio", "Skewness", "Kurtosis", "Tail Ratio", "Daily Value at Risk"],
    "Strategy": strategy_stats,
    "Benchmark": benchmark_stats
})

st.markdown("## Performance Statistics")
st.table(stats_df.set_index("Metric"))

# Additional Visualizations
st.markdown("## Data Visualization :bar_chart:")
fig1, ax1 = plt.subplots(figsize=(12, 6))
sns.lineplot(data=st.session_state.data, ax=ax1)
ax1.set_title("Stock Closing Prices Over Time")
st.pyplot(fig1)
plt.close(fig1)

fig2, ax2 = plt.subplots(figsize=(10, 6))
sns.heatmap(st.session_state.returns.corr(), annot=True, cmap='coolwarm', linewidths=0.5, ax=ax2)
ax2.set_title("Stock Correlation Heatmap")
st.pyplot(fig2)
plt.close(fig2)

fig3, ax3 = plt.subplots(figsize=(12, 6))
sns.histplot(st.session_state.returns, bins=30, kde=True, ax=ax3)
ax3.set_title("Histogram of Daily Returns")
st.pyplot(fig3)
plt.close(fig3)

fig4, ax4 = plt.subplots(figsize=(12, 6))
st.session_state.returns.cumsum().plot(ax=ax4)
ax4.set_title("Cumulative Returns of Selected Stocks")
st.pyplot(fig4)
plt.close(fig4)

fig5, ax5 = plt.subplots(figsize=(12, 6))
sns.boxplot(data=st.session_state.returns, ax=ax5)
ax5.set_title("Boxplot of Stock Returns")
st.pyplot(fig5)
plt.close(fig5)

fig6, ax6 = plt.subplots(figsize=(12, 6))
st.session_state.returns.rolling(window=30).std().plot(ax=ax6)
ax6.set_title("Rolling Volatility of Selected Stocks")
st.pyplot(fig6)
plt.close(fig6)

# Portfolio Optimization
st.subheader("Optimize Portfolio")
if st.button("Run Optimization"):
    def portfolio_volatility(weights, cov_matrix):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    num_assets = len(st.session_state.returns.columns)
    initial_weights = np.ones(num_assets) / num_assets
    bounds = tuple((0, 1) for asset in range(num_assets))
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    
    optimized = sco.minimize(portfolio_volatility, initial_weights, args=(st.session_state.returns.cov()),
                              method='SLSQP', bounds=bounds, constraints=constraints)
    
    if optimized.success:
        st.session_state.portafolios_bl = pd.DataFrame([optimized.x], columns=st.session_state.returns.columns)
        st.success("Optimized portfolio successfully generated!")
        st.markdown("## Optimized Portfolio Weights")
        st.table(st.session_state.portafolios_bl.T.rename(columns={0: "Weight"}))
    else:
        st.error("Portfolio optimization failed.")

