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

# Fetch stock data directly using yfinance
nifty50_stocks = [
    "RELIANCE.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", "TCS.NS",
    "HINDUNILVR.NS", "ITC.NS", "LT.NS", "AXISBANK.NS", "BHARTIARTL.NS"
]
selected_stocks = st.multiselect("Select stocks for portfolio:", nifty50_stocks, default=nifty50_stocks[:5])

start_date = st.date_input("Start Date", value=datetime(2023, 1, 1))
end_date = st.date_input("End Date", value=datetime(2023, 12, 31))

def get_stock_data(tickers, start, end):
    data = yf.download(tickers, start=start, end=end)["Close"].dropna()
    return data

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

# Portfolio Optimization Button
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
    else:
        st.error("Portfolio optimization failed.")

# Compute Performance Metrics
st.subheader("Performance Metrics")
sharpe_ratio = st.session_state.returns.mean() / st.session_state.returns.std() * np.sqrt(252)
st.write("Sharpe Ratio of Selected Stocks:")
st.dataframe(sharpe_ratio.rename("Sharpe Ratio"))

# Compute Drawdown Analysis
cumulative_returns = (1 + st.session_state.returns).cumprod()
rolling_max = cumulative_returns.cummax()
drawdown = (cumulative_returns - rolling_max) / rolling_max
max_drawdown = drawdown.min()

st.subheader("Maximum Drawdown")
st.dataframe(max_drawdown.rename("Max Drawdown"))

fig3, ax3 = plt.subplots(figsize=(12, 6))
sns.lineplot(data=drawdown, ax=ax3)
ax3.set_title("Drawdown Over Time")
st.pyplot(fig3)
plt.close(fig3)

# Backtesting
st.subheader("Backtesting: Portfolio Performance")
initial_capital = 100000  # Set initial investment amount
if st.session_state.portafolios_bl is not None:
    weights = st.session_state.portafolios_bl.iloc[0].values
    if len(weights) != len(st.session_state.returns.columns):
        st.error("Mismatch between portfolio weights and returns. Check asset selection.")
        st.stop()
    
    portfolio_returns = (st.session_state.returns @ weights).cumsum()
    equal_weight_returns = (st.session_state.returns.mean(axis=1)).cumsum()
    
    portfolio_final_value = initial_capital * (1 + portfolio_returns.iloc[-1])
    percentage_return = ((portfolio_final_value - initial_capital) / initial_capital) * 100
    
    fig6, ax6 = plt.subplots(figsize=(12, 6))
    sns.lineplot(x=portfolio_returns.index, y=portfolio_returns, label="Optimized Portfolio", ax=ax6)
    sns.lineplot(x=equal_weight_returns.index, y=equal_weight_returns, label="Equal Weight Portfolio", ax=ax6)
    ax6.set_title("Cumulative Portfolio Returns")
    st.pyplot(fig6)
    plt.close(fig6)
    
    st.write("Final Portfolio Performance:")
    final_returns = pd.DataFrame({
        "Initial Capital": [initial_capital],
        "Final Portfolio Value": [portfolio_final_value],
        "Percentage Return (%)": [percentage_return]
    })
    st.dataframe(final_returns)
else:
    st.warning("Optimized portfolio not found. Run optimization first.")


