import pandas as pd
import numpy as np
import yfinance as yf
import scipy.optimize as sco
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from datetime import datetime
from scipy.linalg import inv

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

# Black-Litterman Parameters
st.subheader("Black-Litterman Model")
market_capitalization = np.array([2.5, 1.8, 1.2, 1.5, 2.0])[:len(selected_stocks)]
market_weights = market_capitalization / market_capitalization.sum()
market_returns = st.session_state.returns.mean()
cov_matrix = st.session_state.returns.cov()

# User-defined subjective views
st.markdown("### Define Subjective Views")
view_stock = st.selectbox("Select stock for view:", selected_stocks)
expected_return_view = st.number_input("Expected return (%)", value=5.0) / 100
confidence = st.slider("Confidence in view (%)", 0, 100, 50) / 100

# Black-Litterman Model Computation
if st.button("Calculate Black-Litterman Portfolio"):
    tau = 0.05  # Scaling factor for covariance uncertainty
    P = np.zeros((1, len(selected_stocks)))
    P[0, selected_stocks.index(view_stock)] = 1
    Q = np.array([expected_return_view])
    Omega = np.diag(np.full(1, confidence * cov_matrix.to_numpy().trace() / len(selected_stocks)))
    
    # Compute Black-Litterman expected returns
    inv_term = inv(inv(tau * cov_matrix.to_numpy()) + P.T @ inv(Omega) @ P)
    adjusted_returns = inv_term @ (inv(tau * cov_matrix.to_numpy()) @ market_returns.to_numpy() + P.T @ inv(Omega) @ Q)
    
    # Optimize portfolio based on Black-Litterman returns
    def portfolio_volatility(weights, cov_matrix):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    num_assets = len(selected_stocks)
    initial_weights = np.ones(num_assets) / num_assets
    bounds = tuple((0, 1) for _ in range(num_assets))
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    
    optimized = sco.minimize(portfolio_volatility, initial_weights, args=(cov_matrix),
                              method='SLSQP', bounds=bounds, constraints=constraints)
    
    if optimized.success:
        st.session_state.portafolios_bl = pd.DataFrame([optimized.x], columns=selected_stocks)
        st.success("Optimized Black-Litterman portfolio successfully generated!")
    else:
        st.error("Portfolio optimization failed.")

# Compute Performance Metrics
st.subheader("Performance Metrics")
st.markdown("### Sharpe Ratio Formula")
st.latex(r"""Sharpe = \frac{R_p - R_f}{\sigma_p} \times \sqrt{252}""")
sharpe_ratio = st.session_state.returns.mean() / st.session_state.returns.std() * np.sqrt(252)
st.dataframe(sharpe_ratio.rename("Sharpe Ratio"))

# Display Optimized Portfolio
if st.session_state.portafolios_bl is not None:
    st.subheader("Optimized Portfolio Weights")
    st.dataframe(st.session_state.portafolios_bl)


