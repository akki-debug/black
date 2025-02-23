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
    else:
        st.error("Portfolio optimization failed.")

# Compute Performance Metrics
st.subheader("Performance Metrics")
st.markdown("### Sharpe Ratio Formula")
st.latex(r"""Sharpe = \frac{R_p - R_f}{\sigma_p} \times \sqrt{252}""")
sharpe_ratio = st.session_state.returns.mean() / st.session_state.returns.std() * np.sqrt(252)
st.dataframe(sharpe_ratio.rename("Sharpe Ratio"))

# Additional Financial Metrics
risk_free_rate = 0.04 / 252  # Assume 4% annual risk-free rate

# Sortino Ratio
st.markdown("### Sortino Ratio Formula")
st.latex(r"""Sortino = \frac{R_p - R_f}{\sigma_d} \times \sqrt{252}""")
negative_returns = st.session_state.returns[st.session_state.returns < 0].std()
sortino_ratio = (st.session_state.returns.mean() - risk_free_rate) / negative_returns * np.sqrt(252)
st.dataframe(sortino_ratio.rename("Sortino Ratio"))

# Maximum Drawdown & Additional Metrics
st.markdown("### Maximum Drawdown Formula")
st.latex(r"""MaxDrawdown = \frac{CumulativeReturn_{min} - CumulativeReturn_{max}}{CumulativeReturn_{max}}""")
cumulative_returns = (1 + st.session_state.returns).cumprod()
rolling_max = cumulative_returns.cummax()
drawdown = (cumulative_returns - rolling_max) / rolling_max
max_drawdown = drawdown.min()
st.dataframe(max_drawdown.rename("Max Drawdown"))

# Annualized Return
annualized_return = (1 + st.session_state.returns.mean()) ** 252 - 1
st.markdown("### Annualized Return")
st.dataframe(annualized_return.rename("Annualized Return"))

# Final Portfolio Evaluation
st.subheader("Portfolio Performance Summary")
final_returns = pd.DataFrame({
    "Initial Capital": [initial_capital],
    "Final Portfolio Value": [portfolio_final_value],
    "Percentage Return (%)": [percentage_return],
    "Annualized Return (%)": [annualized_return.mean() * 100],
    "Max Drawdown (%)": [max_drawdown.mean() * 100]
})
st.dataframe(final_returns.style.format({"Percentage Return (%)": "{:.2f}", "Annualized Return (%)": "{:.2f}", "Max Drawdown (%)": "{:.2f}"}))


