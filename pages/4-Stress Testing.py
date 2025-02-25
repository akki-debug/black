import pandas as pd
import numpy as np
import yfinance as yf
import scipy.optimize as sco
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from datetime import datetime

# Page Config
st.set_page_config(page_title="Black-Litterman Model and Sensitivity Analysis", page_icon="mag")
st.title("Sensitivity Analysis")

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
else:
    st.warning("No data available. Please select valid tickers and date range.")
    st.stop()



# User-defined views
st.markdown("### Define Your Views on Expected Returns")
views = {}
for stock in selected_stocks:
    views[stock] = st.number_input(f"Expected Return for {stock} (%)", value=st.session_state.returns.mean()[stock] * 100)

view_vector = np.array([views[stock] / 100 for stock in selected_stocks])
confidence_levels = st.slider("Confidence Level in Views (%)", min_value=50, max_value=100, value=75) / 100

# Compute Adjusted Returns using Black-Litterman Model
cov_matrix = st.session_state.returns.cov()
mkt_implied_return = st.session_state.returns.mean().values
bl_adjusted_return = (1 - confidence_levels) * mkt_implied_return + confidence_levels * view_vector

# Portfolio Optimization with Adjusted Returns
if st.button("Run Optimization"):
    def portfolio_volatility(weights, cov_matrix):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    num_assets = len(st.session_state.returns.columns)
    initial_weights = np.ones(num_assets) / num_assets
    bounds = tuple((0, 1) for asset in range(num_assets))
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    
    optimized = sco.minimize(portfolio_volatility, initial_weights, args=(cov_matrix),
                              method='SLSQP', bounds=bounds, constraints=constraints)
    
    if optimized.success:
        optimized_weights = pd.DataFrame([optimized.x], columns=st.session_state.returns.columns).T
        optimized_weights.columns = ["Weight"]
        expected_return = np.dot(optimized.x, bl_adjusted_return) * 252
        expected_volatility = portfolio_volatility(optimized.x, cov_matrix)
        expected_sharpe = expected_return / expected_volatility
        
        st.success("Optimized portfolio successfully generated!")
        st.markdown("## Optimized Portfolio Weights")
        st.table(optimized_weights)
        
        st.markdown("### Portfolio Metrics")
        st.write(f"**Expected Annual Return:** {expected_return:.2%}")
        st.write(f"**Expected Volatility:** {expected_volatility:.2%}")
        st.write(f"**Sharpe Ratio:** {expected_sharpe:.2f}")
        
        # Sensitivity Analysis on Views
        st.markdown("## Sensitivity Analysis on Subjective Views")
        num_simulations = 1000
        perturbed_views = np.random.normal(loc=view_vector, scale=0.02, size=(num_simulations, len(selected_stocks)))
        simulated_returns = (1 - confidence_levels) * mkt_implied_return + confidence_levels * perturbed_views
        simulated_portfolio_returns = simulated_returns @ optimized.x * 252
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(simulated_portfolio_returns, bins=30, kde=True, ax=ax)
        ax.axvline(expected_return, color='r', linestyle='dashed', label='Optimized Return')
        ax.set_title("Distribution of Simulated Portfolio Returns Under Varying Views")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)
        
        st.write(f"**Simulated Expected Return Mean:** {np.mean(simulated_portfolio_returns):.2%}")
        st.write(f"**Simulated Standard Deviation:** {np.std(simulated_portfolio_returns):.2%}")
    else:
        st.error("Portfolio optimization failed.")
