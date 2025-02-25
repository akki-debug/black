import pandas as pd
import numpy as np
import yfinance as yf
import scipy.optimize as sco
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from datetime import datetime

# Page Config
st.set_page_config(page_title="Enhanced Black-Litterman Model", page_icon="📈")
st.title("Enhanced Black-Litterman Portfolio Optimization")

# Initialize session state
if "portfolios_bl" not in st.session_state:
    st.session_state.portfolios_bl = None

# Data Retrieval
@st.cache_data
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

if not selected_stocks:
    st.warning("Please select at least one stock")
    st.stop()

try:
    data = get_stock_data(selected_stocks, start_date, end_date)
    returns = data.pct_change().dropna()
except Exception as e:
    st.error(f"Error fetching data: {str(e)}")
    st.stop()

# Black-Litterman Configuration
st.header("Black-Litterman Configuration")

expander = st.expander("Model Parameters Explanation")
expander.markdown("""
- **Delta (Risk Aversion):** Controls the trade-off between risk and return (Typical range: 2.0-3.0)
- **Tau:** Scaling factor for prior covariance matrix (Typical value: 0.05)
- **Confidence:** 0-100% certainty in each view (Idzorek's method)
""")

col1, col2 = st.columns(2)
with col1:
    delta = st.number_input("Risk Aversion (δ)", value=2.5, min_value=0.5, max_value=5.0, step=0.1)
with col2:
    tau = st.number_input("Tau (τ)", value=0.05, min_value=0.01, max_value=1.0, step=0.01)

# View Specification
st.subheader("Investor Views Specification")

views = []
num_views = st.number_input("Number of Views", min_value=0, max_value=5, value=1)

for i in range(num_views):
    st.markdown(f"### View #{i+1}")
    cols = st.columns(3)
    with cols[0]:
        view_type = st.radio("Type", ["Absolute", "Relative"], key=f"view_type_{i}")
    with cols[1]:
        if view_type == "Absolute":
            asset = st.selectbox("Asset", selected_stocks, key=f"asset_{i}")
            ret = st.number_input("Return (%)", min_value=-50.0, max_value=50.0, value=5.0, key=f"ret_{i}") / 100
            assets = [asset]
        else:
            asset1 = st.selectbox("Outperformer", selected_stocks, key=f"asset1_{i}")
            asset2 = st.selectbox("Underperformer", selected_stocks, key=f"asset2_{i}")
            ret = st.number_input("Performance Spread (%)", min_value=0.0, max_value=50.0, value=2.5, key=f"spread_{i}") / 100
            assets = [asset1, asset2]
    with cols[2]:
        confidence = st.slider("Confidence", 1, 100, 75, key=f"conf_{i}") / 100
    views.append({"type": view_type, "assets": assets, "return": ret, "confidence": confidence})

# Black-Litterman Calculations
def calculate_black_litterman(returns, views, delta, tau):
    # Calculate market implied returns
    cov_matrix = returns.cov()
    mean_return = returns.mean()
    
    # Calculate market portfolio using CAPM
    def market_portfolio_weights(delta, cov_matrix):
        n = cov_matrix.shape[0]
        initial_guess = np.repeat(1/n, n)
        bounds = ((0, 1),) * n
        
        def objective(weights):
            portfolio_variance = weights.T @ cov_matrix @ weights
            return delta/2 * portfolio_variance
        
        constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]
        result = sco.minimize(objective, initial_guess, 
                             method="SLSQP", bounds=bounds, constraints=constraints)
        return result.x
    
    weights_mkt = market_portfolio_weights(delta, cov_matrix)
    Pi = delta * cov_matrix @ weights_mkt  # Prior returns
    
    # Process views
    if len(views) > 0:
        P = []
        Q = []
        omega = []
        
        for view in views:
            p_vector = np.zeros(len(selected_stocks))
            if view["type"] == "Absolute":
                idx = selected_stocks.index(view["assets"][0])
                p_vector[idx] = 1
                Q.append(view["return"])
            else:
                idx1 = selected_stocks.index(view["assets"][0])
                idx2 = selected_stocks.index(view["assets"][1])
                p_vector[idx1] = 1
                p_vector[idx2] = -1
                Q.append(view["return"])
            
            P.append(p_vector)
            
            # Idzorek's confidence calculation
            tau_sigma_p = tau * cov_matrix @ p_vector
            p_sigma_p = p_vector @ tau_sigma_p
            omega.append(p_sigma_p * (1 - view["confidence"]) / view["confidence"])
        
        P = np.array(P)
        Q = np.array(Q)
        Omega = np.diag(omega)
        
        # Black-Litterman formula
        try:
            first_term = np.linalg.inv(tau * cov_matrix)
            second_term = P.T @ np.linalg.inv(Omega) @ P
            combined = first_term + second_term
            posterior_rets = np.linalg.inv(combined) @ (first_term @ Pi + P.T @ np.linalg.inv(Omega) @ Q)
        except np.linalg.LinAlgError:
            # Use pseudo-inverse if singular matrix
            first_term = np.linalg.pinv(tau * cov_matrix)
            posterior_rets = first_term @ Pi + P.T @ np.linalg.pinv(Omega) @ Q
            posterior_rets /= (1 + tau)
    else:
        posterior_rets = Pi
    
    return Pi, posterior_rets, weights_mkt

# Optimization
def optimize_portfolio(expected_returns, cov_matrix):
    n = len(expected_returns)
    initial_guess = np.repeat(1/n, n)
    bounds = ((0, 1),) * n
    
    def objective(weights):
        portfolio_return = weights @ expected_returns
        portfolio_risk = np.sqrt(weights.T @ cov_matrix @ weights)
        return -portfolio_return / portfolio_risk  # Maximize Sharpe ratio
    
    constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]
    result = sco.minimize(objective, initial_guess, 
                         method="SLSQP", bounds=bounds, constraints=constraints)
    return result.x

# Execution
if st.button("Run Black-Litterman Optimization"):
    with st.spinner("Calculating..."):
        Pi, posterior_rets, weights_mkt = calculate_black_litterman(returns, views, delta, tau)
        weights_optimal = optimize_portfolio(posterior_rets, returns.cov())
        
        # Calculate metrics
        def portfolio_metrics(weights, returns, cov_matrix):
            ret = weights @ returns
            vol = np.sqrt(weights.T @ cov_matrix @ weights)
            sharpe = ret / vol
            return ret*252, vol*np.sqrt(252), sharpe*np.sqrt(252)
        
        mkt_return, mkt_vol, mkt_sharpe = portfolio_metrics(weights_mkt, Pi, returns.cov())
        bl_return, bl_vol, bl_sharpe = portfolio_metrics(weights_optimal, posterior_rets, returns.cov())
        
        # Display results
        st.success("Optimization Complete!")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Market Portfolio Return", f"{mkt_return:.2%}")
            st.metric("Market Portfolio Volatility", f"{mkt_vol:.2%}")
            st.metric("Market Sharpe Ratio", f"{mkt_sharpe:.2f}")
        
        with col2:
            st.metric("BL Portfolio Return", f"{bl_return:.2%}")
            st.metric("BL Portfolio Volatility", f"{bl_vol:.2%}")
            st.metric("BL Sharpe Ratio", f"{bl_sharpe:.2f}")
        
        # Visualizations
        fig, ax = plt.subplots(1, 2, figsize=(16, 6))
        
        # Returns comparison
        df_rets = pd.DataFrame({
            "Asset": selected_stocks,
            "Prior": Pi,
            "Posterior": posterior_rets
        }).melt(id_vars="Asset", var_name="Type", value_name="Return")
        
        sns.barplot(data=df_rets, x="Asset", y="Return", hue="Type", ax=ax[0])
        ax[0].set_title("Prior vs Posterior Returns")
        ax[0].tick_params(axis='x', rotation=45)
        
        # Weights comparison
        df_weights = pd.DataFrame({
            "Asset": selected_stocks,
            "Market": weights_mkt,
            "BL Optimal": weights_optimal
        }).melt(id_vars="Asset", var_name="Portfolio", value_name="Weight")
        
        sns.barplot(data=df_weights, x="Asset", y="Weight", hue="Portfolio", ax=ax[1])
        ax[1].set_title("Portfolio Weights Comparison")
        ax[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        st.pyplot(fig)

# Performance Analysis
st.header("Historical Performance Analysis")
fig, ax = plt.subplots(figsize=(12, 6))
(1 + returns).cumprod().plot(ax=ax)
ax.set_title("Historical Cumulative Returns")
ax.set_ylabel("Growth of ₹1 Investment")
st.pyplot(fig)
