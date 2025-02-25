import pandas as pd
import numpy as np
import yfinance as yf
import scipy.optimize as sco
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px
from datetime import datetime
from io import BytesIO

# Page Config
st.set_page_config(page_title="Advanced Black-Litterman Portfolio Optimizer", page_icon="📈", layout="wide")
st.title("Advanced Black-Litterman Portfolio Optimizer with Sensitivity Analysis")

# Nifty 50 Constituents (Updated List)
nifty50_stocks = [
    "RELIANCE.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", "TCS.NS",
    "HINDUNILVR.NS", "ITC.NS", "KOTAKBANK.NS", "SBIN.NS", "ASIANPAINT.NS",
    "TITAN.NS", "BAJFINANCE.NS", "LT.NS", "MARUTI.NS", "NTPC.NS", 
    "ONGC.NS", "SUNPHARMA.NS", "TATAMOTORS.NS", "ADANIPORTS.NS", "POWERGRID.NS",
    "NESTLEIND.NS", "ULTRACEMCO.NS", "BAJAJ-AUTO.NS", "GRASIM.NS", "JSWSTEEL.NS",
    "TATASTEEL.NS", "AXISBANK.NS", "HCLTECH.NS", "INDUSINDBK.NS", "DRREDDY.NS",
    "DIVISLAB.NS", "BHARTIARTL.NS", "UPL.NS", "BAJAJFINSV.NS", "HDFCLIFE.NS",
    "SBILIFE.NS", "EICHERMOT.NS", "SHREECEM.NS", "BRITANNIA.NS", "COALINDIA.NS",
    "TECHM.NS", "WIPRO.NS", "CIPLA.NS", "HEROMOTOCO.NS", "APOLLOHOSP.NS",
    "ADANIENT.NS", "BPCL.NS", "IOC.NS", "VEDL.NS"
]

# Initialize session state
if "portfolios_bl" not in st.session_state:
    st.session_state.portfolios_bl = None

# Enhanced Data Handling
@st.cache_data(ttl=3600)
def get_stock_data(tickers, start, end, interval='1d'):
    data = yf.download(tickers, start=start, end=end, interval=interval)["Close"]
    if data.empty:
        st.error("No data found for selected tickers. Please check your selections.")
        st.stop()
    return data.dropna(axis=1, how='all').dropna()

# Risk Metrics Calculations
def calculate_var(returns, confidence_level=0.95):
    return np.percentile(returns, 100 * (1 - confidence_level))

def calculate_cvar(returns, confidence_level=0.95):
    var = calculate_var(returns, confidence_level)
    return returns[returns <= var].mean()

# Efficient Frontier Calculation
def calculate_efficient_frontier(returns, cov_matrix, risk_free_rate=0.0):
    num_assets = len(returns)
    args = (cov_matrix,)
    bounds = tuple((0,1) for _ in range(num_assets))
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    
    target_returns = np.linspace(returns.min(), returns.max(), 50)
    efficient_portfolios = []
    
    for target_return in target_returns:
        constraints.append({'type': 'eq', 'fun': lambda x, r=target_return: r - x @ returns})
        result = sco.minimize(lambda x: np.sqrt(x.T @ cov_matrix @ x), 
                             num_assets*[1./num_assets], 
                             args=args, 
                             method='SLSQP', 
                             bounds=bounds, 
                             constraints=constraints)
        if result.success:
            efficient_portfolios.append({
                'Return': target_return,
                'Volatility': result.fun,
                'Sharpe': (target_return - risk_free_rate) / result.fun,
                'Weights': result.x
            })
    return pd.DataFrame(efficient_portfolios)

# Enhanced Black-Litterman Model
def black_litterman_adjustment(prior_returns, cov_matrix, P, Q, tau=0.05, omega=None):
    if omega is None:
        omega = np.diag(np.diag(tau * P @ cov_matrix @ P.T))
    
    # Black-Litterman formula
    inv_cov = np.linalg.inv(tau * cov_matrix)
    inv_omega = np.linalg.inv(omega)
    
    posterior_returns = np.linalg.inv(inv_cov + P.T @ inv_omega @ P) @ (inv_cov @ prior_returns + P.T @ inv_omega @ Q)
    posterior_cov = np.linalg.inv(inv_cov + P.T @ inv_omega @ P)
    
    return posterior_returns, posterior_cov

# Sidebar Configuration
with st.sidebar:
    st.header("Configuration Panel")
    selected_stocks = st.multiselect("Select Stocks (Nifty50)", 
                                   nifty50_stocks, 
                                   default=nifty50_stocks[:5])
    start_date = st.date_input("Start Date", value=datetime(2022, 1, 1))
    end_date = st.date_input("End Date", value=datetime(2023, 12, 31))
    data_freq = st.selectbox("Data Frequency", ['Daily', 'Weekly', 'Monthly'], index=0)
    risk_free_rate = st.number_input("Risk-Free Rate (%)", value=4.0, step=0.1)/100

# Data Loading and Processing
interval_map = {'Daily': '1d', 'Weekly': '1wk', 'Monthly': '1mo'}
try:
    data = get_stock_data(selected_stocks, start_date, end_date, interval_map[data_freq])
    returns = data.pct_change().dropna()
    annual_factor = {'Daily': 252, 'Weekly': 52, 'Monthly': 12}[data_freq]
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.stop()

# Data Exploration Section
st.header("Data Exploration")
col1, col2 = st.columns(2)
with col1:
    st.subheader("Asset Price History")
    fig = px.line(data, title="Price History")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Correlation Matrix")
    corr_matrix = returns.corr()
    fig = px.imshow(corr_matrix, text_auto=".2f", color_continuous_scale='RdBu_r', 
                   zmin=-1, zmax=1, title="Asset Return Correlations")
    st.plotly_chart(fig, use_container_width=True)

# Black-Litterman Configuration
st.header("Black-Litterman Model Configuration")
col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("Market Implied Returns")
    market_weights = data.iloc[-1] / data.iloc[-1].sum()  # Market-cap weights
    delta = st.slider("Risk Aversion Coefficient (δ)", 1.0, 5.0, 2.5, 0.1)
    prior_returns = delta * returns.cov() @ market_weights.values
    prior_returns_series = pd.Series(prior_returns, index=returns.columns)
    st.bar_chart(prior_returns_series * annual_factor)

with col2:
    st.subheader("Model Parameters")
    tau = st.slider("Uncertainty Scaling (τ)", 0.01, 1.0, 0.05, 0.01)
    confidence_level = st.slider("Base Confidence Level", 0.5, 1.0, 0.75, 0.05)

# User Views Configuration
st.subheader("Investor Views Configuration")
views = {}
P = []
Q = []
confidence_levels = []
for i, stock in enumerate(selected_stocks):
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        return_view = st.number_input(f"{stock} Return View (%)", 
                                    value=(prior_returns_series[stock] * annual_factor * 100).round(2),
                                    step=0.1, key=f"ret_{stock}")
    with col2:
        relative_view = st.checkbox(f"Relative View (vs Market)", key=f"rel_{stock}")
    with col3:
        conf = st.slider("Confidence", 0.1, 1.0, confidence_level, key=f"conf_{stock}")
    
    if relative_view:
        # Create relative view matrix
        P_row = np.zeros(len(selected_stocks))
        P_row[i] = 1
        P_row[-1] = -1  # Relative to last asset
        P.append(P_row)
        Q.append(return_view/100)
    else:
        P_row = np.zeros(len(selected_stocks))
        P_row[i] = 1
        P.append(P_row)
        Q.append(return_view/100)
    
    confidence_levels.append(conf)

P = np.array(P)
Q = np.array(Q)
omega = np.diag([(1 - cl) / cl * tau * (P[i] @ returns.cov() @ P[i].T) for i, cl in enumerate(confidence_levels)])

# Calculate Posterior Returns
posterior_returns, posterior_cov = black_litterman_adjustment(
    prior_returns, returns.cov(), P, Q, tau, omega
)

# Portfolio Optimization
st.header("Portfolio Optimization")
optimization_method = st.selectbox("Optimization Objective", 
                                  ["Minimum Volatility", "Maximum Sharpe", "Custom Target Return"])

target_return = None
if optimization_method == "Custom Target Return":
    target_return = st.number_input("Target Annual Return (%)", 
                                   min_value=0.0, 
                                   value=(posterior_returns.mean() * annual_factor * 100).round(2), 
                                   step=0.1)/100

def optimize_portfolio(objective, returns, cov_matrix, target_return=None):
    n = len(returns)
    bounds = tuple((0.0, 1.0) for _ in range(n))
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    
    if target_return is not None:
        constraints.append({'type': 'eq', 'fun': lambda x: x @ returns - target_return})
    
    if objective == "Minimum Volatility":
        result = sco.minimize(lambda x: x.T @ cov_matrix @ x,
                             n * [1./n], 
                             method='SLSQP',
                             bounds=bounds,
                             constraints=constraints)
    elif objective == "Maximum Sharpe":
        def neg_sharpe(x):
            ret = x @ returns
            vol = np.sqrt(x.T @ cov_matrix @ x)
            return -ret / vol
        result = sco.minimize(neg_sharpe,
                             n * [1./n], 
                             method='SLSQP',
                             bounds=bounds,
                             constraints=constraints)
    return result

result = optimize_portfolio(optimization_method, posterior_returns, posterior_cov, target_return)

if result.success:
    weights = result.x
    portfolio_return = weights @ posterior_returns * annual_factor
    portfolio_vol = np.sqrt(weights @ posterior_cov @ weights) * np.sqrt(annual_factor)
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_vol
    
    # Display Results
    col1, col2, col3 = st.columns(3)
    col1.metric("Expected Return", f"{portfolio_return:.2%}")
    col2.metric("Expected Volatility", f"{portfolio_vol:.2%}")
    col3.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
    
    # Weight Distribution
    st.subheader("Portfolio Composition")
    weights_df = pd.DataFrame(weights, index=selected_stocks, columns=['Weight'])
    fig = px.pie(weights_df, values='Weight', names=weights_df.index, hole=0.3)
    st.plotly_chart(fig, use_container_width=True)
    
   
    
    # Risk Analysis
    st.subheader("Risk Analysis")
    portfolio_returns = (returns @ weights).dropna()
    var_95 = calculate_var(portfolio_returns)
    cvar_95 = calculate_cvar(portfolio_returns)
    
    col1, col2 = st.columns(2)
    col1.metric("95% Value at Risk (VaR)", f"{var_95:.2%}")
    col2.metric("95% Conditional VaR (CVaR)", f"{cvar_95:.2%}")
    
    # Sensitivity Analysis
    st.subheader("Sensitivity Analysis")
    num_simulations = st.slider("Number of Simulations", 100, 10000, 1000)
    perturbation_scale = st.slider("Perturbation Scale", 0.01, 0.2, 0.05)
    
    simulated_weights = np.random.dirichlet(np.ones(len(selected_stocks)), num_simulations)
    simulated_returns = simulated_weights @ posterior_returns * annual_factor
    simulated_vol = np.sqrt((simulated_weights @ posterior_cov) * simulated_weights.sum(1)) * np.sqrt(annual_factor)
    
    fig = px.scatter(x=simulated_vol, y=simulated_returns, 
                    labels={'x':'Volatility', 'y':'Return'},
                    title="Portfolio Simulation Space")
    fig.add_trace(px.scatter(x=[portfolio_vol], y=[portfolio_return]).data[0])
    st.plotly_chart(fig, use_container_width=True)
    
    # Download Results
    st.subheader("Export Results")
    output = BytesIO()
    weights_df.to_excel(output, index=True)
    st.download_button("Download Portfolio Weights", 
                      data=output.getvalue(),
                      file_name="portfolio_weights.xlsx",
                      mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    
else:
    st.error("Portfolio optimization failed. Please adjust your constraints.")
