import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import scipy.optimize as sco
from numpy.linalg import multi_dot
from datetime import datetime
import plotly.express as px

# Page Config
st.set_page_config(page_title="Portfolio Optimization", page_icon="mag")
st.title("Portfolio Optimization & Backtesting")

# Fetch stock data directly using yfinance
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

# Check for NaN or Infinite Values
if st.session_state.returns.isnull().values.any() or np.isinf(st.session_state.returns.values).any():
    st.error("❌ Data contains NaN or infinite values. Please adjust the date range or stock selection.")
    st.stop()

# Ensure Enough Data Points for Optimization
if len(st.session_state.returns) < len(st.session_state.returns.columns):
    st.error("❌ Not enough data points for optimization. Try a longer date range.")
    st.stop()

# Additional Visualizations
st.markdown("## Data Visualization :bar_chart:")

fig1 = px.line(st.session_state.data, title="Stock Closing Prices Over Time")
st.plotly_chart(fig1)

fig2 = px.imshow(st.session_state.returns.corr(), text_auto=True, title="Stock Correlation Heatmap", color_continuous_scale='viridis')
st.plotly_chart(fig2)

def portfolio_stats(weights, returns, return_df=False):
    weights = np.array(weights).flatten()
    port_rets = np.dot(weights, returns.mean() * 252)
    port_vols = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    sharpe_ratio = port_rets / port_vols
    cumulative_returns = (1 + returns).cumprod()
    max_drawdown = (cumulative_returns / cumulative_returns.cummax() - 1).min().min()
    resultados = np.array([port_rets, port_vols, sharpe_ratio, max_drawdown])
    
    if return_df:
        return pd.DataFrame(data=np.round(resultados, 4), index=["Returns", "Volatility", "Sharpe Ratio", "Max Drawdown"], columns=["Resultado"])
    else:
        return resultados

st.markdown("## Optimization :muscle:")

def get_volatility(weights, returns):
    return portfolio_stats(weights, returns)[1]

def min_vol_opt(returns):
    cons = ({'type': 'eq', 'fun': lambda x: sum(x) - 1})
    bnds = tuple((0, 1) for x in range(len(returns.columns)))
    initial_wts = np.ones(len(returns.columns)) / len(returns.columns)  # Ensure valid initial weights
    opt_vol = sco.minimize(fun=get_volatility, x0=initial_wts, args=(returns,), method='SLSQP', bounds=bnds, constraints=cons)
    
    if not opt_vol.success:
        raise ValueError(f"Optimization failed: {opt_vol.message}")
    
    min_vol_pesos = pd.DataFrame(data=np.around(opt_vol['x'] * 100, 2), index=returns.columns, columns=["Min_Vol"])
    min_vol_stats = portfolio_stats(opt_vol['x'], returns, return_df=True)
    min_vol_stats = min_vol_stats.rename(columns={"Resultado": "Min_Vol"})
    return {"min_vol_pesos": min_vol_pesos, "min_vol_stats": min_vol_stats}

if "min_vol_resultados" not in st.session_state:
    st.session_state.min_vol_resultados = None

if st.button("Go!"):
    if st.session_state.returns is not None:
        try:
            st.session_state.min_vol_resultados = min_vol_opt(st.session_state.returns)
            st.success("Minimum Volatility Portfolio successfully optimized!")
        except Exception as e:
            st.error(f"⚠️ Optimization failed: {e}")

if st.session_state.min_vol_resultados is not None:
    st.subheader("Minimum Volatility Portfolio")
    st.markdown("Weights: :weight_lifter:")
    st.dataframe(st.session_state.min_vol_resultados["min_vol_pesos"])
    st.markdown("Stats: :money_with_wings:")
    st.dataframe(st.session_state.min_vol_resultados["min_vol_stats"])
    
    fig3 = px.pie(st.session_state.min_vol_resultados["min_vol_pesos"], values="Min_Vol", names=st.session_state.min_vol_resultados["min_vol_pesos"].index, title="Portfolio Composition")
    st.plotly_chart(fig3)
    
    fig4 = px.line(st.session_state.returns.cumsum(), title="Cumulative Returns Over Time")
    st.plotly_chart(fig4)
