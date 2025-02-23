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

fig1 = px.line(st.session_state.data, title="Stock Closing Prices Over Time")
st.plotly_chart(fig1)

fig2 = px.imshow(st.session_state.returns.corr(), text_auto=True, title="Stock Correlation Heatmap", color_continuous_scale='viridis')
st.plotly_chart(fig2)

def portfolio_stats(weights, returns, return_df=False):
    weights = np.array(weights)[:, np.newaxis]
    port_rets = weights.T @ np.array(returns.mean() * 252)[:, np.newaxis]
    port_vols = np.sqrt(multi_dot([weights.T, returns.cov() * 252, weights]))
    sharpe_ratio = port_rets / port_vols
    max_drawdown = (returns.cumsum().min() - returns.cumsum().max()) / returns.cumsum().max()
    resultados = np.array([port_rets, port_vols, sharpe_ratio, max_drawdown]).flatten()
    
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
    initial_wts = np.array(len(returns.columns) * [1. / len(returns.columns)])
    opt_vol = sco.minimize(fun=get_volatility, x0=initial_wts, args=(returns), method='SLSQP', bounds=bnds, constraints=cons)
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
        except:
            st.warning("An error occurred while optimizing the Minimum Volatility Portfolio.")

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
