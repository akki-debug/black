import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import scipy.optimize as sco
from numpy.linalg import multi_dot
from datetime import datetime

# Link to Financial Data Download
st.session_state.data = st.session_state.get("data", None)
st.session_state.returns = st.session_state.get("returns", None)
st.session_state.start_date = st.session_state.get("start_date", None)
st.session_state.end_date = st.session_state.get("end_date", None)

# Page Config
st.set_page_config(page_title="Portfolio Optimization", page_icon="mag")
st.title("Portfolio Optimization & Backtesting")

if st.session_state.data is not None and st.session_state.returns is not None:
    st.success("Session data successfully loaded from Financial Data Download.")
    st.write("Closing prices:")
    st.dataframe(st.session_state.data.tail())
    st.write("Daily returns:")
    st.dataframe(st.session_state.returns.tail())
else:
    st.warning("No data available. Please return to the Financial Data Download page to load the required data.")
    st.stop()

def portfolio_stats(weights, returns, return_df=False):
    weights = np.array(weights)[:, np.newaxis]
    port_rets = weights.T @ np.array(returns.mean() * 252)[:, np.newaxis]
    port_vols = np.sqrt(multi_dot([weights.T, returns.cov() * 252, weights]))
    sharpe_ratio = port_rets / port_vols
    resultados = np.array([port_rets, port_vols, sharpe_ratio]).flatten()
    
    if return_df:
        return pd.DataFrame(data=np.round(resultados, 4), index=["Returns", "Volatility", "Sharpe_Ratio"], columns=["Resultado"])
    else:
        return resultados

st.markdown("## Optimization :muscle:")

opt_range = st.slider("Select a date range:", min_value=st.session_state.start_date, max_value=st.session_state.end_date, value=(st.session_state.start_date, st.session_state.end_date), format="YYYY-MM-DD")
st.session_state.start_date_opt, st.session_state.end_date_opt = opt_range

st.write("Start:", st.session_state.start_date_opt)
st.write("End:", st.session_state.end_date_opt)

if "returns1" not in st.session_state:
    st.session_state.returns1 = None

if st.session_state.returns is not None:
    st.session_state.returns1 = st.session_state.returns.loc[st.session_state.start_date_opt:st.session_state.end_date_opt]

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
    if st.session_state.returns1 is not None:
        try:
            st.session_state.min_vol_resultados = min_vol_opt(st.session_state.returns1)
            st.success("Minimum Volatility Portfolio successfully optimized!")
        except:
            st.warning("An error occurred while optimizing the Minimum Volatility Portfolio.")

if st.session_state.min_vol_resultados is not None:
    st.subheader("Minimum Volatility Portfolio")
    st.markdown("Weights: :weight_lifter:")
    st.dataframe(st.session_state.min_vol_resultados["min_vol_pesos"])
    st.markdown("Stats: :money_with_wings:")
    st.dataframe(st.session_state.min_vol_resultados["min_vol_stats"])
