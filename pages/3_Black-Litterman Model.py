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

fig1 = sns.lineplot(data=st.session_state.data)
plt.title("Stock Closing Prices Over Time")
st.pyplot(fig1.figure)
plt.close()

fig2 = sns.heatmap(st.session_state.returns.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Stock Correlation Heatmap")
st.pyplot(fig2.figure)
plt.close()

# Compute Covariance Matrix
st.subheader("Covariance Matrix of the Excess Annual Returns:")
cov_matrix = st.session_state.returns.cov()
st.dataframe(cov_matrix)

# Compute Equilibrium Excess Returns
ew_pesos = np.ones(len(cov_matrix)) / len(cov_matrix)
desv_est_bl = np.sqrt(ew_pesos.T @ cov_matrix @ ew_pesos)
Lambda = (1 / desv_est_bl) * 0.5
vec_ec_bl = (cov_matrix @ ew_pesos) * Lambda

st.subheader("Equilibrium Vector:")
st.dataframe(vec_ec_bl)

# Compute Prior Variance
Tau = 1 / len(st.session_state.returns)
var_priori = Tau * cov_matrix
st.subheader("Prior Variance:")
st.dataframe(var_priori)

st.subheader("Financial Views:")
st.write("Now, introduce your views!")
num_views = st.number_input("Select the number of views:", min_value=1, step=1)
views_editable = pd.DataFrame(0, index=[f"View {i+1}" for i in range(num_views)], columns=cov_matrix.columns)
edited_views = st.data_editor(views_editable)

returns_editable = pd.DataFrame(0, index=[f"View {i+1}" for i in range(num_views)], columns=["Expected Return"])
edited_returns = st.data_editor(returns_editable)

conf_editable = pd.DataFrame(0, index=[f"View {i+1}" for i in range(num_views)], columns=["Confidence"])
edited_conf = st.data_editor(conf_editable)

if st.button("Continue"):
    try:
        P = np.array(edited_views)
        Q = np.array(edited_returns / 100)
        O = np.diag(edited_conf / 100)
        st.success("Information loaded successfully!")
    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.stop()

    aux1 = np.array(P @ var_priori @ P.T)
    O2 = np.diag(np.diag(aux1))
    E = np.linalg.inv(np.linalg.inv(Tau * cov_matrix) + P.T @ np.linalg.inv(O2) @ P) @ (
        np.linalg.inv(Tau * cov_matrix) @ vec_ec_bl + P.T @ np.linalg.inv(O2) @ Q)
    varianza_E = np.linalg.inv(np.linalg.inv(Tau * cov_matrix) + P.T @ np.linalg.inv(O2) @ P)

    Lambda1 = Lambda
    list_lambda = np.arange(1, 7.5, 0.5).tolist()
    list_lambda.insert(0, Lambda1.round(4))

    portafolios_bl = pd.DataFrame()
    for i in list_lambda:
        weights_bl = np.linalg.inv(cov_matrix * i) @ E
        portafolios_bl = pd.concat([portafolios_bl, pd.DataFrame(weights_bl).T.rename(index={0: i})])

    portafolios_bl.index.name = "Lambda"
    portafolios_bl.columns = cov_matrix.columns
    portafolios_bl['Total'] = portafolios_bl.sum(axis=1)
    portafolios_bl = portafolios_bl.sort_index()
    st.session_state.portafolios_bl = portafolios_bl

if "portafolios_bl" in st.session_state and st.session_state.portafolios_bl is not None:
    st.subheader("Optimized Portfolio with Varying Risk Aversion Levels (λ)")
    st.dataframe(st.session_state.portafolios_bl)
    
    fig3 = sns.barplot(data=st.session_state.portafolios_bl.drop(columns=["Total"]).T)
    plt.title("Asset Allocation by Risk Aversion Level (λ)")
st.pyplot(fig3.figure)
plt.close()
