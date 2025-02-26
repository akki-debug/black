import streamlit as st
st.set_page_config(
    page_title="Black-Litterman Project",
    page_icon = "mag"
)

texto = '''
# :moneybag: :bar_chart: Analysis and Optimization of Financial Portfolios with Python



## :bulb: Project Features

### 1.1. Financial Data Download :open_file_folder:
- **Data Source**: Downloads historical financial asset data using the Yahoo Finance API.


### 1.2. Price Visualization :chart_with_upwards_trend:
- **Closing Price Charts**: Visualization of historical prices for each asset.
- **Daily Returns**: Calculation of daily returns to analyze volatility.


### 1.3. Risk and Return Metrics Calculation :straight_ruler:
- **Value at Risk (VaR)**: Estimates the maximum risk over a given period under normal conditions.
- **Excess Kurtosis**: Analyzes the risk of extreme events in the assets.
- **Sortino Ratio**: Risk-adjusted performance metric focusing on negative returns.
- **Sharpe Ratio**: Risk-return ratio adjusted for the risk-free rate.

### 2.1. Portfolio Optimization : :muscle: 
Optimization based on:
- **Minimum Volatility**: Portfolio with the lowest volatility.
- **Maximum Sharpe Ratio**: Portfolio with the best risk-return ratio.
- **Minimum Volatility with Target Return**: Portfolio with minimum volatility for a specific 
return level.

### 2.2. Portfolio Backtesting  :thinking_face:
- **Comparison with Nifty 50**: Backtesting evaluates the effectiveness of generated strategies
by comparing them against the Nifty 50 index performance.

### 3.1 Black-Litterman Model :exploding_head:
- **Custom Financial Views**: Adjusts portfolio optimization based on expected returns for each
asset using the Black-Litterman model. 


'''

st.markdown(texto)

col1, col2 = st.columns([1,0.2])

with col2:
    if st.button("Start!"):
        st.switch_page("pages/1_Financial Data Download.py")
