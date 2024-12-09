import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from prophet import Prophet
from scipy.optimize import minimize
from bs4 import BeautifulSoup
import requests

# Streamlit settings
st.set_option('deprecation.showPyplotGlobalUse', False)
st.markdown("""
<style>
    .stActionButton button[kind="header"], .stActionButton div[data-testid="stActionButtonIcon"] {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)

# --- Helper Functions ---

def fetch_yfinance_data(tickers, start_date, end_date):
    """
    Fetches stock data from Yahoo Finance for given tickers and date range.
    """
    try:
        data = yf.download(tickers, start=start_date, end=end_date, progress=False)
        return data
    except Exception as e:
        st.error(f"Error fetching stock data: {e}")
        return pd.DataFrame()


def plot_time_series(data, title):
    """
    Plots a time series line chart for the given data.
    """
    st.subheader(title)
    if not data.empty:
        st.line_chart(data['Adj Close'])
    else:
        st.warning("No data to display for the selected date range.")


def scrape_stock_info(ticker):
    """
    Scrapes basic stock information using Yahoo Finance API.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            "Current Price": info.get("currentPrice"),
            "Market Cap": info.get("marketCap"),
            "PE Ratio": info.get("trailingPE"),
            "52W High": info.get("fiftyTwoWeekHigh"),
            "52W Low": info.get("fiftyTwoWeekLow"),
            "Profit Margins": info.get("profitMargins"),
            "ROA": info.get("returnOnAssets"),
            "ROE": info.get("returnOnEquity")
        }
    except Exception as e:
        st.error(f"Error scraping data for {ticker}: {e}")
        return {}


def generate_prophet_forecast(ticker, start_date, end_date, forecast_days):
    """
    Generates a Prophet forecast plot for a given stock ticker.
    """
    try:
        data = fetch_yfinance_data(ticker, start_date, end_date)
        if data.empty:
            st.warning("No data available for Prophet forecast.")
            return
        ph_data = data[['Close']].reset_index()
        ph_data.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)
        
        # Fit the Prophet model
        model = Prophet()
        model.fit(ph_data)
        
        # Make future predictions
        future = model.make_future_dataframe(periods=forecast_days)
        forecast = model.predict(future)
        
        # Plot the forecast
        fig = model.plot(forecast)
        plt.title(f'Forecast for {ticker}')
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error generating Prophet forecast: {e}")


def monte_carlo_simulation(data, num_simulations=1000, forecast_days=252):
    """
    Performs a Monte Carlo simulation for the given stock data.
    """
    if data.empty:
        st.warning("No data available for Monte Carlo simulation.")
        return

    mean_return = data.pct_change().mean()
    std_dev = data.pct_change().std()
    initial_price = data.iloc[-1]

    simulations = []
    for _ in range(num_simulations):
        price_series = [initial_price]
        for _ in range(forecast_days):
            daily_return = np.random.normal(mean_return, std_dev)
            price_series.append(price_series[-1] * (1 + daily_return))
        simulations.append(price_series[-1])

    # Plot simulation results
    fig, ax = plt.subplots()
    ax.hist(simulations, bins=50, color='blue', alpha=0.7)
    ax.set_title('Monte Carlo Simulation: Final Price Distribution')
    ax.set_xlabel('Price')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)

    st.write(f"Simulated Mean Final Price: {np.mean(simulations):.2f}")
    st.write(f"Simulated Median Final Price: {np.median(simulations):.2f}")
    st.write(f"Simulated Std Dev of Final Price: {np.std(simulations):.2f}")


def optimize_portfolio(tickers, start_date, end_date, risk_free_rate):
    """
    Optimizes portfolio weights to maximize the Sharpe ratio.
    """
    try:
        data = fetch_yfinance_data(tickers, start_date, end_date)['Adj Close']
        if data.empty:
            st.warning("No data available for optimization.")
            return
        
        returns = data.pct_change().dropna()
        cov_matrix = returns.cov()
        
        def negative_sharpe(weights):
            portfolio_return = np.dot(weights, returns.mean()) * 252
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
            return -((portfolio_return - risk_free_rate) / portfolio_volatility)
        
        bounds = [(0, 1) for _ in tickers]
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        initial_guess = [1 / len(tickers)] * len(tickers)
        
        result = minimize(negative_sharpe, initial_guess, bounds=bounds, constraints=constraints)
        weights = result.x
        
        # Portfolio Stats
        annual_return = np.dot(weights, returns.mean()) * 252
        volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
        sharpe_ratio = (annual_return - risk_free_rate) / volatility
        
        # Display Results
        st.write("### Optimized Portfolio Weights")
        st.table(pd.DataFrame(weights, index=tickers, columns=['Weight']))
        st.write(f"**Annual Return:** {annual_return:.2f}")
        st.write(f"**Volatility:** {volatility:.2f}")
        st.write(f"**Sharpe Ratio:** {sharpe_ratio:.2f}")
    except Exception as e:
        st.error(f"Error optimizing portfolio: {e}")


def fetch_news(ticker):
    """
    Fetches news headlines related to a given stock ticker.
    """
    try:
        url = f"https://finance.yahoo.com/quote/{ticker}"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        headlines = soup.find_all('h3', class_="Mb(5px)")
        for i, headline in enumerate(headlines[:5], start=1):
            title = headline.get_text()
            link = f"https://finance.yahoo.com{headline.find('a')['href']}"
            st.markdown(f"{i}. [{title}]({link})")
    except Exception as e:
        st.error(f"Error fetching news: {e}")


# --- Streamlit UI ---

st.title("Stock Portfolio Analysis")

# Sidebar Inputs
st.sidebar.header("User Inputs")
tickers = st.sidebar.text_input("Enter stock tickers (comma-separated)", "AAPL,MSFT,GOOGL").split(',')
start_date = st.sidebar.date_input("Start Date", datetime(2020, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.today())

# Main App
if st.sidebar.button("Show Stock Performance"):
    stock_data = fetch_yfinance_data(tickers, start_date, end_date)
    plot_time_series(stock_data, "Stock Performance")

if st.sidebar.checkbox("Show Prophet Forecast"):
    forecast_days = st.sidebar.slider("Forecast Days", 30, 365, 90)
    selected_ticker = st.sidebar.selectbox("Select Ticker for Forecast", tickers)
    generate_prophet_forecast(selected_ticker, start_date, end_date, forecast_days)

if st.sidebar.checkbox("Run Monte Carlo Simulation"):
    num_simulations = st.sidebar.slider("Number of Simulations", 100, 10000, 1000)
    forecast_days = st.sidebar.slider("Forecast Days", 30, 252, 100)
    selected_ticker = st.sidebar.selectbox("Select Ticker for Monte Carlo", tickers)
    stock_data = fetch_yfinance_data([selected_ticker], start_date, end_date)['Adj Close']
    monte_carlo_simulation(stock_data, num_simulations, forecast_days)

if st.sidebar.checkbox("Optimize Portfolio"):
    risk_free_rate = st.sidebar.number_input("Risk-Free Rate (%)", 0.5) / 100
    optimize_portfolio(tickers, start_date, end_date, risk_free_rate)

if st.sidebar.checkbox("Show News & Articles"):
    selected_ticker = st.sidebar.selectbox("Select Ticker for News", tickers)
    fetch_news(selected_ticker)
