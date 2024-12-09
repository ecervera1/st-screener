import streamlit as st
import pandas as pd
import asyncio
from aiohttp_retry import RetryClient, ExponentialRetry
from datetime import datetime

# Function to fetch quote data for a single ticker
async def fetch_quote_data(ticker, session):
    try:
        from pyfinviz.quote import Quote  # Ensure this is available in your environment
        quote = Quote(ticker=ticker)
        if quote.exists:
            result = {
                "fundamental_data": [],
                "outer_ratings": [],
                "outer_news": [],
                "income_statement": [],
                "insider_trading": [],
                "reuters_income_statement": []
            }
            for record in quote.fundamental_df.to_dict('records'):
                record.update({"Ticker": quote.ticker, "CDATE": datetime.now()})
                result["fundamental_data"].append(record)
            for record in quote.outer_ratings_df.to_dict('records'):
                record.update({"Ticker": quote.ticker, "CDATE": datetime.now()})
                result["outer_ratings"].append(record)
            for record in quote.outer_news_df.to_dict('records'):
                record.update({"Ticker": quote.ticker, "CDATE": datetime.now()})
                result["outer_news"].append(record)
            for record in quote.income_statement_df.to_dict('records'):
                record.update({"Ticker": quote.ticker, "CDATE": datetime.now()})
                result["income_statement"].append(record)
            return result
        else:
            st.warning(f"No data found for ticker {ticker}")
            return None
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return None

# Function to fetch data for multiple tickers
async def fetch_all_quote_data(tickers):
    retry_options = ExponentialRetry(attempts=3)
    async with RetryClient(raise_for_status=False, retry_options=retry_options) as session:
        tasks = [fetch_quote_data(ticker, session) for ticker in tickers]
        return await asyncio.gather(*tasks, return_exceptions=True)

# Function to combine results into DataFrames
def combine_results(results):
    fundamental_data = []
    outer_ratings = []
    outer_news = []
    income_statement = []
    for result in results:
        if isinstance(result, dict):
            fundamental_data.extend(result["fundamental_data"])
            outer_ratings.extend(result["outer_ratings"])
            outer_news.extend(result["outer_news"])
            income_statement.extend(result["income_statement"])
    return (
        pd.DataFrame(fundamental_data),
        pd.DataFrame(outer_ratings),
        pd.DataFrame(outer_news),
        pd.DataFrame(income_statement)
    )

# Function to clean and format data
def clean_data(df):
    return df.replace([None, pd.NA], "N/A").fillna("N/A")

# Main Streamlit app
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", ["Home", "Ticker Metrics Dashboard"])

    if page == "Home":
        st.title("Welcome to the Financial Dashboard")
        st.write("Navigate to the 'Ticker Metrics Dashboard' from the sidebar to view ticker-specific metrics.")
    
    elif page == "Ticker Metrics Dashboard":
        st.title("Ticker Metrics Dashboard")
        
        # User inputs for tickers
        tickers = st.text_input("Enter tickers separated by commas", "AAPL,MSFT,GOOGL").split(",")
        tickers = [ticker.strip() for ticker in tickers if ticker.strip()]
        
        # Filter for tickers
        selected_tickers = st.multiselect("Filter tickers to display", tickers, default=tickers)
        
        # Fetch data button
        if st.button("Fetch Metrics"):
            with st.spinner("Fetching metrics..."):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                results = loop.run_until_complete(fetch_all_quote_data(selected_tickers))
                loop.close()
                
                # Combine and clean data
                fundamental_data, outer_ratings, outer_news, income_statement = combine_results(results)
                fundamental_data = clean_data(fundamental_data)
                outer_ratings = clean_data(outer_ratings)
                outer_news = clean_data(outer_news)
                income_statement = clean_data(income_statement)

            # Display data in tabs
            st.subheader("Metrics Overview")
            tabs = st.tabs(["Fundamental Data", "Outer Ratings", "News", "Income Statement"])

            with tabs[0]:
                st.write("### Fundamental Data")
                st.dataframe(fundamental_data)

            with tabs[1]:
                st.write("### Outer Ratings")
                st.dataframe(outer_ratings)

            with tabs[2]:
                st.write("### News")
                st.dataframe(outer_news)

            with tabs[3]:
                st.write("### Income Statement")
                st.dataframe(income_statement)

if __name__ == "__main__":
    main()
