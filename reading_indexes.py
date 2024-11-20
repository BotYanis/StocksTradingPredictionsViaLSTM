import yfinance as yf
import pandas as pd
import os
import schedule
import time
from datetime import datetime, timedelta

# Define the tickers
tickers = {
    '^GSPC': 'S&P 500',
    '^IXIC': 'NASDAQ',
    '^DJI': 'Dow Jones',
    '^RUT': 'Russell 2000'
}

# File to store combined data
combined_file = "data/indexes_live_data.csv"

# Ensure the combined file exists with the desired structure
def create_combined_file():
    if not os.path.exists("data"):
        os.makedirs("data")
    if not os.path.exists(combined_file):
        cols = ['Date'] + \
               [f'Adj Close {ticker}' for ticker in tickers.keys()] + \
               [f'Close {ticker}' for ticker in tickers.keys()] + \
               [f'High {ticker}' for ticker in tickers.keys()] + \
               [f'Low {ticker}' for ticker in tickers.keys()] + \
               [f'Open {ticker}' for ticker in tickers.keys()] + \
               [f'Volume {ticker}' for ticker in tickers.keys()]
        pd.DataFrame(columns=cols).to_csv(combined_file, index=False)
        print(f"Combined file {combined_file} created.")

# Fetch and append data for all tickers
def fetch_and_append_data():
    combined_data = pd.read_csv(combined_file)

    # Create a new row with the current date and time
    datetime_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    new_row = {'Date': datetime_str}

    for ticker in tickers.keys():
        # Fetch latest price data
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1d", interval="5m")

        if hist.empty:
            print(f"No data available for the given ticker: {ticker}.")
            continue

        # Calculate the target time for the previous candle (e.g., 20:40 if the current time is 20:49)
        current_time = datetime.now()
        target_time = (current_time - timedelta(minutes=10)).replace(second=0, microsecond=0)

        # Ensure target_time is timezone-naive to match hist.index
        target_time = target_time.replace(tzinfo=None)

        # Ensure hist.index is timezone-naive
        hist.index = hist.index.tz_localize(None)

        # Find the candle closest to the target time
        previous_candle = hist.loc[hist.index <= target_time].iloc[-2]
        if previous_candle.empty:
            print(f"No previous candle found for {ticker} at {target_time}.")
            continue

        # Extract data for the row
        new_row[f'Adj Close {ticker}'] = previous_candle.get('Adj Close', None)
        new_row[f'Close {ticker}'] = previous_candle['Close']
        new_row[f'High {ticker}'] = previous_candle['High']
        new_row[f'Low {ticker}'] = previous_candle['Low']
        new_row[f'Open {ticker}'] = previous_candle['Open']
        new_row[f'Volume {ticker}'] = previous_candle['Volume']

    # Add missing columns as NaN for consistency
    for col in combined_data.columns:
        if col not in new_row:
            new_row[col] = None

    # Append the new row to the combined data
    combined_data = pd.concat([combined_data, pd.DataFrame([new_row])], ignore_index=True)
    combined_data.to_csv(combined_file, index=False)
    print(f"Added new row: {new_row}")

# Schedule fetching data every 5 minutes
def schedule_five_minute_fetch():
    create_combined_file()
    
    # Fetch data immediately when the program runs
    fetch_and_append_data()
    
    # Schedule fetching data every 5 minutes
    schedule.every(5).minutes.do(fetch_and_append_data)
    print(f"Scheduled fetching data every 5 minutes.")

    while True:
        schedule.run_pending()
        time.sleep(1)

# Main function to start the scheduler
if __name__ == "__main__":
    schedule_five_minute_fetch()