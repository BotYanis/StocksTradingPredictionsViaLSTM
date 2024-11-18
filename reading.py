import yfinance as yf
import pandas as pd
import os
import schedule
import time
from datetime import datetime, timedelta
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator
from ta.trend import MACD

# File to store candles and indicators
def create_file_if_not_exists(ticker):
    output_file = f"data/{ticker}_live_data.csv"
    if not os.path.exists(output_file):
        pd.DataFrame(columns=["Datetime", "Open", "High", "Low", "Close", "Volume", 
                              "SMA_20", "SMA_50", "RSI_14", "MACD", "MACD_signal", "MACD_diff",
                              "BB_upper", "BB_lower"]).to_csv(output_file, index=False)
        print(f"File {output_file} created.")
    return output_file

# Function to fetch the previous candle and append indicators
def fetch_previous_candle_with_indicators(ticker, output_file):
    # Fetch latest price data
    stock = yf.Ticker(ticker)
    hist = stock.history(period="1d", interval="5m")  # Keep enough data for indicators
    
    if hist.empty:
        print("No data available for the given ticker.")
        return None

    # Calculate the target time for the previous candle (e.g., 20:40 if the current time is 20:49)
    current_time = datetime.now()
    target_time = (current_time - timedelta(minutes=10)).replace(second=0, microsecond=0)

    # Ensure target_time is timezone-naive to match hist.index
    target_time = target_time.replace(tzinfo=None)

    # Ensure hist.index is timezone-naive
    hist.index = hist.index.tz_localize(None)

    # Find the candle closest to the target time
    previous_candle = hist.loc[hist.index <= target_time].iloc[-2]
    datetime_str = previous_candle.name.strftime('%Y-%m-%d %H:%M:%S')
    open_price = previous_candle['Open']
    high_price = previous_candle['High']
    low_price = previous_candle['Low']
    close_price = previous_candle['Close']
    volume = previous_candle['Volume']

    # Check if volume is zero and handle it
    if volume == 0:
        print(f"Volume is zero for {datetime_str}. Skipping this entry.")
        return

    # Create a DataFrame for calculations
    df = hist.reset_index()
    df = df.rename(columns={"Datetime": "Date", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})

    # Add indicators using `ta` library
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    df['SMA_50'] = df['close'].rolling(window=50).mean()
    df['RSI_14'] = RSIIndicator(df['close'], window=14).rsi()
    macd = MACD(df['close'])
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    df['MACD_diff'] = macd.macd_diff()
    bb = BollingerBands(df['close'])
    df['BB_upper'] = bb.bollinger_hband()
    df['BB_lower'] = bb.bollinger_lband()

    # Get the latest enhanced row
    latest_row = df.loc[df['Date'] == previous_candle.name].iloc[-1].to_dict()

    #фиксирует через 1:09 после формирования свечи

    # Append to CSV
    existing_data = pd.read_csv(output_file)
    new_row = {
        "Datetime": datetime_str,
        "Open": open_price,
        "High": high_price,
        "Low": low_price,
        "Close": close_price,
        "Volume": volume,
        "SMA_20": latest_row['SMA_20'],
        "SMA_50": latest_row['SMA_50'],
        "RSI_14": latest_row['RSI_14'],
        "MACD": latest_row['MACD'],
        "MACD_signal": latest_row['MACD_signal'],
        "MACD_diff": latest_row['MACD_diff'],
        "BB_upper": latest_row['BB_upper'],
        "BB_lower": latest_row['BB_lower']
    }

    # Append and save using pd.concat
    existing_data = pd.concat([existing_data, pd.DataFrame([new_row])], ignore_index=True)
    existing_data.to_csv(output_file, index=False)
    print(f"Added new candle with indicators: {new_row}")

# Function to schedule fetching data every 5 minutes
def schedule_five_minute_fetch(ticker):
    output_file = create_file_if_not_exists(ticker)
    
    # Fetch data immediately when the program runs
    fetch_previous_candle_with_indicators(ticker, output_file)
    
    # Schedule fetching data every 5 minutes
    schedule.every(5).minutes.do(fetch_previous_candle_with_indicators, ticker=ticker, output_file=output_file)
    print(f"Scheduled fetching data for {ticker} every 5 minutes.")

    while True:
        schedule.run_pending()
        time.sleep(1)

# Example usage
ticker = "AAPL"
schedule_five_minute_fetch(ticker)
