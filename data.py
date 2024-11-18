import ta
import yfinance as yf
import os

# Список компаний
companies = ["AAPL", "BLK", "CVX", "GS", "JPM", "KO", "MSFT", "NVDA", "PG", "XOM"]

def load_data_from_yfinance(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date)
    df.reset_index(inplace=True)
    return df

def add_technical_indicators(df):
    try:
        # Simple Moving Averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        
        # RSI - using alternative calculation method
        df['RSI_14'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi().squeeze()
        
        df['RSI_14'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi()
        macd = ta.trend.MACD(close=df['Close'])
        df['MACD'] = macd.macd().squeeze()
        df['MACD_signal'] = macd.macd_signal().squeeze()
        df['MACD'] = macd.macd()
        df['MACD_signal'] = macd.macd_signal()
        df['MACD_diff'] = macd.macd_diff()
        bollinger = ta.volatility.BollingerBands(close=df['Close'])
        df['BB_upper'] = bollinger.bollinger_hband().squeeze()
        df['BB_lower'] = bollinger.bollinger_lband().squeeze()
        df['BB_upper'] = bollinger.bollinger_hband()
        df['BB_lower'] = bollinger.bollinger_lband()
        df = df.fillna(method='bfill')
        
        return df
    
    except Exception as e:
        print(f"Error calculating indicators: {str(e)}")
        return None

# Main execution
for company in companies:
    try:
        file_path = f"data/{company}_with_indicators.csv"
        
        # Download data
        print(f"Processing {company}...")
        start_date = '2000-01-01'
        end_date = '2024-09-20'
        df = load_data_from_yfinance(company, start_date, end_date)
        
        # Validate downloaded data
        if df is None or df.empty:
            print(f"No data downloaded for {company}")
            continue
            
        # Add indicators
        df = add_technical_indicators(df)
        if df is None:
            print(f"Failed to calculate indicators for {company}")
            continue
            
        # Remove first 49 rows due to NaN values
        df = df.iloc[49:]
        
        # Save to CSV
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        print(f"Successfully saved data for {company}")
        
    except Exception as e:
        print(f"Error processing {company}: {str(e)}")
        continue
