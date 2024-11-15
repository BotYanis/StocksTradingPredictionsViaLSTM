import yfinance as yf
import pandas as pd

# Define the tickers and their corresponding names
tickers = {
    '^GSPC': 'S&P 500',
    '^IXIC': 'NASDAQ',
    '^DJI': 'Dow Jones',
    '^RUT': 'Russell 2000'
}

# Fetch the data
data = yf.download(list(tickers.keys()), start="2000-01-01", end="2024-01-01")

# Remove the time, leaving just the date
data.reset_index(inplace=True)
data['Date'] = data['Date'].dt.date

# Flatten the multi-index columns
data.columns = [' '.join(col).strip() for col in data.columns.values]

# Add a column for the index name
def get_index_name(row):
    for ticker, name in tickers.items():
        if f'Adj Close {ticker}' in row.index:
            return name
    return None

# Create a new DataFrame to store the combined data
combined_data = pd.DataFrame()

# Iterate over each ticker and append the data to the combined DataFrame
for ticker, name in tickers.items():
    ticker_data = data[[col for col in data.columns if ticker in col] + ['Date']].copy()
    ticker_data.columns = [col.replace(f' {ticker}', '') for col in ticker_data.columns]
    ticker_data['Index'] = name
    combined_data = pd.concat([combined_data, ticker_data])

# Reorder the columns to match the specified order
cols = ['Index', 'Date', 'Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']
combined_data = combined_data[cols]

# Save the data to a CSV file named combined_data.csv
data.to_csv('indexes.csv', index=False)