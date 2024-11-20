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
data = yf.download(list(tickers.keys()), start="2024-01-01", end="2024-09-20", interval="1d")

# Remove the time, leaving just the date
data.reset_index(inplace=True)
data['Date'] = data['Date'].dt.strftime('%Y-%m-%d %H:%M:%S')

# Flatten the multi-index columns
data.columns = [' '.join(col).strip() for col in data.columns.values]

# Create a new DataFrame to store the combined data
combined_data = pd.DataFrame()

# Iterate over each ticker and append the data to the combined DataFrame
for ticker in tickers.keys():
    ticker_data = data[[col for col in data.columns if ticker in col] + ['Date']].copy()
    ticker_data.columns = [col.replace(f' {ticker}', f' {ticker}') for col in ticker_data.columns]
    if combined_data.empty:
        combined_data = ticker_data
    else:
        combined_data = combined_data.merge(ticker_data, on='Date', how='outer')

# Reorder the columns to match the specified order
cols = ['Date'] + [f'Adj Close {ticker}' for ticker in tickers.keys()] + \
       [f'Close {ticker}' for ticker in tickers.keys()] + \
       [f'High {ticker}' for ticker in tickers.keys()] + \
       [f'Low {ticker}' for ticker in tickers.keys()] + \
       [f'Open {ticker}' for ticker in tickers.keys()] + \
       [f'Volume {ticker}' for ticker in tickers.keys()]
combined_data = combined_data[cols]

# Save the data to a CSV file named combined_data.csv
combined_data.to_csv('data/pisyin.csv', index=False)