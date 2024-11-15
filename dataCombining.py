import pandas as pd

# List of companies
companies = ["AAPL", "BLK", "CVX", "GS", "JPM", "KO", "MSFT", "XOM", "PG", "NVDA"]

# Load and combine data for all companies
combined_data = pd.DataFrame()

for company in companies:
    data = pd.read_csv(f"{company}_historical_data.csv")
    data['Company'] = company
    combined_data = pd.concat([combined_data, data], ignore_index=True)

# Sort the combined data by date
combined_data['Date'] = pd.to_datetime(combined_data['Date'])
combined_data.sort_values(by='Date', inplace=True)

# Save the combined data to a CSV file
combined_data.to_csv("combined_data.csv", index=False)

print(combined_data.head())