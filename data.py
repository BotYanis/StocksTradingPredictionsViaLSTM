import yfinance as yf
import pandas as pd

start_date = "2000-01-01"
end_date = "2024-01-01"

company = "NVDA"

# Download data
data = yf.download(company, start=start_date, end=end_date, interval="1d")
# Reset index to ensure 'Date' column is included
data.reset_index(inplace=True)
# Convert 'Date' column to date-only format
data['Date'] = pd.to_datetime(data['Date']).dt.date
# Add 'Company' column as the first column
data.insert(0, 'Company', company)
# Save to CSV file
data.to_csv(f"{company}_historical_data.csv", index=False)

# Load data from CSV file, skipping the second line
data = pd.read_csv(f"{company}_historical_data.csv", skiprows=[1])

# Debugging: Print columns to check if 'Date' column is present
print(f"Columns in {company}_historical_data.csv: {data.columns}")

# Ensure 'Date' column is present and convert it to datetime
if 'Date' in data.columns:
    data['Date'] = pd.to_datetime(data['Date']).dt.date
else:
    print(f"'Date' column not found in {company}_historical_data.csv")

# Save the processed data back to CSV file
data.to_csv(f"{company}_historical_data.csv", index=False)