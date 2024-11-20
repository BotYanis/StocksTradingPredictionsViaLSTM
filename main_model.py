import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import EarlyStopping
import time
import os
from collections import deque

# Configuration
company_name = "AAPL"
train_file = f"data/{company_name}_with_indicators.csv"
live_data_file = f"data/{company_name}_live_data.csv"
market_file = "data/indexes.csv"
market_live_file = "data/indexes_live_data.csv"
merged_train_file = f"data/{company_name}_merged_train_data.csv"

lookback_years = 3
time_steps = 32
predictions_queue = deque(maxlen=1000)

def load_recent_data(file, date_column, years):
    data = pd.read_csv(file)
    # Standardize date column name and remove duplicates
    if 'Datetime' in data.columns:
        data = data.rename(columns={'Datetime': 'Date'})
    data['Date'] = pd.to_datetime(data[date_column])
    data = data.drop_duplicates(subset=['Date'])
    cutoff_date = pd.Timestamp.today() - pd.DateOffset(years=years)
    return data[data['Date'] >= cutoff_date]

def merge_company_and_market_data(company_data, market_data):
    # Ensure 'Date' column exists and remove duplicates
    company_data['Date'] = pd.to_datetime(company_data['Date'])
    company_data = company_data.drop_duplicates(subset=['Date'])
    market_data['Date'] = pd.to_datetime(market_data['Date'])
    market_data = market_data.drop_duplicates(subset=['Date'])
    
    # Merge data and remove duplicates
    merged_data = pd.merge(company_data, market_data, on='Date', how='inner')
    merged_data = merged_data.drop_duplicates(subset=['Date'])
    merged_data = merged_data.sort_values('Date').reset_index(drop=True)
    return merged_data

def process_live_data(live_data_file, market_live_file):
    # Load and standardize date columns
    live_data = pd.read_csv(live_data_file)
    live_market_data = pd.read_csv(market_live_file)
    
    # Rename Datetime to Date for consistency
    if 'Datetime' in live_data.columns:
        live_data = live_data.rename(columns={'Datetime': 'Date'})
    if 'Datetime' in live_market_data.columns:
        live_market_data = live_market_data.rename(columns={'Datetime': 'Date'})
    
    live_data['Date'] = pd.to_datetime(live_data['Date'])
    live_market_data['Date'] = pd.to_datetime(live_market_data['Date'])
    
    # Merge data
    live_merged = pd.merge(live_data, live_market_data, on='Date', how='inner')
    
    # Add timestamp feature
    live_merged['Timestamp'] = live_merged['Date'].map(pd.Timestamp.timestamp)
    
    # Select features and make prediction
    live_features = live_merged[features].copy()
    
    if len(live_features) < time_steps:
        raise ValueError("Not enough live data points for prediction")
        
    live_scaled = scaler.transform(live_features)
    X_live = np.array([live_scaled[-time_steps:]])
    
    prediction_scaled = model.predict(X_live)
    prediction = scaler.inverse_transform(
        np.concatenate((live_scaled[-1, :-1].reshape(1, -1), prediction_scaled), axis=1)
    )[:, -1]
    
    return live_merged['Date'].iloc[-1], prediction[0]

# Load and process training data
train_company_data = load_recent_data(train_file, "Date", lookback_years)
market_data = load_recent_data(market_file, "Date", lookback_years)
train_data = merge_company_and_market_data(train_company_data, market_data)
train_data.set_index('Date', inplace=True)

# Save merged training data
train_data.to_csv(merged_train_file)
print(f"Merged training data saved to {merged_train_file}")

# Feature preparation
features = ['Open', 'High', 'Low', 'Close', 'Volume', 
           'SMA_20', 'SMA_50', 'RSI_14', 'MACD', 'MACD_signal', 'MACD_diff',
           'BB_upper', 'BB_lower', 'Adj Close ^DJI', 'Adj Close ^GSPC', 
           'Adj Close ^IXIC', 'Adj Close ^RUT']

# Add timestamp feature
train_data['Timestamp'] = train_data.index.map(pd.Timestamp.timestamp)
features.append('Timestamp')

# Масштабирование данных
scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train_data[features])

# Prepare data for training and testing
X, y = [], []
for i in range(time_steps, len(train_scaled)):
    X.append(train_scaled[i - time_steps:i, :])
    y.append(train_scaled[i, features.index('Close')])

X = np.array(X)
y = np.array(y)

# Split data into training and testing sets (e.g., 80% training, 20% testing)
split_index = int(0.8 * len(X))
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Create and compile the model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])

# Make predictions
predictions = model.predict(X_test)

# Reshape predictions
predictions_reshaped = predictions.reshape(-1, 1)
y_test_reshaped = y_test.reshape(-1, 1)

# Combine predictions with corresponding features (excluding the target)
X_test_last = X_test[:, -1, :-1]  # Exclude the target feature from features
num_features = X_test_last.shape[1]

# Concatenate features with predictions
pred_input = np.concatenate((X_test_last, predictions_reshaped), axis=1)
actual_input = np.concatenate((X_test_last, y_test_reshaped), axis=1)

# Create DataFrames to prevent duplicate indices
pred_df = pd.DataFrame(pred_input, columns=features)
actual_df = pd.DataFrame(actual_input, columns=features)

# Inverse transform
predicted_prices = scaler.inverse_transform(pred_df)[:, features.index('Close')]
actual_prices = scaler.inverse_transform(actual_df)[:, features.index('Close')]

# Evaluate the model
mae = mean_absolute_error(actual_prices, predicted_prices)
print(f"Mean Absolute Error: {mae}")

# Цикл обработки новых данных
while True:
    try:
        # Делаем предсказание
        prediction_time, next_prediction = process_live_data(live_data_file, market_live_file)
        print(f"Predicted close price for {prediction_time}: {next_prediction:.2f}")
        
        # Сохраняем предсказание
        predictions_queue.append((prediction_time, next_prediction))
        
        # Проверяем, есть ли реальное значение для дообучения
        real_data = pd.read_csv(live_data_file)  # Загружаем актуальные данные
        real_data['Datetime'] = pd.to_datetime(real_data['Datetime'])
        real_row = real_data[real_data['Datetime'] == prediction_time]
        
        if not real_row.empty:
            real_close_price = real_row['Close'].iloc[0]
            print(f"Real close price for {prediction_time}: {real_close_price:.2f}")
            
            # Масштабируем реальные данные
            real_scaled = scaler.transform(real_row[features])
            
            # Создаём данные для дообучения
            X_new = np.array([real_scaled[-time_steps:]])
            y_new = scaler.transform([[0] * (len(features) - 1) + [real_close_price]])[0, -1]
            
            # Дообучение модели
            model.fit(X_new, np.array([y_new]), epochs=1, verbose=0)
        
        # Ждём 5 минут
        time.sleep(300)
    except Exception as e:
        print(f"Error during prediction: {e}")
        break