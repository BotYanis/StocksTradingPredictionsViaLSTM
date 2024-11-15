import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from keras.src.models.sequential import Sequential
from keras.src.layers.rnn.lstm import LSTM
from keras.src.layers.core.dense import Dense
from keras.src.layers.regularization.dropout import Dropout
from keras.src.layers.core.input_layer import Input
from keras.src.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# List of companies
companies = ["AAPL", "BLK", "CVX", "GS", "JPM", "KO", "MSFT", "XOM", "PG", "NVDA"]

# Load and combine data for all companies
combined_data = pd.read_csv("combined_data.csv")

# Preprocess the combined data
combined_data['Date'] = pd.to_datetime(combined_data['Date'])
combined_data.set_index('Date', inplace=True)

print(combined_data.head())

# Scale the combined data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(combined_data[['Open', 'High', 'Low', 'Close', 'Volume']])

# Extract data for the target company (e.g., AAPL)
target_company = "AAPL"
target_data = combined_data[combined_data['Company'] == target_company]
target_scaled = scaler.transform(target_data[['Open', 'High', 'Low', 'Close', 'Volume']])

# Load and preprocess the valid data
valid_data = pd.read_csv("AAPL_valid.csv")
valid_data['Date'] = pd.to_datetime(valid_data['Date'])
valid_data.set_index('Date', inplace=True)
valid_scaled = scaler.transform(valid_data[['Open', 'High', 'Low', 'Close', 'Volume']])

# Prepare training data
X_train = scaled_data[:, :-1]
y_train = scaled_data[:, -1]

# Prepare validation data for the target company
X_valid = valid_scaled[:, :-1]
y_valid = valid_scaled[:, -1]

# Reshape data to include timesteps dimension
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_valid = X_valid.reshape((X_valid.shape[0], 1, X_valid.shape[1]))

# Build the model
model = Sequential([
    LSTM(100, return_sequences=True, input_shape=(1, X_train.shape[2])),
    Dropout(0.2),
    LSTM(100, return_sequences=False),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_valid, y_valid), callbacks=[early_stopping])

# Predict for the target company
predictions = model.predict(X_valid)

# Inverse transform the predictions and the actual values
predictions = scaler.inverse_transform(np.concatenate((X_valid.reshape(X_valid.shape[0], X_valid.shape[2]), predictions), axis=1))[:, -1]
y_valid = scaler.inverse_transform(np.concatenate((X_valid.reshape(X_valid.shape[0], X_valid.shape[2]), y_valid.reshape(-1, 1)), axis=1))[:, -1]

# Evaluate the model
mae = mean_absolute_error(y_valid, predictions)
print(f"Mean Absolute Error: {mae}")

# Plot the results
plt.figure(figsize=(14, 5))
plt.plot(valid_data.index, y_valid, color='blue', label='Actual')
plt.plot(valid_data.index, predictions, color='red', label='Predicted')
plt.title(f'{target_company} Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()