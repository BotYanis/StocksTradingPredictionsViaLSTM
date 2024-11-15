import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from keras.src.models.sequential import Sequential
from keras.src.layers.rnn.lstm import LSTM
from keras.src.layers.core.dense import Dense
from keras.src.layers.regularization.dropout import Dropout
from keras.src.layers.core.input_layer import Input
from keras.src.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

# 1. Helper Functions
def merge_company_and_market_data(company_data, market_data):
    # Проверка на пустой вход
    if company_data.empty or market_data.empty:
        raise ValueError("Input DataFrames cannot be empty")
    
    # Проверка пересечения дат
    common_dates = set(company_data['Date']).intersection(set(market_data['Date']))
    if not common_dates:
        raise ValueError("No overlapping dates between company_data and market_data")

    # Слияние данных
    merged_data = pd.merge(company_data, market_data, on='Date', how='inner')
    
    if merged_data.empty:
        raise ValueError("Merged DataFrame is empty - check if dates overlap")
    
    return merged_data

def preprocess_data(data, scaler=None, fit=False):
    # Validate input data
    if data.empty:
        raise ValueError("Input data cannot be empty")
        
    if fit:
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_scaled = scaler.fit_transform(data)
        return data_scaled, scaler
    
    # Validate scaler exists for transform
    if scaler is None:
        raise ValueError("Scaler must be provided when fit=False")
        
    return scaler.transform(data)

def create_sequences(data, time_steps):
    X, y = [], []
    for i in range(time_steps, len(data)):
        X.append(data[i-time_steps:i, :])
        y.append(data[i, 3])  # Close price
    return np.array(X), np.array(y)

def create_model(time_steps, features):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(time_steps, features)),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    return model

# 2. Configuration
time_steps = 60
features = ['Open', 'High', 'Low', 'Close', 'Volume', 
            'SMA_20', 'SMA_50', 'RSI_14', 'MACD', 'MACD_signal', 'MACD_diff', 
            'Adj Close ^DJI', 'Adj Close ^GSPC', 'Adj Close ^IXIC', 'Adj Close ^RUT']

# 3. Load and prepare data
# Load data with validation
train_company_data = pd.read_csv("AAPL_with_indicators.csv")
test_company_data = pd.read_csv("AAPL_valid.csv")
market_data = pd.read_csv("indexes.csv")

# Validate loaded data
print(f"Train company data shape: {train_company_data.shape}")
print(f"Test company data shape: {test_company_data.shape}")
print(f"Market data shape: {market_data.shape}")

# Convert dates
for df in [train_company_data, test_company_data, market_data]:
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])

# 1. First, split the test data range
test_start_date = test_company_data['Date'].min()
test_end_date = test_company_data['Date'].max()
print(f"\nTest data date range: {test_start_date} to {test_end_date}")

# 2. Load and filter market data for testing period
market_test_data = pd.read_csv("indexes.csv")  # Load from a new file for test period
market_test_data['Date'] = pd.to_datetime(market_test_data['Date'])

# 3. Print date ranges to debug
print(f"\nMarket data date range: {market_test_data['Date'].min()} to {market_test_data['Date'].max()}")

# 4. Create separate market data files
market_train_path = "market_train.csv"
market_test_path = "market_test.csv"

# Save separate market data files if they don't exist
try:
    market_train_data = pd.read_csv(market_train_path)
    market_test_data = pd.read_csv(market_test_path)
except FileNotFoundError:
    # Get all required market data
    all_market_data = pd.read_csv("indexes.csv")
    all_market_data['Date'] = pd.to_datetime(all_market_data['Date'])
    
    # Split market data
    mask_train = (all_market_data['Date'] < test_start_date)
    mask_test = (all_market_data['Date'] >= test_start_date) & (all_market_data['Date'] <= test_end_date)
    
    market_train_data = all_market_data[mask_train]
    market_test_data = all_market_data[mask_test]
    
    # Save splits
    market_train_data.to_csv(market_train_path, index=False)
    market_test_data.to_csv(market_test_path, index=False)

# Merge data
train_data = merge_company_and_market_data(train_company_data, market_train_data)
test_data = merge_company_and_market_data(test_company_data, market_test_data)

print(f"\nMerged train data shape: {train_data.shape}")
print(f"Merged test data shape: {test_data.shape}")

# Set date index
train_data.set_index('Date', inplace=True)
test_data.set_index('Date', inplace=True)

# Validate features exist in data
missing_features = [f for f in features if f not in train_data.columns]
if missing_features:
    raise ValueError(f"Missing features in train data: {missing_features}")

missing_features = [f for f in features if f not in test_data.columns]
if missing_features:
    raise ValueError(f"Missing features in test data: {missing_features}")

# Prepare training data
train_scaled, scaler = preprocess_data(train_data[features], fit=True)
X_train, y_train = create_sequences(train_scaled, time_steps)

# Prepare test data
test_scaled = preprocess_data(test_data[features], scaler)
X_test, y_test = create_sequences(test_scaled, time_steps)

print(f"\nScaled train data shape: {train_scaled.shape}")
print(f"Scaled test data shape: {test_scaled.shape}")

print(f"Training shapes - X: {X_train.shape}, y: {y_train.shape}")
print(f"Test shapes - X: {X_test.shape}, y: {y_test.shape}")

# 4. Create and train model
model = create_model(time_steps, len(features))
model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

# Add callbacks
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        verbose=1
    )
]

# Train model
history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    callbacks=callbacks,
    verbose=1,
    shuffle=False
)

# Make predictions
predictions = model.predict(X_test, batch_size=32, verbose=1)

# Обратное преобразование масштабирования для предсказаний и истинных значений
X_test_last = X_test[:, -1, :-1]
predictions_reshaped = predictions.reshape(-1, 1)
predicted_prices = scaler.inverse_transform(
    np.concatenate((X_test_last, predictions_reshaped), axis=1))[:, -1]

y_test_reshaped = y_test.reshape(-1, 1)
actual_prices = scaler.inverse_transform(
    np.concatenate((X_test_last, y_test_reshaped), axis=1))[:, -1]

# Оценка модели
mae = mean_absolute_error(actual_prices, predicted_prices)
print(f"Mean Absolute Error: {mae}")

# График результатов
plt.figure(figsize=(14, 5))
plt.plot(test_data.index[-len(actual_prices):], actual_prices, color='blue', label='Actual')
plt.plot(test_data.index[-len(predicted_prices):], predicted_prices, color='red', label='Predicted')
plt.title('Stock Price Prediction with Market Influence')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()