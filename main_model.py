import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from keras.src.models import Sequential
from keras.src.layers import Dense, Dropout, LSTM
from keras.src.callbacks import EarlyStopping
import time
import os
from collections import deque

# Конфигурация для компании
company_name = "AAPL"
train_file = f"data/{company_name}_with_indicators.csv"
live_data_file = f"data/{company_name}_live_data.csv"
market_file = "data/indexes.csv"
market_live_file = "data/indexes_live_data.csv"

lookback_years = 3  # Ограничение исторических данных (лет)
time_steps = 32     # Временные окна

# Очередь для хранения предсказаний
predictions_queue = deque(maxlen=1000)

# Функция для загрузки и ограничения данных
def load_recent_data(file, date_column, years):
    data = pd.read_csv(file)
    data[date_column] = pd.to_datetime(data[date_column])
    cutoff_date = pd.Timestamp.today() - pd.DateOffset(years=years)
    return data[data[date_column] >= cutoff_date]

# Загружаем данные
train_company_data = load_recent_data(train_file, "Date", lookback_years)
market_data = load_recent_data(market_file, "Date", lookback_years)

# Объединяем данные компании и рынка
def merge_company_and_market_data(company_data, market_data):
    merged_data = pd.merge(company_data, market_data, on='Date', how='inner')
    return merged_data

train_data = merge_company_and_market_data(train_company_data, market_data)
train_data.set_index('Date', inplace=True)

# Список признаков
features = ['Open', 'High', 'Low', 'Close', 'Volume', 
            'SMA_20', 'SMA_50', 'RSI_14', 'MACD', 'MACD_signal', 'MACD_diff', 
            'BB_upper', 'BB_lower',
            'Adj Close ^DJI', 'Adj Close ^GSPC', 'Adj Close ^IXIC', 'Adj Close ^RUT']

# Масштабирование данных
scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train_data[features])

# Подготовка данных для обучения
X_train, y_train = [], []
for i in range(time_steps, len(train_scaled)):
    X_train.append(train_scaled[i-time_steps:i, :])
    y_train.append(train_scaled[i, features.index('Close')])

X_train, y_train = np.array(X_train), np.array(y_train)

# Создаём модель
model = Sequential([
    LSTM(512, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(256, return_sequences=False, activation='relu'),
    Dropout(0.2),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')

# Обучение модели
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

# Функция для обработки новых данных
def process_live_data(live_data_file, market_live_file):
    # Загружаем новые данные
    live_data = pd.read_csv(live_data_file)
    live_market_data = pd.read_csv(market_live_file)
    
    # Объединяем данные
    live_merged = pd.merge(live_data, live_market_data, left_on='Datetime', right_on='Datetime', how='inner')
    live_merged['Datetime'] = pd.to_datetime(live_merged['Datetime'])
    
    # Масштабируем данные
    live_scaled = scaler.transform(live_merged[features])
    
    # Проверяем достаточность данных
    if len(live_scaled) < time_steps:
        raise ValueError("Not enough live data to create time steps for prediction.")
    
    # Создаём временные окна
    X_live = np.array([live_scaled[-time_steps:]])
    
    # Предсказание
    prediction_scaled = model.predict(X_live)
    prediction = scaler.inverse_transform(
        np.concatenate((live_scaled[-1, :-1].reshape(1, -1), prediction_scaled), axis=1)
    )[:, -1]
    
    return live_merged['Datetime'].iloc[-1], prediction[0]

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