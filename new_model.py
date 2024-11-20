import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
import time
import os
from collections import deque

def load_and_preprocess_data(train_file, market_file):
    # Загрузка данных
    company_data = pd.read_csv(train_file, parse_dates=['Date'])
    market_data = pd.read_csv(market_file, parse_dates=['Date'])

    # Объединение данных компании с рынком по дате
    merged_data = pd.merge(company_data, market_data, on='Date', how='inner')

    # Удаляем лишние столбцы, если нужно
    columns_to_use = [
        'Open', 'High', 'Low', 'Close', 'Volume', 'SMA_20', 'SMA_50', 'RSI_14',
        'MACD', 'MACD_signal', 'MACD_diff', 'BB_upper', 'BB_lower',
        'Adj Close ^DJI', 'Adj Close ^GSPC', 'Adj Close ^IXIC', 'Adj Close ^RUT'
    ]
    merged_data = merged_data[['Date'] + columns_to_use]

    # Нормализация данных
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(merged_data[columns_to_use])

    return merged_data, scaled_data, scaler

def create_sequences(data, sequence_length):
    sequences = []
    targets = []
    for i in range(len(data) - sequence_length):
        sequences.append(data[i:i + sequence_length])
        targets.append(data[i + sequence_length, 3])  # Целевая переменная: Close
    return np.array(sequences), np.array(targets)

def build_model(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)  # Прогноз цены закрытия
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

def train_model(model, X_train, y_train, epochs=50, batch_size=32):
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, shuffle=False)
    return model

def predict_and_update(model, live_data, market_live_data, scaler, sequence_length):
    # Загрузка и объединение текущих данных
    live_data = pd.read_csv(live_data_file, parse_dates=['Date'])
    market_live_data = pd.read_csv(market_live_file, parse_dates=['Date'])
    live_merged = pd.merge(live_data, market_live_data, on='Date', how='inner')

    # Нормализация
    live_scaled = scaler.transform(live_merged.iloc[:, 1:])

    # Создание последовательности для прогноза
    latest_sequence = live_scaled[-sequence_length:]
    latest_sequence = np.expand_dims(latest_sequence, axis=0)

    # Прогноз
    predicted_close = model.predict(latest_sequence)[0][0]

    return predicted_close

# Основной процесс
company_name = "AAPL"
train_file = f"data/{company_name}_with_indicators.csv"
market_file = "data/indexes.csv"
live_data_file = f"data/{company_name}_live_data.csv"
market_live_file = "data/indexes_live_data.csv"

# Подготовка данных
sequence_length = 60
merged_data, scaled_data, scaler = load_and_preprocess_data(train_file, market_file)
X, y = create_sequences(scaled_data, sequence_length)

# Разделение данных на обучающие и тестовые
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Создание и обучение модели
model = build_model((sequence_length, X.shape[2]))
model = train_model(model, X_train, y_train, epochs=10, batch_size=32)

# Цикл предсказаний и обновлений
import time
while True:
    predicted_close, latest_sequence, actual_close = predict_and_update(model, live_data_file, market_live_file, scaler, sequence_length)
    print(f"Предсказанная цена закрытия: {predicted_close}, Фактическая цена закрытия: {actual_close}")

    # Рассчитываем ошибку
    error = actual_close - predicted_close
    print(f"Ошибка предсказания: {error}")

    # Создаем новые данные для обновления модели
    X_new = np.expand_dims(latest_sequence, axis=0)  # Последняя последовательность
    y_new = np.array([actual_close])  # Фактическая цена закрытия

    # Обновление модели на основе нового наблюдения
    model = update_model(model, X_new, y_new)

    # Ждем 5 минут
    time.sleep(300)