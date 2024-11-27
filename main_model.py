import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

# Функция для объединения рыночных индексов и данных компании
def merge_company_and_market_data(company_data, market_data):
    market_data['Datetime'] = pd.to_datetime(market_data['Datetime'])
    company_data['Datetime'] = pd.to_datetime(company_data['Datetime'])
    merged_data = pd.merge(company_data, market_data, on='Datetime', how='inner')
    logger.info("Данные успешно объединены.")
    return merged_data

# Нормализация данных
def normalize_data(data, columns_to_scale):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[columns_to_scale])
    return scaled_data, scaler

# Создание последовательностей
def create_sequences(data, sequence_length, target_index=3):
    sequences, targets = [], []
    for i in range(len(data) - sequence_length):
        sequences.append(data[i:i + sequence_length])
        targets.append(data[i + sequence_length, target_index])  # Целевая переменная: Close
    return np.array(sequences), np.array(targets)

# Построение модели
def build_model(input_shape):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.3),
        LSTM(128, return_sequences=False),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    logger.info("Модель построена.")
    return model

# Прогнозирование
def predict_and_update(model, live_data_file, market_live_file, scaler, sequence_length, columns_to_use):
    live_data = pd.read_csv(live_data_file, parse_dates=['Datetime'])
    market_live_data = pd.read_csv(market_live_file, parse_dates=['Datetime'])
    
    # Используем последние несколько пятиминутных свечей для прогноза
    latest_data = live_data.tail(sequence_length)
    if latest_data.empty:
        logger.error("Нет данных для прогноза.")
        return None, None, None
    
    # Объединяем данные с рыночными индексами
    live_merged = pd.merge(latest_data, market_live_data, on='Datetime', how='inner')
    live_merged = live_merged[columns_to_use]

    # Нормализуем данные
    live_scaled = scaler.transform(live_merged)
    latest_sequence = live_scaled[-sequence_length:]
    latest_sequence = np.expand_dims(latest_sequence, axis=0)

    # Прогнозируем
    predicted_close_scaled = model.predict(latest_sequence)[0][0]
    predicted_close = scaler.inverse_transform([[0] * (len(columns_to_use) - 1) + [predicted_close_scaled]])[0][-1]
    actual_close = latest_data['Close'].iloc[-1]

    logger.info(f"Последовательность для прогнозирования: {latest_sequence.flatten()}")
    logger.info(f"Предсказанное нормализованное значение: {predicted_close_scaled}")
    logger.info(f"Предсказанная цена: {predicted_close}, Фактическая: {actual_close}")

    return predicted_close, latest_sequence, actual_close

# Основной процесс
def main():
    company_name = "AAPL"
    train_file = f"data/{company_name}_with_indicators.csv"
    market_file = "data/indexes.csv"
    live_data_file = f"data/{company_name}_live_data.csv"
    market_live_file = "data/indexes_live_data.csv"

    sequence_length = 60  # Будем использовать 60 последних пятиминутных свечей
    columns_to_use = [
        'Open', 'High', 'Low', 'Close', 'Volume', 'SMA_20', 'SMA_50', 'RSI_14',
        'MACD', 'MACD_signal', 'MACD_diff', 'BB_upper', 'BB_lower',
        'Adj Close ^DJI', 'Adj Close ^GSPC', 'Adj Close ^IXIC', 'Adj Close ^RUT'
    ]

    # Загружаем и объединяем данные для тренировки
    company_data = pd.read_csv(train_file)
    market_data = pd.read_csv(market_file)
    data = merge_company_and_market_data(company_data, market_data)
    scaled_data, scaler = normalize_data(data, columns_to_use)

    # Тренировка модели на данных
    X, y = create_sequences(scaled_data, sequence_length)

    model = build_model((sequence_length, len(columns_to_use)))
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Добавление коллбека для отслеживания потерь
    history = model.fit(X, y, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

    # График обучения
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss During Training')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Прогнозирование с использованием последних данных с 19:55:00
    while True:
        predicted_close, latest_sequence, actual_close = predict_and_update(
            model, live_data_file, market_live_file, scaler, sequence_length, columns_to_use
        )
        if predicted_close is not None:
            logger.info(f"Предсказанная цена закрытия: {predicted_close}, Фактическая цена закрытия: {actual_close}")
        
            # Дообучаем модель с новыми данными
            X_new = latest_sequence
            y_new = np.array([actual_close])

            # Переобучаем модель с новыми данными (дообучение)
            model.fit(X_new, y_new, epochs=1, batch_size=1, shuffle=False)

        # Задержка перед следующим обновлением (например, 5 минут)
        time.sleep(300)

if __name__ == "__main__":
    main()
