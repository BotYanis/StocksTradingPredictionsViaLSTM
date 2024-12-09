import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, BatchNormalization, Bidirectional
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import time
import os
import logging
import h5py
from tensorflow.keras.models import load_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

# === Функции подготовки данных ===

def merge_company_and_market_data(company_data, market_data):
    """Объединение данных компании и индексов."""
    market_data['Datetime'] = pd.to_datetime(market_data['Datetime'])
    company_data['Datetime'] = pd.to_datetime(company_data['Datetime'])
    merged_data = pd.merge(company_data, market_data, on='Datetime', how='inner')
    logger.info("Данные успешно объединены.")
    return merged_data

def zscore_normalize_data(data, columns_to_scale):
    """Z-Score нормализация данных."""
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[columns_to_scale])
    return scaled_data, scaler

def inverse_transform(scaler, predicted_value, column_index):
    """Обратная трансформация Z-Score."""
    mean = scaler.mean_[column_index]
    std = scaler.scale_[column_index]
    return predicted_value * std + mean

def create_sequences(data, sequence_length, target_index):
    """Создание последовательностей для обучения."""
    sequences, targets = [], []
    for i in range(len(data) - sequence_length):
        sequences.append(data[i:i + sequence_length])
        targets.append(data[i + sequence_length, target_index])  # Целевая переменная: Close
    return np.array(sequences), np.array(targets)

# === Модель ===

def build_model(input_shape):
    """Построение LSTM модели."""
    model = Sequential([
        Bidirectional(LSTM(128, return_sequences=True), input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.2),
        Bidirectional(LSTM(128, return_sequences=False)),
        BatchNormalization(),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(1)  # Прогнозируем только Close
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    logger.info("Модель построена.")
    return model

def validate_h5_file(filename):
    """Проверка валидности H5 файла."""
    try:
        with h5py.File(filename, 'r') as _:
            return True
    except Exception as e:
        logger.error(f"Файл {filename} поврежден или имеет неверный формат: {e}")
        return False

def save_model(model, filename="model.h5"):
    """
    Сохранение модели на диск с созданием директории при необходимости.
    
    Args:
        model: Модель Keras для сохранения
        filename: Путь для сохранения модели
    Returns:
        bool: True если сохранение успешно, False в противном случае
    """
    try:
        # Получаем директорию из пути к файлу
        directory = os.path.dirname(filename)
        
        # Создаем директорию если она не существует
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Создана директория: {directory}")

        # Сохраняем модель
        model.save(filename)
        
        # Проверяем валидность сохраненного файла
        if validate_h5_file(filename):
            logger.info(f"Модель успешно сохранена в {filename}")
            return True
        return False
        
    except Exception as e:
        logger.error(f"Ошибка при сохранении модели: {e}")
        return False

def load_saved_model(filename="model.h5"):
    """Загрузка модели с диска."""
    if not os.path.exists(filename):
        logger.info("Сохраненная модель не найдена, требуется создать новую.")
        return None
        
    if not validate_h5_file(filename):
        logger.warning("Файл модели поврежден, требуется создать новую модель.")
        return None
        
    try:
        model = load_model(filename)
        logger.info(f"Модель успешно загружена из {filename}")
        return model
    except Exception as e:
        logger.error(f"Ошибка при загрузке модели: {e}")
        return None

# === Основной процесс ===

def fine_tune_and_predict(model, live_data_file, market_live_file, scaler, sequence_length, columns_to_use, target_index):
    """Обновление и предсказание модели."""
    live_data = pd.read_csv(live_data_file, parse_dates=['Datetime'])
    market_live_data = pd.read_csv(market_live_file, parse_dates=['Datetime'])
    merged_live_data = merge_company_and_market_data(live_data, market_live_data)
    scaled_live_data = scaler.transform(merged_live_data[columns_to_use])

    # Предсказание
    latest_sequence = scaled_live_data[-sequence_length:]
    latest_sequence = np.expand_dims(latest_sequence, axis=0)
    predicted_close_scaled = model.predict(latest_sequence)[0][0]
    predicted_close = inverse_transform(scaler, predicted_close_scaled, target_index)
    actual_close = merged_live_data['Close'].iloc[-1]

    logger.info(f"Предсказанная цена: {predicted_close}, Фактическая цена: {actual_close}")

    # Дообучение модели
    X_new, y_new = create_sequences(scaled_live_data, sequence_length, target_index)
    if len(X_new) > 0:
        model.fit(X_new, y_new, epochs=2, batch_size=1, shuffle=False)

    return predicted_close, actual_close

# === Главная функция ===
def main():
    company_name = "AAPL"
    train_file = f"data/{company_name}/{company_name}_with_indicators.csv"
    market_file = "data/indexes.csv"
    live_data_file = f"data/{company_name}/{company_name}_live_data.csv"
    market_live_file = "data/indexes_live_data.csv"
    model_file = "model.h5"

    sequence_length = 60
    columns_to_use = [
        'Open', 'High', 'Low', 'Close', 'Volume', 'SMA_20', 'SMA_50', 'RSI_14',
        'MACD', 'MACD_signal', 'MACD_diff', 'BB_upper', 'BB_lower',
        'Adj Close ^DJI', 'Adj Close ^GSPC', 'Adj Close ^IXIC', 'Adj Close ^RUT'
    ]
    target_column = 'Close'
    target_index = columns_to_use.index(target_column)

    # Загрузка данных
    company_data = pd.read_csv(train_file)
    market_data = pd.read_csv(market_file)
    data = merge_company_and_market_data(company_data, market_data)
    scaled_data, scaler = zscore_normalize_data(data, columns_to_use)

    X, y = create_sequences(scaled_data, sequence_length, target_index)

    # Загрузка или создание модели
    model = load_saved_model(model_file) or build_model((sequence_length, len(columns_to_use)))
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Первичное обучение
    if not os.path.exists(model_file):
        model.fit(X, y, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping])
        save_model(model, model_file)

    while True:
        try:
            predicted_close, actual_close = fine_tune_and_predict(
                model, live_data_file, market_live_file, scaler, sequence_length, columns_to_use, target_index
            )
            logger.info(f"Предсказанная цена: {predicted_close}, Фактическая цена: {actual_close}")
            time.sleep(300)  # Ждем 5 минут до новой свечи
        except Exception as e:
            logger.error(f"Ошибка: {e}")
            break

if __name__ == "__main__":
    main()