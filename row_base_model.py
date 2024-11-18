import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from keras.src.models import Sequential
from keras.src.layers import Dense, Dropout, LSTM
from keras.src.callbacks import EarlyStopping
import matplotlib.pyplot as plt

f_company_data = "data/PG_with_indicators.csv"
f_valid_company_data = "data/PG_valid.csv"

# Функция для объединения рыночных индексов и данных компании
def merge_company_and_market_data(company_data, market_data):
    market_data['Date'] = pd.to_datetime(market_data['Date'])
    company_data['Date'] = pd.to_datetime(company_data['Date'])
    # Объединяем данные по дате
    merged_data = pd.merge(company_data, market_data, on='Date', how='inner')
    return merged_data

# Загружаем тренировочные данные и данные индексов
train_company_data = pd.read_csv(f_company_data)
market_data = pd.read_csv("data/indexes.csv")

# Объединяем тренировочные данные
train_data = merge_company_and_market_data(train_company_data, market_data)

# Преобразуем дату в индекс
train_data.set_index('Date', inplace=True)

# Список колонок для масштабирования
features = ['Open', 'High', 'Low', 'Close', 'Volume', 
            'SMA_20', 'SMA_50', 'RSI_14', 'MACD', 'MACD_signal', 'MACD_diff', 
            'Adj Close ^DJI', 'Adj Close ^GSPC', 'Adj Close ^IXIC', 'Adj Close ^RUT']

# Проверка на наличие NaN значений
if train_data[features].isnull().values.any():
    raise ValueError("Training data contains NaN values. Please clean the data before proceeding.")

# Масштабируем тренировочные данные
scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train_data[features])

# Подготовка тренировочных данных для LSTM
time_steps = 32  # Временные окна
X_train, y_train = [], []
for i in range(time_steps, len(train_scaled)):
    X_train.append(train_scaled[i-time_steps:i, :])  # Временные окна
    y_train.append(train_scaled[i, features.index('Close')])  # Цена закрытия ('Close')

X_train, y_train = np.array(X_train), np.array(y_train)

# Проверьте форму данных
print(f"\n\nX_train shape: {X_train.shape}\n\n")  # Должно быть (num_samples, 32, num_features)

# Загружаем тестовые данные
test_data = pd.read_csv(f_valid_company_data)
test_data['Date'] = pd.to_datetime(test_data['Date'])
test_data.set_index('Date', inplace=True)

missing_features = [feature for feature in features if feature not in test_data.columns]
print(f"Missing features in test data: {missing_features}\n\n")

# Заполняем отсутствующие столбцы
for feature in missing_features:
    test_data[feature] = 0  # Или используйте средние значения, если это оправдано

print(test_data.head())

# Обновляем список признаков
updated_features = [feature for feature in features if feature in test_data.columns]
print(f"Updated feature list: {updated_features}\n\n")

# Проверка на наличие NaN значений
if test_data[updated_features].isnull().values.any():
    print("Test data contains NaN values. Filling NaN values with column mean.")
    test_data[updated_features] = test_data[updated_features].fillna(test_data[updated_features].mean())

# Проверка на наличие NaN значений после заполнения
if test_data[updated_features].isnull().values.any():
    raise ValueError("Test data still contains NaN values after filling. Please check the data.")

# Масштабируем тестовые данные
test_scaled = scaler.transform(test_data[updated_features])

#проверка на достаточность тренировочных данных(кол-во строк)
if len(test_scaled) < time_steps:
    raise ValueError("Test data is too small to create required time windows.")

# Создаем временные окна для тестового набора
X_test, y_test = [], []
for i in range(time_steps, len(test_scaled)):
    X_test.append(test_scaled[i-time_steps:i, :])  # Временные окна
    y_test.append(test_scaled[i, updated_features.index('Close')])  # Цена закрытия ('Close')

X_test, y_test = np.array(X_test), np.array(y_test)

# Проверьте форму данных
print(f"X_test shape: {X_test.shape}\n\n")  # Должно быть (num_samples, 32, num_features)

# Проверка на достаточное количество данных для тестирования
if X_test.shape[0] == 0:
    raise ValueError("Insufficient test data. Ensure that the test data has enough samples to create the required time steps.")

# Создаем модель LSTM
model = Sequential([
    LSTM(512, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),  # time_steps, num_features
    Dropout(0.2),
    LSTM(256, return_sequences=False, activation='relu'),
    Dropout(0.2),
    # LSTM(128, return_sequences=False, activation='relu'),
    # Dropout(0.2),
    Dense(1)
])


model.compile(optimizer='adam', loss='mean_squared_error')

# Обучение модели
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])

# Предсказания на тестовом наборе
predictions = model.predict(X_test)

# Проверка на наличие NaN значений в предсказаниях
if np.isnan(predictions).any():
    raise ValueError("Predictions contain NaN values. Please check the model and data preprocessing steps.")

# Обратное преобразование масштабирования для предсказаний и истинных значений
X_test_last = X_test[:, -1, :-1]
predictions_reshaped = predictions.reshape(-1, 1)

# Ensure the dimensions match before concatenation
if X_test_last.shape[0] != predictions_reshaped.shape[0]:
    raise ValueError(f"Dimension mismatch: X_test_last has {X_test_last.shape[0]} samples, but predictions_reshaped has {predictions_reshaped.shape[0]} samples.")

predicted_prices = scaler.inverse_transform(
    np.concatenate((X_test_last, predictions_reshaped), axis=1))[:, -1]

y_test_reshaped = y_test.reshape(-1, 1)
actual_prices = scaler.inverse_transform(
    np.concatenate((X_test_last, y_test_reshaped), axis=1))[:, -1]

# Оценка модели
mae = mean_absolute_error(actual_prices, predicted_prices)
print(f"Mean Absolute Error: {mae}\n\n")

# График результатов
plt.figure(figsize=(14, 5))
plt.plot(test_data.index[-len(actual_prices):], actual_prices, color='blue', label='Actual')
plt.plot(test_data.index[-len(predicted_prices):], predicted_prices, color='red', label='Predicted')
plt.title('Stock Price Prediction with Market Influence')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()