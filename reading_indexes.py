import yfinance as yf
import pandas as pd
import os
import schedule
import time
from datetime import datetime, timedelta

# Список индексов
indices = ['^GSPC', '^IXIC', '^DJI', '^RUT']

# Файл для хранения данных
output_file = "data/indexes_live_data.csv"

# Создание файла, если он не существует
def create_file_if_not_exists():
    if not os.path.exists(output_file):
        columns = [
            "Datetime",
            "Adj Close ^GSPC", "Adj Close ^IXIC", "Adj Close ^DJI", "Adj Close ^RUT",
            "Close ^GSPC", "Close ^IXIC", "Close ^DJI", "Close ^RUT",
            "High ^GSPC", "High ^IXIC", "High ^DJI", "High ^RUT",
            "Low ^GSPC", "Low ^IXIC", "Low ^DJI", "Low ^RUT",
            "Open ^GSPC", "Open ^IXIC", "Open ^DJI", "Open ^RUT",
            "Volume ^GSPC", "Volume ^IXIC", "Volume ^DJI", "Volume ^RUT",
        ]
        pd.DataFrame(columns=columns).to_csv(output_file, index=False)
        print(f"File {output_file} created.")
    return output_file

# Функция для получения данных за предыдущую 5-минутную свечу
def fetch_previous_candle_with_indicators():
    global indices

    # Словарь для данных
    data = {}

    # Текущие дата и время
    current_time = datetime.now()
    # Round down to nearest 5 minutes
    minutes = (current_time.minute // 5) * 5
    target_time = current_time.replace(minute=minutes, second=0, microsecond=0) - timedelta(minutes=5)
    target_time = target_time.replace(tzinfo=None)

    # Сбор данных для каждого индекса
    for ticker in indices:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1d", interval="5m")

        if hist.empty:
            print(f"No data available for {ticker}. Skipping.")
            continue

        # Убираем временную зону
        hist.index = hist.index.tz_localize(None)

        try:
            # Получаем предыдущую свечу
            previous_candle = hist.loc[hist.index <= target_time].iloc[-2]
        except IndexError:
            print(f"Not enough data for {ticker}. Skipping.")
            continue

        data[f"Adj Close {ticker}"] = previous_candle['Close']
        data[f"Close {ticker}"] = previous_candle['Close']
        data[f"High {ticker}"] = previous_candle['High']
        data[f"Low {ticker}"] = previous_candle['Low']
        data[f"Open {ticker}"] = previous_candle['Open']
        data[f"Volume {ticker}"] = previous_candle['Volume']

    # Добавляем общие данные
    if data:
        data["Datetime"] = target_time.strftime('%Y-%m-%d %H:%M:%S')

        # Загружаем существующий файл и добавляем новую строку
        existing_data = pd.read_csv(output_file)
        new_row = pd.DataFrame([data])
        updated_data = pd.concat([existing_data, new_row], ignore_index=True)
        updated_data.to_csv(output_file, index=False)
        print(f"Added new data row: {data}")
    else:
        print("No data to add this time.")

# Планирование сбора данных каждые 5 минут
def schedule_five_minute_fetch():
    create_file_if_not_exists()

    # Немедленный сбор данных
    fetch_previous_candle_with_indicators()
    
    # Планирование
    schedule.every(5).minutes.do(fetch_previous_candle_with_indicators)
    print("Scheduled fetching data for indices every 5 minutes.")

    while True:
        schedule.run_pending()
        time.sleep(1)

# Запуск
schedule_five_minute_fetch()