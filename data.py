import yfinance as yf
import ta
import os

def get_stock_data_with_indicators(ticker, start_date="1999-10-20", end_date="2024-09-20"):
    # Загрузка данных с помощью yfinance
    data = yf.download(ticker, start=start_date, end=end_date, interval="1d")
    
    # Проверка, что данные загружены
    if data.empty:
        raise ValueError(f"Нет данных для тикера {ticker} за указанный период.")
    
    # Убедимся, что данные представляют собой одномерные массивы для каждого столбца
    close = data['Close']  # Это уже Series, так что можно использовать напрямую
    
    # Убедимся, что данные передаются как Series (возможно, они были интерпретированы как DataFrame)
    close = close.squeeze()  # Эта команда избавится от лишних измерений, если они есть

    # Добавление индикаторов с помощью ta
    # Скользящие средние
    data['SMA_20'] = ta.trend.sma_indicator(close, window=20)
    data['SMA_50'] = ta.trend.sma_indicator(close, window=50)
    
    # Индекс относительной силы (RSI)
    data['RSI_14'] = ta.momentum.rsi(close, window=14)
    
    # MACD (MACD, сигнальная линия, гистограмма)
    macd = ta.trend.MACD(close)
    data['MACD'] = macd.macd()
    data['MACD_signal'] = macd.macd_signal()
    data['MACD_diff'] = macd.macd_diff()
    
    # Полосы Боллинджера
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    data['BB_upper'] = bb.bollinger_hband()
    data['BB_lower'] = bb.bollinger_lband()
    
    return data

# Пример использования
if __name__ == "__main__":
    ticker = "AAPL"  # Тикер компании
    stock_data = get_stock_data_with_indicators(ticker)
    
    # Удаление первых 49 строк
    stock_data = stock_data.iloc[49:]
    
    # Переупорядочивание столбцов
    stock_data.reset_index(inplace=True)  # Ensure 'Datetime' is a column
    stock_data['Date'] = stock_data['Date'].dt.strftime('%Y-%m-%d %H:%M:%S')  # Format 'Date' column
    
    stock_data = stock_data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'SMA_20', 'SMA_50', 'RSI_14', 'MACD', 'MACD_signal', 'MACD_diff', 'BB_upper', 'BB_lower']]
    
    # Сохранение данных в файл
    file_path = f"data/{ticker}_with_indicators.csv"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    stock_data.to_csv(file_path, index=False)
    
    print(stock_data.tail())  # Печать последних строк