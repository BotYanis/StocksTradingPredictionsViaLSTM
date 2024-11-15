import pandas as pd
import ta

# Список компаний
companies = ["AAPL"]

# Добавление индикаторов технического анализа
def add_technical_indicators(df):
    """
    Добавляет технические индикаторы в DataFrame с колонками ['Open', 'High', 'Low', 'Close', 'Volume'].
    """
    # Добавление скользящих средних
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # Индикатор RSI (Relative Strength Index)
    df['RSI_14'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi()
    
    # MACD (Moving Average Convergence Divergence)
    macd = ta.trend.MACD(close=df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    df['MACD_diff'] = macd.macd_diff()
    
    # Индикатор волатильности Bollinger Bands
    bollinger = ta.volatility.BollingerBands(close=df['Close'], window=20, window_dev=2)
    df['BB_upper'] = bollinger.bollinger_hband()
    df['BB_lower'] = bollinger.bollinger_lband()
    
    return df

# Пример: добавление индикаторов для каждой компании
for company in companies:
    # Загрузка данных компании
    df = pd.read_csv(f"{company}_valid.csv")
    
    # Проверка наличия колонки 'Date'
    if 'Date' not in df.columns:
        raise KeyError(f"'Date' column is missing in the data for {company}")
    
    # Преобразование колонки 'Date' в формат datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Установка колонки 'Date' в качестве индекса
    df.set_index('Date', inplace=True)
    
    # Добавление технических индикаторов
    df = add_technical_indicators(df)
    
    # Сохранение данных с индикаторами
    df.to_csv(f"{company}_valid.csv")