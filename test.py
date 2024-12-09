# Delete if not needed

import requests
import pandas as pd
from datetime import datetime, timedelta
from io import StringIO
import time

# Список ваших API-ключей
API_KEYS = ['IDPWVPO8ELKSRHJR', ' 8GM7Y9JE7PZO7KBE', '6WDNN8MFTRN842LN', 'QNATLP38KR3ZLX7U', 'UH0RXYZQXRMU03V0',
            '3URKUOCDOHXYT9C5' , 'JYIU01E1UO3VIILY', '', 'PJQ4OGNDXEYMEPRJ', '5B7X3DYWS0I893YX', 'E21YGK92OF6WEXTZ',
            '4QZFD2IS0XTLNSB2', 'HUG2RAOVF2J353HV']
current_api_index = 0

# Настройки запроса
symbol = 'AAPL'
interval = '5min'
start_date = datetime(2021, 12, 30)
end_date = datetime(2024, 9, 23)
output_file = 'data/AAPL/apple_5min_2022_2024.csv'

# Функция для запроса данных за один день
def fetch_data_for_day(date):
    global current_api_index
    api_key = API_KEYS[current_api_index]
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval={interval}&apikey={api_key}&datatype=csv&outputsize=full'
    
    # Add delay to avoid rate limits
    time.sleep(12)  # Alpha Vantage has a rate limit of 5 calls per minute for free API keys
    
    response = requests.get(url)     
    
    if response.status_code == 200:
        try:
            # Print first few lines of response for debugging
            print("Response preview:", response.text[:200])
            
            df = pd.read_csv(StringIO(response.text))
            print("DataFrame columns:", df.columns.tolist())
            
            # Check if we got an error message instead of data
            if 'Note' in response.text or 'Information' in response.text:
                print(f"API limit reached for key {current_api_index}")
                current_api_index = (current_api_index + 1) % len(API_KEYS)
                return None
                
            # The API might use 'time' instead of 'timestamp'
            time_col = 'timestamp' if 'timestamp' in df.columns else 'time'
            
            if time_col not in df.columns:
                print(f"Error: No time column found. Available columns: {df.columns.tolist()}")
                return None
                
            df[time_col] = pd.to_datetime(df[time_col])
            df = df[df[time_col].dt.time.between(datetime.strptime('09:30', '%H:%M').time(),
                                              datetime.strptime('15:55', '%H:%M').time())]
            return df
            
        except Exception as e:
            print(f"Error processing data: {str(e)}")
            print("Response content:", response.text)
            return None
    else:
        print(f'Error fetching data for {date}: Status code {response.status_code}')
        current_api_index = (current_api_index + 1) % len(API_KEYS)
        return None

# Итерация по каждому дню
current_date = start_date
all_data = []

while current_date <= end_date:
    print(f'Получение данных за {current_date.date()}')
    data = fetch_data_for_day(current_date)
    if data is not None:
        all_data.append(data)
    # Переходим к следующему торговому дню (будние дни)
    current_date += timedelta(days=1)
    if current_date.weekday() >= 5:
        current_date += timedelta(days=(7 - current_date.weekday()))

# Объединяем все данные и сортируем
if all_data:
    combined_df = pd.concat(all_data)
    combined_df.sort_values(by='timestamp', ascending=True, inplace=True)
    combined_df.to_csv(output_file, index=False)
    print(f'Данные сохранены в файл {output_file}')
else:
    print('Не удалось получить данные')