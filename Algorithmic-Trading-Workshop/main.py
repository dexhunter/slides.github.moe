import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# CoinAPI details
API_KEY = '55144C52-5186-494D-8846-941B30A2CAAA'
BASE_URL = 'https://rest.coinapi.io/v1/'

# Trading parameters
symbol = 'BTC/USDC'
fast_window = 10  # Short-term moving average
slow_window = 50  # Long-term moving average

def get_historical_data(symbol, start_date, end_date):
    headers = {'X-CoinAPI-Key': API_KEY}
    url = f"{BASE_URL}ohlcv/BINANCE_SPOT_{symbol.replace('/', '_')}/history?period_id=1DAY&time_start={start_date}&time_end={end_date}"
    response = requests.get(url, headers=headers)
    data = response.json()
    if response.status_code == 200:
        return pd.DataFrame(data)
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

def preprocess_data(df):
    df['time_period_start'] = pd.to_datetime(df['time_period_start'])
    df.set_index('time_period_start', inplace=True)
    df['price_close'] = df['price_close'].astype(float)
    return df

def calculate_sma(df, window):
    return df['price_close'].rolling(window=window).mean()

def backtest_strategy(df):
    df['fast_sma'] = calculate_sma(df, fast_window)
    df['slow_sma'] = calculate_sma(df, slow_window)
    df.dropna(inplace=True)  # Remove rows with NaN values

    df['position'] = 0
    df.loc[df['fast_sma'] > df['slow_sma'], 'position'] = 1  # Buy signal
    df.loc[df['fast_sma'] < df['slow_sma'], 'position'] = -1  # Sell signal
    
    # Calculate returns
    df['returns'] = df['price_close'].pct_change()
    df['strategy_returns'] = df['position'].shift(1) * df['returns']
    df['cumulative_returns'] = (1 + df['strategy_returns']).cumprod()

    return df

def plot_results(df):
    plt.figure(figsize=(10, 6))
    plt.plot(df['price_close'], label='BTC/USDC Price')
    plt.plot(df['fast_sma'], label=f'{fast_window}-Day SMA', linestyle='--')
    plt.plot(df['slow_sma'], label=f'{slow_window}-Day SMA', linestyle='--')
    plt.title('BTC/USDC Price and SMA Strategy')
    plt.xlabel('Date')
    plt.ylabel('Price (USDC)')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(df['cumulative_returns'], label='Cumulative Returns')
    plt.title('Strategy Cumulative Returns')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.show()

def main():
    start_date = '2023-01-01'
    end_date = '2024-01-01'
    df = get_historical_data(symbol, start_date, end_date)

    print(df)

    if df is not None:
        df = preprocess_data(df)
        df = backtest_strategy(df)
        plot_results(df)
    else:
        print("Failed to retrieve data")

if __name__ == "__main__":
    main()
