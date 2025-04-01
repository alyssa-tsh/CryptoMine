#Data Loading
import ccxt
import pandas as pd
from datetime import datetime, timedelta
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator,money_flow_index
import numpy as np

def get_data():

    # Initialize exchange
    exchange = ccxt.binance()

    # Define symbol and timeframe
    symbol = 'BTC/USDT'
    timeframe = '1d'

    # Calculate timestamp for 1 year ago from Janauary as train data 
    end_date = datetime(2025, 1, 1) 
    start_date = end_date - timedelta(days=366)  

    # Convert to milliseconds (Binance API uses Unix timestamp in milliseconds)
    since = int(start_date.timestamp() * 1000)

    # Fetch historical OHLCV data
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since)

    # Convert to DataFrame
    data = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')

    return data


# MODULE: DATA_PROCESSING - main dataframe

# Calculate technical indicators
def calculate_technical_indicators(data):
    for n in [5,10,15]:
        data[f'SMA_{n}'] = SMAIndicator(data['close'], window = n).sma_indicator()
    data['EMA_9'] = EMAIndicator(data['close'], window=9).ema_indicator()
    data['RSI'] = RSIIndicator(data['close'], window=14).rsi()
    macd = MACD(data['close'], window_fast=12, window_slow=26, window_sign=9)
    data['MACD'] = macd.macd() 
    # data['MACD_signal'] = macd.macd_signal()
    # data['MACD_hist'] = macd.macd_diff()
    data['OBV'] = OnBalanceVolumeIndicator(data['close'], data['volume']).on_balance_volume()
    data['MFI_10'] = money_flow_index(data['high'], data['low'], data['close'], data['volume'], window=10)
    data['ATR_14'] = AverageTrueRange(data['high'], data['low'], data['close'], window=14).average_true_range()
    bands = BollingerBands(data['close'], window=20, window_dev = 2)
    data['upper_band'] = bands.bollinger_hband()
    # data['middle_band'] = bands.bollinger_mavg()
    data['lower_band'] = bands.bollinger_lband() 
    # data['BBW'] = (data['upper_band'] - data['lower_band']) / data['middle_band']
    data['daily_momentum'] = data['open'] - data['close']
    data['rolling_mean_3'] = data['close'].rolling(window=3).mean()
    data['rolling_std_3'] = data['close'].rolling(window=3).std()
    
    # Lagged Close prices
    data['close_lag_1'] = data['close'].shift(1)
    data['close_lag_2'] = data['close'].shift(2)
    #lead close prices
    for n in [1, 2, 5, 7, 10, 15, 20, 30]:
        data[f'close_{n}_ahead'] = data['close'].shift(-n)
    data['log_return'] = np.log(data['close']).diff()
    data['return_1'] =  (data['close_1_ahead'] - data['close']) / data['close']  
    
    return data 

#handle missing values
def handle_missing_data(data):
    data.ffill(inplace=True)
    data.bfill(inplace = True)
    return data 



#MODULE: SPLITTING DATA
def split_data():
    btc_data = get_data()
    data = btc_data
    # Features and target
    features = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'SMA_5',
        'SMA_10', 'SMA_15', 'EMA_9', 'RSI', 'MACD', 'OBV', 'MFI_10', 'ATR_14',
        'upper_band', 'lower_band', 'daily_momentum', 'rolling_mean_3',
        'rolling_std_3', 'close_lag_1', 'close_lag_2']
    targets = ['close_1_ahead', 'close_2_ahead', 'close_5_ahead', 'close_7_ahead', 'close_10_ahead', 'close_15_ahead', 'close_20_ahead', 'close_30_ahead', 'return_1']

    #define date range correctly
    train_end_date = pd.to_datetime('2025-01-01') 
    val_end_date = pd.to_datetime('2025-02-28')
    test_end_date = pd.to_datetime('2025-03-01')

    # Split the data
    train_data = data[data['timestamp'] < train_end_date]
    val_data = data[(data['timestamp'] >= train_end_date) & (data['timestamp'] <= val_end_date)]
    test_data = data[(data['timestamp'] >= val_end_date)]

    # Split into X & y 
    X_train = train_data[features].drop(columns = ['timestamp'])
    X_val = val_data[features].drop(columns = ['timestamp'])
    X_test = test_data[features].drop(columns = ['timestamp'])

    y_train = train_data[targets]
    y_val = val_data[targets]
    y_test = test_data[targets]

    print(f"X_train size: {X_train.shape}")
    print(f"X_val size: {X_val.shape}")
    print(f"X_test size: {X_test.shape}")
    print(f"y_train size: {y_train.shape}")
    print(f"y_val size: {y_val.shape}")
    print(f"y_test size: {y_test.shape}")

    print(f"Train data: {train_data['timestamp'].min()} to {train_data['timestamp'].max()}")
    print(f"Train data: {val_data['timestamp'].min()} to {val_data['timestamp'].max()}")
    print(f"Test data: {test_data['timestamp'].min()} to {test_data['timestamp'].max()}")

    return X_train, X_test, X_val, y_train, y_val, y_test