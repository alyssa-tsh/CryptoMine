import pandas as pd
import numpy as np

def calculate_rsi(data, period=14):
    delta = data.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(close, fast_period=12, slow_period=26, signal_period=9):
    fast_ema = close.ewm(span=fast_period, adjust=False).mean()
    slow_ema = close.ewm(span=slow_period, adjust=False).mean()
    macd = fast_ema - slow_ema
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    hist = macd - signal
    return macd, signal, hist

def calculate_atr(data, period=14):
    high_low = data['High'] - data['Low']
    high_close = (data['High'] - data['Close'].shift()).abs()
    low_close = (data['Low'] - data['Close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1)
    tr_max = tr.max(axis=1)
    atr = tr_max.rolling(window=period, min_periods=1).mean()
    return atr

def calculate_bbands(data, window=20, num_std=2):
    sma = data['Close'].rolling(window=window).mean()
    rolling_std = data['Close'].rolling(window=window).std()
    upper_band = sma + (rolling_std * num_std)
    lower_band = sma - (rolling_std * num_std)
    return upper_band, sma, lower_band

def calculate_obv(data):
    obv = (data['Volume'] * np.sign(data['Close'].diff())).fillna(0)
    return obv.cumsum()

def calculate_stoch(data, k_period=14, d_period=3):
    low_min = data['Low'].rolling(window=k_period).min()
    high_max = data['High'].rolling(window=k_period).max()
    stoch_k = 100 * ((data['Close'] - low_min) / (high_max - low_min))
    stoch_d = stoch_k.rolling(window=d_period).mean()
    return stoch_k, stoch_d

def calculate_adx(data, period=14):
    high = data['High']
    low = data['Low']
    close = data['Close']

    up_move = high.diff()
    down_move = low.diff()

    up_move[up_move < 0] = 0
    down_move[down_move > 0] = 0

    trur = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = trur.rolling(window=period).mean()

    plus_dm = up_move.rolling(window=period).sum()
    minus_dm = down_move.abs().rolling(window=period).sum()

    dx = (np.abs(plus_dm - minus_dm) / (plus_dm + minus_dm)) * 100
    adx = dx.rolling(window=period).mean()
    return adx
