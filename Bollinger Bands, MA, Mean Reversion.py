import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def moving_average(prices, window):
    return np.convolve(prices, np.ones(window)/window, mode='valid')

def bollinger_bands(prices, window, num_std):
    rolling_mean = pd.Series(prices).rolling(window=window).mean()
    rolling_std = pd.Series(prices).rolling(window=window).std()
    upper_band = rolling_mean + num_std * rolling_std
    lower_band = rolling_mean - num_std * rolling_std
    return rolling_mean, upper_band, lower_band

def mean_reversion_strategy(prices, window, num_std, volume):
    
    rolling_mean, upper_band, lower_band = bollinger_bands(prices, window, num_std)
    
    position = 0
    cash = 10000
    trades = []
    for i in range(len(prices)):
        
        if prices[i] > upper_band[i] and volume[i] > np.mean(volume):
            trade_amount = min(cash, prices[i] * num_std)
            position -= trade_amount / prices[i]
            cash += trade_amount
            trades.append((i, 'sell', trade_amount))
        
        elif prices[i] < lower_band[i] and volume[i] > np.mean(volume):
            trade_amount = min(cash, prices[i] * num_std)
            position += trade_amount / prices[i]
            cash -= trade_amount
            trades.append((i, 'buy', trade_amount))
    return trades


prices = np.random.normal(100, 10, 1000)
volume = np.random.normal(1000, 100, 1000)
trades = mean_reversion_strategy(prices, 20, 2, volume)


plt.plot(prices)
for trade in trades:
    plt.plot(trade[0], prices[trade[0]], 'ro' if trade[1] == 'sell' else 'go')
plt.show()