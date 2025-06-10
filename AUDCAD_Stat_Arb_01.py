# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 16:03:57 2025
@author: AJ
"""
import pandas as pd
import numpy as np
import datetime as dt
import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

# Download VIX data
df = yf.download('AUDCAD=X',start='2021-01-01', end='2025-06-01', interval="1d")

# Clean the data and handle MultiIndex columns if present
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.droplevel(1)

# Calculate moving average and standard deviation
df['moving_average'] = df['Close'].rolling(window=5, min_periods=5).mean()
df['moving_std_dev'] = df['Close'].rolling(window=5, min_periods=5).std()

# Drop rows with NaN values to avoid alignment issues
df = df.dropna()

'''Bollinger Bands Boundary
Since you are trading Currencies - its not very volatile.
Hence, you consider 0.5 as the width.
While trading Crypto and some volatile equity - consider increasing the width of the band
'''
df['upper_band'] = df['moving_average'] + 0.5 * df['moving_std_dev']
df['lower_band'] = df['moving_average'] - 0.5 * df['moving_std_dev']

''' Long Entry'''
df['long_entry'] = df['Close'] < df['lower_band']
df['long_exit'] = df['Close'] > df['moving_average']
df['positions_long'] = np.nan
df.loc[df['long_entry'], 'positions_long'] = 1
df.loc[df['long_exit'], 'positions_long'] = 0
df['positions_long'] = df['positions_long'].ffill().fillna(0)

''' Short Entry'''
df['short_entry'] = df['Close'] > df['upper_band']
df['short_exit'] = df['Close'] < df['moving_average']
df['positions_short'] = np.nan
df.loc[df['short_entry'], 'positions_short'] = -1
df.loc[df['short_exit'], 'positions_short'] = 0
df['positions_short'] = df['positions_short'].ffill().fillna(0)

''' Final Positions'''
df['positions'] = df['positions_long'] + df['positions_short']

''' PnL'''
df['price_difference'] = df['Close'].diff()
df['pnl'] = df['positions'].shift(1) * df['price_difference']
df['cumpnl'] = df['pnl'].cumsum()

''' Returns'''
df['percentage_change'] = df['Close'].pct_change()
df['strategy_returns'] = df['positions'].shift(1) * df['percentage_change']
df['cumulative_returns'] = (df['strategy_returns'] + 1).cumprod()

# Plot results
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(df.index, df['Close'], label='VIX Close', alpha=0.7)
plt.plot(df.index, df['moving_average'], label='Moving Average', alpha=0.8)
plt.plot(df.index, df['upper_band'], label='Upper Band', linestyle='--', alpha=0.6)
plt.plot(df.index, df['lower_band'], label='Lower Band', linestyle='--', alpha=0.6)
plt.title('VIX with Bollinger Bands')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 1, 2)
plt.plot(df.index, df['cumulative_returns'], label='Strategy Cumulative Returns', linewidth=2)
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.title('VIX Trading Strategy - Cumulative Returns')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print some basic statistics
print(f"Total Return: {df['cumulative_returns'].iloc[-1]:.2%}")
print(f"Sharpe Ratio: {df['strategy_returns'].mean() / df['strategy_returns'].std() * np.sqrt(252):.2f}")
print(f"Max Drawdown: {(df['cumulative_returns'] / df['cumulative_returns'].cummax() - 1).min():.2%}")
print(f"Number of trades: {df['positions'].diff().abs().sum() / 2:.0f}")