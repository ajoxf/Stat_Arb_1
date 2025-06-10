# -*- coding: utf-8 -*-
"""
Flexible Bollinger Bands Mean Reversion Strategy
Works for any currency pair and time frame

@author: AJ
"""

import pandas as pd
import numpy as np
import datetime as dt
import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

# ============================================================================
# CONFIGURATION SECTION - MODIFY THESE PARAMETERS
# ============================================================================

# Currency pair configuration
CURRENCY_PAIR = 'GBPINR=X'  # Change this to any pair: AUDCAD=X, EURUSD=X, GBPJPY=X, etc.
PAIR_NAME = 'GBPINR'        # Display name for charts and output

# Time frame configuration
START_DATE = '2016-01-01'   # Start date (YYYY-MM-DD)
END_DATE = '2025-06-10'     # End date (YYYY-MM-DD)

# Strategy parameters
MOVING_AVERAGE_WINDOW = 5   # Moving average period
STD_DEV_MULTIPLIER = 0.5    # Bollinger Bands width (0.5 for FX, higher for crypto/stocks)

# ============================================================================
# STRATEGY IMPLEMENTATION - NO CHANGES NEEDED BELOW
# ============================================================================

print(f"=== BOLLINGER BANDS STRATEGY ANALYSIS ===")
print(f"Currency Pair: {PAIR_NAME}")
print(f"Period: {START_DATE} to {END_DATE}")
print(f"Strategy Parameters: MA={MOVING_AVERAGE_WINDOW}, STD={STD_DEV_MULTIPLIER}")
print("=" * 50)

# Download currency data
try:
    df = yf.download(CURRENCY_PAIR, start=START_DATE, end=END_DATE, interval="1d")
    print(f"✓ Successfully downloaded {len(df)} days of data for {PAIR_NAME}")
except Exception as e:
    print(f"✗ Error downloading data: {e}")
    exit()

# Clean the data and handle MultiIndex columns if present
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.droplevel(1)

# Calculate moving average and standard deviation
df['moving_average'] = df['Close'].rolling(window=MOVING_AVERAGE_WINDOW, min_periods=MOVING_AVERAGE_WINDOW).mean()
df['moving_std_dev'] = df['Close'].rolling(window=MOVING_AVERAGE_WINDOW, min_periods=MOVING_AVERAGE_WINDOW).std()

# Drop rows with NaN values to avoid alignment issues
df = df.dropna()

# Bollinger Bands Boundary
df['upper_band'] = df['moving_average'] + STD_DEV_MULTIPLIER * df['moving_std_dev']
df['lower_band'] = df['moving_average'] - STD_DEV_MULTIPLIER * df['moving_std_dev']

# Long Entry Logic
df['long_entry'] = df['Close'] < df['lower_band']
df['long_exit'] = df['Close'] > df['moving_average']
df['positions_long'] = np.nan
df.loc[df['long_entry'], 'positions_long'] = 1
df.loc[df['long_exit'], 'positions_long'] = 0
df['positions_long'] = df['positions_long'].ffill().fillna(0)

# Short Entry Logic
df['short_entry'] = df['Close'] > df['upper_band']
df['short_exit'] = df['Close'] < df['moving_average']
df['positions_short'] = np.nan
df.loc[df['short_entry'], 'positions_short'] = -1
df.loc[df['short_exit'], 'positions_short'] = 0
df['positions_short'] = df['positions_short'].ffill().fillna(0)

# Final Positions
df['positions'] = df['positions_long'] + df['positions_short']

# PnL Calculations
df['price_difference'] = df['Close'].diff()
df['pnl'] = df['positions'].shift(1) * df['price_difference']
df['cumpnl'] = df['pnl'].cumsum()

# Returns Calculations
df['percentage_change'] = df['Close'].pct_change()
df['strategy_returns'] = df['positions'].shift(1) * df['percentage_change']
df['cumulative_returns'] = (df['strategy_returns'] + 1).cumprod()

# Buy and Hold Strategy
df['buy_hold_returns'] = df['percentage_change']
df['buy_hold_cumulative'] = (df['buy_hold_returns'] + 1).cumprod()

# Generate Plots
plt.figure(figsize=(15, 12))

# Price Change Over Time (New Graph)
plt.subplot(3, 1, 1)
plt.plot(df.index, df['Close'], label=f'{PAIR_NAME} Price', linewidth=1.5, color='#2E86AB')
plt.title(f'{PAIR_NAME} Price Over Time ({START_DATE} to {END_DATE})')
plt.ylabel('Price')
plt.legend()
plt.grid(True, alpha=0.3)

# Price and Bollinger Bands
plt.subplot(3, 1, 2)
plt.plot(df.index, df['Close'], label=f'{PAIR_NAME} Close', alpha=0.7)
plt.plot(df.index, df['moving_average'], label='Moving Average', alpha=0.8)
plt.plot(df.index, df['upper_band'], label='Upper Band', linestyle='--', alpha=0.6)
plt.plot(df.index, df['lower_band'], label='Lower Band', linestyle='--', alpha=0.6)
plt.title(f'{PAIR_NAME} with Bollinger Bands')
plt.ylabel('Price')
plt.legend()
plt.grid(True, alpha=0.3)

# Cumulative Returns Comparison
plt.subplot(3, 1, 3)
plt.plot(df.index, df['cumulative_returns'], label='Strategy Cumulative Returns', linewidth=2)
plt.plot(df.index, df['buy_hold_cumulative'], label='Buy & Hold Cumulative Returns', linewidth=2, alpha=0.8)
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.title(f'{PAIR_NAME} Trading Strategy vs Buy & Hold - Cumulative Returns')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Calculate and Display Performance Metrics
print(f"\n=== STRATEGY PERFORMANCE COMPARISON - {PAIR_NAME} ===")
print(f"Strategy Total Return: {(df['cumulative_returns'].iloc[-1] - 1) * 100:.2f}%")
print(f"Buy & Hold Total Return: {(df['buy_hold_cumulative'].iloc[-1] - 1) * 100:.2f}%")
print(f"Strategy Sharpe Ratio: {df['strategy_returns'].mean() / df['strategy_returns'].std() * np.sqrt(252):.2f}")
print(f"Buy & Hold Sharpe Ratio: {df['buy_hold_returns'].mean() / df['buy_hold_returns'].std() * np.sqrt(252):.2f}")
print(f"Strategy Max Drawdown: {(df['cumulative_returns'] / df['cumulative_returns'].cummax() - 1).min():.2%}")
print(f"Buy & Hold Max Drawdown: {(df['buy_hold_cumulative'] / df['buy_hold_cumulative'].cummax() - 1).min():.2%}")
print(f"Number of trades: {df['positions'].diff().abs().sum() / 2:.0f}")
print(f"Strategy Volatility (annualized): {df['strategy_returns'].std() * np.sqrt(252):.2%}")
print(f"Buy & Hold Volatility (annualized): {df['buy_hold_returns'].std() * np.sqrt(252):.2%}")

# Additional Analysis
years = (pd.to_datetime(END_DATE) - pd.to_datetime(START_DATE)).days / 365.25
trades_per_year = (df['positions'].diff().abs().sum() / 2) / years
print(f"\n=== ADDITIONAL METRICS ===")
print(f"Analysis Period: {years:.1f} years")
print(f"Average Trades per Year: {trades_per_year:.1f}")
print(f"Strategy Annualized Return: {((df['cumulative_returns'].iloc[-1]) ** (1/years) - 1) * 100:.2f}%")
print(f"Buy & Hold Annualized Return: {((df['buy_hold_cumulative'].iloc[-1]) ** (1/years) - 1) * 100:.2f}%")

print(f"\n✓ Analysis completed for {PAIR_NAME} from {START_DATE} to {END_DATE}")