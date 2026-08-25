# =============================================================================
# NumPy Tutorial: Stock Price Risk-Return Analysis
#
# PURPOSE:
# This script serves as a beginner-friendly example of how to use the NumPy
# library for a practical financial analysis task. We will:
#   1. Download historical stock price data for two assets (NVIDIA and S&P 500).
#   2. Convert this data from a pandas DataFrame into NumPy arrays.
#   3. Use NumPy's powerful "vectorized" operations to efficiently calculate
#      daily and monthly returns without using slow `for` loops.
#   4. Calculate the annualized risk (volatility) and return for each asset.
#   5. Create a scatter plot to visualize the risk-return tradeoff.
#
# =============================================================================


# =============================================================================
# 1. SETUP AND DATA DOWNLOAD
# =============================================================================

# --- Import Libraries ---
# - numpy: The core library for numerical computing in Python. We'll use it for our array operations.
# - pandas: Used for data handling. We'll use it as a convenient starting point to get our data.
# - yfinance: A popular library to download financial data from Yahoo Finance.
# - matplotlib.pyplot: The primary library for creating plots and charts.

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt


# --- Download Configuration ---
# Define the stock tickers we are interested in.
# SPY is an ETF that tracks the S&P 500, a benchmark for the whole market.
# NVDA is a single, volatile tech stock (NVIDIA).
tickers = ['SPY', 'NVDA']

# Define the time period for our analysis.
start_date = '2020-01-01'
end_date = '2025-01-01'


# --- Download the Data ---
# We use yfinance to download the historical data for our tickers.
# We are only interested in the 'Adj Close' price, which is the closing price
# adjusted for dividends and stock splits. This is the best price to use for return calculations.
print("Downloading historical stock data...")

# Download stock data with auto_adjust=False to include both 'Close' and 'Adj Close'
adj_close_df = yf.download(tickers, 
                           start=start_date, 
                           end=end_date,
                           auto_adjust=False)['Adj Close']# Ensures 'Adj Close' is included explicitly

# Let's look at the first 5 rows of our downloaded data.
# It's a pandas DataFrame, which is like a spreadsheet, with dates as the index.
print("\n--- Downloaded Data (First 5 Rows) ---")
print(adj_close_df.head())


# =============================================================================
# 2. CONVERTING TO NUMPY ARRAYS
# =============================================================================
# For pure mathematical calculations, NumPy arrays are faster and more memory-efficient
# than pandas DataFrames. We will now extract the raw numbers into arrays.

# Extract the 'Adj Close' prices for each stock as a NumPy array.
# The `.values` attribute of a pandas Series gives us the underlying data as a NumPy array.
# We also use `.dropna()` to remove any rows with missing data just in case.
nvda_prices = adj_close_df['NVDA'].dropna().values
spy_prices = adj_close_df['SPY'].dropna().values

# Let's inspect one of our new arrays to see what it looks like.
print(f"\n--- Inspecting the NVDA NumPy Array ---")
print(f"Data Type: {nvda_prices.dtype}")      # Shows the type of data (usually float64 for prices)
print(f"Shape: {nvda_prices.shape}")          # Shows the dimensions (e.g., (1258,) means 1258 rows, 1 column)
print(f"First 5 prices: {nvda_prices[:5]}")   # Slicing works just like Python lists!


# =============================================================================
# 3. CALCULATING DAILY RETURNS (VECTORIZATION)
# =============================================================================
# This is where NumPy shines. The formula for daily return is: `(Today's Price / Yesterday's Price) - 1`.
# Instead of a slow `for` loop, we can perform this calculation on the entire array at once.
# This is called "vectorization" and it's the core concept of NumPy.

# --- The NumPy "Vectorized" Way ---
# To calculate the percentage change, we divide every element from the 2nd day onwards
# by every element from the 1st day up to the second-to-last day.

# Slicing the array:
# nvda_prices[1:]  --> Creates an array starting from the SECOND element to the end.
# nvda_prices[:-1] --> Creates an array starting from the FIRST element up to the second-to-last.
# These two arrays are now perfectly aligned for our calculation and have the same length.

nvda_daily_returns = nvda_prices[1:] / nvda_prices[:-1] - 1
spy_daily_returns = spy_prices[1:] / spy_prices[:-1] - 1

print("\n--- Daily Return Calculation ---")
print(f"Number of prices for NVDA: {len(nvda_prices)}")
print(f"Number of daily returns for NVDA: {len(nvda_daily_returns)}") # Note: We have one less return than prices
print(f"First 5 daily returns for NVDA: {nvda_daily_returns[:5]}")


# =============================================================================
# 4. CALCULATING MONTHLY RETURNS
# =============================================================================
# This shows how to use the "right tool for the job." Pandas is excellent for
# time-series tasks like resampling, so we'll use it to get monthly prices,
# and then switch back to NumPy for the vectorized return calculation.

# We go back to our original DataFrame to easily resample our daily data to monthly.
# 'M' stands for Month End frequency. `.last()` gets the last price of each month.
monthly_prices_df = adj_close_df.resample('M').last()

print("\n--- Monthly Resampled Data (First 5 Rows) ---")
print(monthly_prices_df.head())

# Convert the new monthly prices to NumPy arrays
nvda_monthly_prices = monthly_prices_df['NVDA'].dropna().values
spy_monthly_prices = monthly_prices_df['SPY'].dropna().values

# Use the same vectorized logic as before to calculate monthly returns.
# This reinforces the power and reusability of the NumPy approach.
nvda_monthly_returns = nvda_monthly_prices[1:] / nvda_monthly_prices[:-1] - 1
spy_monthly_returns = spy_monthly_prices[1:] / spy_monthly_prices[:-1] - 1

print("\n--- First 5 Monthly Returns for NVDA ---")
print(nvda_monthly_returns[:5])


# =============================================================================
# 5. RISK-RETURN ANALYSIS AND PLOTTING
# =============================================================================
# Now we can use our calculated daily returns to create the final plot.
# In finance:
# - Return is measured by the *average* return.
# - Risk is measured by the *standard deviation* of returns (volatility).

# We need to "annualize" our daily stats to make them comparable.
# There are roughly 252 trading days in a year.
# - Annualized Return = Mean Daily Return * 252
# - Annualized Risk (Volatility) = Std Dev of Daily Returns * sqrt(252)

# --- Calculate Annualized Risk and Return using NumPy functions ---
nvda_avg_return = np.mean(nvda_daily_returns) * 252
nvda_risk = np.std(nvda_daily_returns) * np.sqrt(252)

spy_avg_return = np.mean(spy_daily_returns) * 252
spy_risk = np.std(spy_daily_returns) * np.sqrt(252)

print("\n--- Annualized Risk-Return Metrics ---")
# The :.2% formats the number as a percentage with 2 decimal places.
print(f"NVDA: Annualized Return = {nvda_avg_return:.2%}, Annualized Risk = {nvda_risk:.2%}")
print(f"SPY:  Annualized Return = {spy_avg_return:.2%}, Annualized Risk = {spy_risk:.2%}")


# --- Create the Final Plot ---
# Set the plotting style for a professional look.
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(10, 6))

# Create a scatter plot. Risk is on the x-axis, Return is on the y-axis.
ax.scatter(nvda_risk, nvda_avg_return, s=150, label='NVDA', alpha=0.7, edgecolors='black') # s=150 makes the dot bigger
ax.scatter(spy_risk, spy_avg_return, s=150, label='SPY', alpha=0.7, edgecolors='black')

# Add text annotations next to each point for clarity.
# This makes the plot easier to read without a legend.
ax.annotate('NVDA', (nvda_risk, nvda_avg_return), textcoords="offset points", xytext=(0,15), ha='center', fontsize=12)
ax.annotate('SPY', (spy_risk, spy_avg_return), textcoords="offset points", xytext=(0,15), ha='center', fontsize=12)

# Add titles and labels for clarity.
ax.set_title('Risk vs. Return Tradeoff (Last 5 Years)', fontsize=16, pad=20)
ax.set_xlabel('Annualized Risk (Volatility)', fontsize=12)
ax.set_ylabel('Annualized Return', fontsize=12)

# Format the axes to display values as percentages.
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.0%}'.format(x)))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.0%}'.format(x)))

# Add a grid and show the final plot.
plt.grid(True)
plt.show()