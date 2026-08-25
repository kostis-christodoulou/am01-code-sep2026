# ==============================================================================
# 1. SETUP: IMPORT LIBRARIES
# ==============================================================================
# Import necessary libraries for data handling, analysis, and plotting.

import pandas as pd
from datetime import date
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import seaborn as sns
import statsmodels.formula.api as smf


# ==============================================================================
# 2. DATA GATHERING: DOWNLOAD AAPL AND SPY PRICES
# ==============================================================================
# The goal of this section is to download the historical daily stock prices
# for Apple (AAPL) and the S&P 500 ETF (SPY).

# We define the tickers we want to download.
tickers_to_download = ['AAPL', 'SPY']

# We dynamically set the date range. 'date.today()' gets the current date.
end_date = date.today()
# The start date is set to January 1st, five years prior to the current year.
start_date = date(end_date.year - 5, 1, 1)

print(f"--- Downloading historical stock data for {tickers_to_download} ---")

print(f"Date Range: {start_date} to {end_date}\n")

# We download the 'Adjusted Close' prices from Yahoo Finance for our tickers.
# The 'Adj Close' price is adjusted for dividends and stock splits, which is
# best for calculating historical returns.
adj_close_df = yf.download(
    tickers_to_download, # The list of tickers.
    start=start_date,    # The start of our date range.
    end=end_date,        # The end of our date range.
    auto_adjust=False,   # We set this to False because we want to specifically select the 'Adj Close' column.
    progress=True        # Shows the download progress bar.
)['Adj Close']

print("--- Sample of Daily Adj Closing Prices ---")
print(adj_close_df.head())



# ==============================================================================
# 3. DATA PROCESSING: CALCULATE MONTHLY RETURNS
# ==============================================================================
# We convert the daily prices into monthly returns. Using monthly data is standard
# for this type of regression as it reduces the "noise" of daily price swings.

monthly_returns = (
    adj_close_df
    # 1. Resample the daily data to get the last price of each month.
    .resample('M').last()
    # 2. Calculate the percentage change from one month-end to the next.
    .pct_change()
    # 3. Drop the first row, which will be NaN (Not a Number) since there's
    #    no previous month to calculate a return from.
    .dropna()
)

print("\n--- Sample of Monthly Returns ---")
print(monthly_returns.head())


# ==============================================================================
# 4. ANALYSIS: FIT THE REGRESSION MODEL (Y=AAPL, X=SPY)
# ==============================================================================
# This is the core of the analysis where we fit the Ordinary Least Squares (OLS)
# regression model to understand the relationship between AAPL's returns and SPY's returns.

# We define our regression formula using the column names.
# The formula 'AAPL ~ SPY' means we are modeling AAPL's returns as a function of SPY's returns.
model_formula = 'AAPL ~ SPY'

# We use statsmodels to define and fit the regression model.
# The .fit() method performs the calculation to find the best-fit line.
model = smf.ols(formula=model_formula, data=monthly_returns).fit()

# The .summary() method provides a comprehensive table of the regression results,
# including the R-squared, coefficients (Alpha and Beta), and their statistical significance.
print("\n--- Regression Results: AAPL ~ SPY ---")
print(model.summary())
print(f"Residual SE: {model.scale ** 0.5:.3f}")  # Residual standard error is the square root of the scale (MSE of residuals); for stocks it's a measure of the specific risk of AAPL not explained by SPY.

print(model.summary2().tables[1])
print("R-squared:", model.rsquared.round(4))
print("Adjusted R-squared:", model.rsquared_adj.round(4))
print(f"Residual SE: {model.scale ** 0.5:.3f}")  # Residual standard error is the square root of the scale (MSE of residuals).

from statsmodels.stats.anova import anova_lm
anova_results = anova_lm(model, typ=2)  # typ=2 is Type II sum of squares, popular choice
print(anova_results)

# Extracting Alpha and Beta for easier interpretation
alpha = model.params['Intercept']
beta = model.params['SPY']
r_squared = model.rsquared
residual_se = model.scale ** 0.5

print(f"\n--- Key Statistics ---")
print(f"Alpha (Intercept): {alpha:.4f}")
print(f"Beta (Coefficient for SPY): {beta:.4f}")
print(f"R-squared: The market (SPY) explains {r_squared:.4f} of AAPL's volatility.")
print(f"Residual Standard Error: {residual_se:.4f}")
print("\nInterpretation:")
print(f"An R-squared of {r_squared:.2%} means that {r_squared:.2%} of AAPL's monthly price movements can be explained by the movements in the S&P 500 (SPY).")
print(f"A Residual Standard Error of {residual_se:.4f} is a measure of the specific risk of AAPL not explained by SPY.")


# ==============================================================================
# 5. VISUALIZATION: PLOT THE REGRESSION
# ==============================================================================
# A visual check is the best way to understand the relationship. We create a
# scatter plot of the monthly returns and overlay the regression line.

print("\n--- Generating Regression Plot ---")

# Set the style and create the plot canvas.
sns.set_style("whitegrid") # Shows grid lines with ticks on axes
plt.figure(figsize=(10, 6))

# Use seaborn's regplot to create a scatter plot and automatically fit/draw the regression line.
capm_plot = sns.regplot(
    x='SPY',
    y='AAPL',
    data=monthly_returns,
    color='black',        # Color of the data points.
    line_kws={"color": "blue"}, # Make the regression line blue to stand out.
    scatter_kws={"alpha": 0.6} # Make the data points slightly transparent.
)

# Format X and Y axes as percentages (values from 0 to 1 will show as 0% to 100%)
capm_plot.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
capm_plot.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))

# Add a descriptive title and labels.
plt.title(f'Monthly Returns: AAPL vs. SPY\n({start_date.strftime("%b %Y")} to {end_date.strftime("%b %Y")})', fontsize=16)
plt.xlabel('SPY Monthly Return', fontsize=12)
plt.ylabel('AAPL Monthly Return', fontsize=12)

# Display the plot.
plt.show()