# ==============================================================================
# 1. SETUP: IMPORT LIBRARIES
# ==============================================================================
# Think of this section as gathering your tools before starting a project. Each
# "import" statement brings in a pre-built set of tools (a "library") that helps
# us perform a specific task without having to write the code from scratch.

# The 'requests' library allows our Python script to act like a web browser,
# enabling it to download the content of web pages.
import requests

# 'BeautifulSoup' is a specialized tool for parsing HTML and XML documents.
# It takes the messy raw HTML code from a webpage and turns it into a structured
# object that's easy to navigate and search.
from bs4 import BeautifulSoup

# 'pandas' is the most essential library for data science in Python. It provides
# the "DataFrame," a powerful object that lets us work with data in tables (like
# an Excel spreadsheet, but with much more power). We give it the nickname 'pd'.
import pandas as pd

# The 'datetime' library provides tools for working with dates and times. We
# specifically import the 'date' object to define our start and end dates.
from datetime import date

# The 'yfinance' library is a popular and easy-to-use tool for downloading
# historical market data from Yahoo! Finance.
import yfinance as yf

# 'numpy' is the fundamental library for numerical computing in Python. It's
# especially good at performing mathematical operations on large arrays of numbers
# very quickly. We use it here for the square root function.
import numpy as np

# 'matplotlib' is the foundational plotting library in Python. It gives us fine-grained
# control over every aspect of a visualization. We give it the nickname 'plt'.
import matplotlib.pyplot as plt

# 'seaborn' is built on top of Matplotlib and provides a high-level interface for
# creating beautiful and informative statistical graphics. It makes complex plots easier.
import seaborn as sns

# 'statsmodels' is a powerful library for estimating and interpreting statistical models.
# We import its 'formula.api', which allows us to define our regression model using a
# simple, readable formula syntax (like 'Y ~ X'), similar to the R programming language.
import statsmodels.formula.api as smf

# The 'time' library gives us access to time-related functions. We will use it to
# measure how long our regression analysis takes to run.
import time


# ==============================================================================
# 2. DATA GATHERING: S&P 500 COMPANIES AND STOCK PRICES
# ==============================================================================
# The goal of this section is to collect our two raw ingredients:
# 1. An up-to-date list of all companies in the S&P 500 index and their industries.
# 2. The historical daily stock prices for each of those companies, plus the SPY ETF.

# --- Step 2.1: Get the list of S&P 500 companies from Wikipedia ---

# We store the URL of the Wikipedia page in a variable for easy access.
url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
# It's considered polite practice ("polite scraping") to identify our script when
# we request data from a website. We do this by setting a 'User-Agent' in the request headers.
headers = {'User-Agent': 'Stock Analyzer/1.0 (kc@example.com)'}
# We use the 'requests.get()' function to send an HTTP GET request to the URL,
# which is like asking the server for the content of the page. The server's
# response is stored in the 'response' object.
response = requests.get(url, headers=headers)
# We then use 'BeautifulSoup' to parse the raw HTML content of the page ('response.content').
# 'html.parser' is the built-in Python parser we'll use to process the HTML.
soup = BeautifulSoup(response.content, 'html.parser')
# 'pandas.read_html()' is a powerful function that scans the provided HTML and
# automatically extracts any tables it finds. It returns a list of DataFrames.
# Since the first table on the page ([0]) is the one we want, we select it.
constituents_table = pd.read_html(str(soup.find('table')))[0]

# --- Step 2.2: Clean the constituents data ---
# We use a "method chain" here, where each operation is on a new line. This makes
# the sequence of data cleaning steps clear and readable, like a recipe.
constituents_df = (
    constituents_table
    # The '.rename()' method allows us to change the names of the columns. We
    # choose simpler, script-friendly names (lowercase, no spaces).
    .rename(columns={'Symbol': 'ticker', 'GICS Sector': 'industry'})
    # After renaming, we select ONLY the columns we need for our analysis using
    # double square brackets. This keeps our DataFrame tidy and efficient.
    [['ticker', 'industry']]
)
print("--- S&P 500 Constituents (Sample) ---")
# '.head()' is a useful method to display the first 5 rows of a DataFrame,
# allowing us to quickly check that our data was loaded and cleaned correctly.
print(constituents_df.head())

# --- Step 2.3: Download historical stock prices from Yahoo Finance ---
# We need a list of all ticker symbols to download. We get this by taking the
# 'ticker' column from our DataFrame and converting it to a Python list with '.tolist()'.
# We then add 'SPY' to this list, as it's an ETF that tracks the S&P 500 and will
# serve as our market benchmark for the regressions.
tickers_to_download = constituents_df['ticker'].tolist() + ['SPY']
# We dynamically set the date range. 'date.today()' gets the current date.
end_date = date.today()
# The start date is set to January 1st, five years prior to the current year.
start_date = date(end_date.year - 5, 1, 1)

print(f"\nDownloading historical stock data from {start_date} to {end_date}...")
# This is the main function call to yfinance. It downloads all the data we need in one go.
adj_close_df = yf.download(
    tickers_to_download, # The list of tickers we want.
    start=start_date,    # The start of our historical date range.
    end=date.today(),      # The end of our date range.
    auto_adjust=False,   # We set this to False because we want to specifically select the 'Adj Close' column.
    progress=False       # Hides the download progress bar for a cleaner output.
)['Adj Close'] # From the downloaded data, we select only the 'Adj Close' (Adjusted Close) price.
               # This price is adjusted for dividends and stock splits, making it the most accurate
               # price to use for calculating historical returns.
print("Download complete.")


# ==============================================================================
# 3. DATA PROCESSING: CALCULATE AND TRANSFORM MONTHLY RETURNS
# ==============================================================================
# The goal here is to convert the raw daily prices into the monthly returns
# that we will use in our regression analysis. Daily prices are too noisy for
# this type of analysis, so we smooth them out by using monthly data points.

# We use another method chain to perform the calculation in a series of clear steps.
monthly_returns_wide = (
    adj_close_df
    # Step A: '.resample("M")' groups the daily data into monthly bins.
    # '.last()' then selects the LAST valid price for each month. This gives us
    # a DataFrame of month-end prices for every stock.
    .resample('M').last()

    # Step B: '.pct_change()' calculates the percentage change from the previous row
    # to the current row. Since our rows are now months, this automatically computes
    # the monthly return for each stock.
    .pct_change()

    # Step C: The very first row of the result will be empty (NaN) because there's no
    # prior month to calculate a return from. '.iloc[1:]' selects all rows from
    # the second row to the end, effectively dropping that first empty row.
    .iloc[1:]
    
    # Step D: We rename the 'SPY' column to 'SPY_Return'. This makes our
    # regression formula later more readable and explicit.
    .rename(columns={'SPY': 'SPY_Return'})
)
print("\n--- Monthly Returns (Wide Format, with NaNs preserved) ---")
# We print the head of the resulting DataFrame to see what our monthly returns look like.
# Note that there will be 'NaN' (Not a Number) values for stocks that didn't exist
# or weren't in the index for the full 5-year period. This is expected.
print(monthly_returns_wide.head())


# ==============================================================================
# 4. ANALYSIS: RUN REGRESSION FOR EACH STOCK AGAINST THE MARKET (SPY)
# ==============================================================================
# This is the analytical core of the script. The goal is to measure the relationship
# between each individual stock's returns and the overall market's returns (SPY).
# We do this by fitting a linear regression model for each stock.

# We define a function to perform the regression. A function is a reusable block of
# code that performs a specific task. This is efficient because we can write the
# logic once and then apply it to hundreds of stocks.
def calculate_regression_stats(stock_data, market_return_col):
    # This function takes two inputs: 'stock_data' (the monthly returns for one
    # stock) and 'market_return_col' (the monthly returns for SPY).

    # We combine the stock's returns and the market's returns into a single, temporary
    # DataFrame. 'axis=1' means we are combining them as columns.
    # '.dropna()' is CRITICAL here. It removes any rows where EITHER the stock OR SPY
    # has a missing value. This ensures we only run the regression on valid, paired data points.
    regression_df = pd.concat([stock_data, market_return_col], axis=1).dropna()

    # This is a data quality check. A reliable regression requires a minimum number
    # of data points. We set a threshold of 35 months (almost 3 years). If there are
    # fewer than 35 valid months, we skip the calculation and return empty results.
    if len(regression_df) < 35:
        return pd.Series({'Alpha': np.nan, 'Beta': np.nan, 'R_Squared': np.nan, 'Residual_SE': np.nan})

    # This line constructs the regression formula as a string.
    # The `~` means "is explained by". So, "Stock Return is explained by SPY_Return".
    # We must wrap the stock's name in `Q()` because the ticker symbol (e.g., "AAPL")
    # is stored in a variable, and `Q()` tells the model to look for a column with that name.
    model_formula = f'Q("{stock_data.name}") ~ SPY_Return'

    # Here, we use statsmodels to define and fit the Ordinary Least Squares (OLS) regression model.
    # '.fit()' runs the actual calculation to find the best-fit line.
    model = smf.ols(formula=model_formula, data=regression_df).fit()

    # After the model is fit, we extract the key results.
    # Alpha is the model's intercept. It represents the stock's theoretical excess return if the market had a 0% return.
    alpha = model.params.get('Intercept', np.nan)
    # Beta is the coefficient of our market variable. It measures the stock's volatility relative to the market.
    beta = model.params.get('SPY_Return', np.nan)
    # R-squared measures what percentage of the stock's price movements can be explained by the market's movements.
    r_squared = model.rsquared
    # Residual Standard Error measures the typical size of the model's prediction errors. A smaller value is better.
    residual_se = np.sqrt(model.mse_resid)

    # The function returns all the calculated statistics packaged neatly as a pandas Series (which will become a row in our final results).
    return pd.Series({'Alpha': alpha, 'Beta': beta, 'R_Squared': r_squared, 'Residual_SE': residual_se})

print("\n--- Fitting Regression Models for Each Stock ---")
# We start a high-precision timer right before the main calculation begins.
start_time = time.perf_counter()

# This method chain orchestrates the entire analysis for all stocks.
regression_results = (
    # Step A: We start with our monthly returns, but drop the 'SPY_Return' column
    # because we don't want to run a regression of SPY against itself.
    monthly_returns_wide.drop(columns='SPY_Return')
    # Step B: '.apply()' is the workhorse. It iterates through EVERY column (i.e., every stock)
    # of the DataFrame and runs our 'calculate_regression_stats' function on it.
    .apply(calculate_regression_stats, market_return_col=monthly_returns_wide[['SPY_Return']])
    # Step C: The result of '.apply()' has tickers as columns. '.T' transposes the
    # DataFrame, flipping it so that tickers become the rows, a more standard format.
    .T
    # Step D: We merge our regression results with our original industry information.
    # This is like a VLOOKUP in Excel, matching on the ticker symbol to add the correct industry to each row.
    .merge(constituents_df.set_index('ticker'), left_index=True, right_index=True)
    # Step E: A final cleanup step to remove any stocks where the regression failed
    # (e.g., because they had too little data and our function returned NaNs).
    .dropna(subset=['Beta'])
)
# We stop the timer immediately after the calculation is finished.
end_time = time.perf_counter()
# We calculate the duration and print a summary of the work done.
duration = end_time - start_time
num_regressions = len(regression_results)
print(f"Successfully fit {num_regressions} regressions in {duration:.2f} seconds.")
print("\n--- Final Regression Results (Sample) ---")
# We print the first few rows of our final, combined results table.
print(regression_results.head())


# ==============================================================================
# 5. VISUALIZATION 1: SORTED & FLIPPED BOXPLOT (BLACK & WHITE)
# ==============================================================================
# The goal of this visualization is to compare the distribution of Betas across
# all the different industries in a clear and concise way.

print("\n--- Generating Black & White Sorted Beta Boxplot ---")

# --- Step 5.1: Prepare the sorting order for the plot ---
# To make the plot more insightful, we want to sort the industries from highest
# average beta to lowest. This makes it easy to see which sectors are most volatile.
industry_order = (
    regression_results
    .groupby('industry')       # This groups all the stocks by their industry, like a pivot table.
    ['Beta']                   # We select the 'Beta' column to perform a calculation on.
    .mean()                    # We calculate the average (mean) Beta for each industry group.
    .sort_values(ascending=False) # We sort these average Betas from highest to lowest.
    .index                     # Finally, we get the list of industry names in this new sorted order.
)

# --- Step 5.2: Create the plot ---
# Set a clean visual style for the plot with visible axis ticks.
sns.set_style("ticks")
# Create the "canvas" for our plot and set its size in inches (width, height).
plt.figure(figsize=(12, 10))
# This is the main command to create the boxplot using the Seaborn library.
# A boxplot is a standardized way of displaying the distribution of data based
# on a five-number summary (minimum, first quartile, median, third quartile, and maximum).
sns.boxplot(
    data=regression_results, # The DataFrame containing our data.
    x='Beta',                # The variable whose distribution we want to see (will be on the horizontal axis).
    y='industry',            # The variable we use to group the data (will be on the vertical axis).
    order=industry_order,    # This tells Seaborn to draw the industries in our custom sorted order.
    color="white"            # We set the fill color to white for a clean, black-and-white look.
)
# This adds a vertical dashed line to the plot at x=1. This is a crucial reference
# point, as a Beta of 1 represents the volatility of the market itself.
plt.axvline(x=1, color='red', linestyle='--', alpha=0.8)
# We create a dynamic title that includes the actual start and end dates of our analysis.
title_text = f"Beta Estimation, {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
# We set the title of the plot, making it larger and bold for emphasis.
plt.title(title_text, fontsize=16, fontweight='bold')
# We add descriptive labels to the x and y axes.
plt.xlabel('Beta (Volatility vs. Market)', fontsize=12)
plt.ylabel('Industry (GICS Sector)', fontsize=12)
# This command automatically adjusts plot parameters to give a tight layout,
# preventing labels from overlapping.
plt.tight_layout()
# This final command displays the plot we have built.
plt.show()

# ==============================================================================
# 6. SUMMARY STATISTICS
# ==============================================================================
print("\n--- Beta Summary Statistics by Industry ---")
# Group the results by industry and calculate summary stats for Beta
beta_summary = regression_results.groupby('industry')['Beta'].agg(
    ['count', 'mean', 'std', 'min', 'max']
).round(3)
print(beta_summary)

print("\n--- Stocks with Highest Betas (Most Volatile) ---")
print(regression_results.nlargest(10, 'Beta')[['Beta', 'industry']])

print("\n--- Stocks with Lowest Betas (Least Volatile) ---")
print(regression_results.nsmallest(10, 'Beta')[['Beta', 'industry']])