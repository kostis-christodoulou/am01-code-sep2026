# =============================================================================
# 1. IMPORT LIBRARIES
# =============================================================================
# We start by importing all the tools (libraries) we will need for our analysis.

# - pandas: The primary library for working with data tables (called DataFrames).
# - numpy: A fundamental library for numerical and mathematical operations.
# - seaborn & matplotlib.pyplot: Powerful libraries for creating beautiful statistical visualizations.
# - scipy.stats: A part of a scientific library that gives us advanced statistical functions, like the t-distribution.
# - skimpy: A handy library for creating a quick, detailed summary of our dataset.

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from skimpy import skim


# =============================================================================
# 2. LOAD AND EXPLORE THE DATA
# =============================================================================
# Here, we load our dataset from a file and take a first look to understand its structure.

# Load the dataset from a CSV (Comma-Separated Values) file into a pandas DataFrame.
# We assume the data file is in a subfolder named 'data'.
gss = pd.read_csv('data/gss_extract_2022.csv')

# Display the first 5 rows of the DataFrame to get a quick preview of the columns and data.
print("--- First 5 Rows of the GSS Dataset ---")
print(gss.head())

# Use the `skim` function from the skimpy library to generate a comprehensive summary.
# This is great for seeing data types, missing values, and a mini-histogram for each variable.
print("\n--- Dataset Summary with skimpy ---")
skim(gss)


# =============================================================================
# 3. VISUALIZE THE DISTRIBUTION OF A SINGLE VARIABLE (INCOME)
# =============================================================================
# A key step in analysis is to understand the shape of your data. We'll plot the
# distribution of household income (`realinc`) to see if it's symmetric or skewed.

# --- Step 3.1: Initial Plot Setup ---

# Set a visual theme for all our plots. "ticks" provides a clean white background
# with black borders around the plot, similar to a classic scientific chart.
sns.set_style("ticks")

# Create a "figure" (the overall window) and an "axes" (the actual plot inside it).
# This is the standard, flexible way to create plots. `figsize` controls the size in inches.
fig, ax = plt.subplots(figsize=(10, 6))


# --- Step 3.2: Create the Density Plot ---

# The column `realinc` records family income, converted to 1986 dollars.
# We use `sns.kdeplot` (Kernel Density Estimate) to draw a smooth line that
# represents the distribution of this income data.
sns.kdeplot(
    data=gss,                # The DataFrame containing our data.
    x=gss['realinc'] / 1000, # The data we want to plot. We divide by 1000 to show income in thousands of dollars.
    ax=ax,                   # Tells seaborn to draw the plot on the `ax` object we created above.
    fill=True,               # Shades the area under the curve, making the shape easier to see.
    alpha=0.6,               # Makes the shaded area semi-transparent.
    linewidth=2,             # Makes the outline of the curve thicker.
    cut=0                    # Prevents the curve from extending to unrealistic values (like negative income).
)


# --- Step 3.3: Add Annotations to Tell a Story ---

# A great plot highlights key information. For skewed data like income, comparing
# the mean and the median is very insightful.
# The median is the "middle" value, while the mean is the "average" value, which
# can be pulled higher by a few very wealthy households.

# First, calculate the mean and median of the income data.
mean_income = (gss['realinc'] / 1000).mean().round(2)
median_income = (gss['realinc'] / 1000).median().round(2)   


# Add a vertical line for the Mean.
ax.axvline(
    x=mean_income,
    color='red',        # Use a distinct color to make it stand out.
    linestyle=':',      # A dotted line style.
    linewidth=2,
    label=f'Mean: ${mean_income:,.0f}k' # A descriptive label for the legend.
)

# Add a text label next to the line to make it even clearer.
# The coordinates (x, y) are chosen manually to place the text nicely.
ax.text(mean_income * 1.1, 0.012, f'Mean: ${mean_income:,.0f}k',
        fontsize=12, color='red')

# Add a vertical line for the Median.
ax.axvline(
    x=median_income,
    color='black',
    linestyle='--',     # A dashed line style to distinguish it from the mean.
    linewidth=2,
    label=f'Median: ${median_income:,.0f}k' # A descriptive label for the legend.
)

# Add a text label next to the median line.
ax.text(median_income * 1.1, 0.005, f'Median: ${median_income:,.0f}k',
        fontsize=12, color='black')


# --- Step 3.4: Finalize and Show the Plot ---

# Add descriptive titles and labels to the plot.
ax.set_title('Distribution of Real Household Income (GSS Data)', fontsize=16, pad=20)
ax.set_xlabel('Family Income (in thousands of 1986 $)', fontsize=12)
ax.set_ylabel('Probability Density (PDF)', fontsize=12)

# Display the legend, which will automatically use the `label` arguments we defined above.
ax.legend()

# Ensure all plot elements (titles, labels) fit nicely within the figure frame.
plt.tight_layout()

# Finally, display the plot.
plt.show()


# =============================================================================
# 5. THE BOOTSTRAPPING FUNCTION
# =============================================================================
# This is the core of our script. It's a general-purpose function that can
# perform a bootstrap simulation on any numerical column in a DataFrame.
# Bootstrapping allows us to estimate the uncertainty of a statistic (like the mean)
# by resampling from our own data, without making strong assumptions about the
# underlying population distribution.

def bootstrap_simulation(data, variable, statistic_func, num_iterations=1000):
    """
    Performs a bootstrap simulation to estimate the confidence interval of a statistic.
    Args:
        data (pd.DataFrame): The DataFrame containing the data.
        variable (str): The name of the column (variable) of interest.
        statistic_func (function): A function to compute the statistic (e.g., np.mean, np.median).
        num_iterations (int): The number of bootstrap samples to generate.
    Returns:
        pd.Series: A series of the calculated statistics from each bootstrap sample.
        tuple: A tuple containing the lower and upper bounds of the 95% percentile CI.
    """
    # --- Step 5.1: Prepare the original data ---
    # Select the column of interest and drop any missing values to be safe.
    original_series = data[variable].dropna()
    original_size = len(original_series)

    # --- Step 5.2: Run the bootstrap resampling loop ---
    # We will create many new "resamples" and calculate our statistic on each one.
    bootstrap_stats = []
    for i in range(num_iterations):
        # Create a bootstrap sample by sampling WITH REPLACEMENT from the original data.
        # `replace=True` is the key. It means we can pick the same data point more than once.
        # The sample size is the same as the original.
        bootstrap_sample = original_series.sample(n=original_size, replace=True)

        # Calculate the statistic of interest on this new bootstrap sample.
        stat = statistic_func(bootstrap_sample)

        # Append the calculated statistic to our list.
        bootstrap_stats.append(stat)

    # Convert the list of stats into a pandas Series for easier analysis.
    # This series of values is called the "bootstrap distribution".
    bootstrap_stats_series = pd.Series(bootstrap_stats)

    # --- Step 5.3: Calculate the percentile confidence interval ---
    # The 95% confidence interval is found by taking the 2.5th and 97.5th percentiles
    # of our bootstrap distribution. This range contains the central 95% of our simulated values.
    ci_low = bootstrap_stats_series.quantile(0.025)
    ci_high = bootstrap_stats_series.quantile(0.975)

    # Calculate the statistic on the original, un-resampled data for comparison.
    original_stat = statistic_func(original_series)

    # --- Step 5.4: Plot the results ---
    # Set the plotting style (can be different from the global style).
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the histogram of the bootstrap distribution.
    sns.histplot(bootstrap_stats_series, ax=ax, kde=True, stat="density",
                 label="Bootstrap Distribution of " + statistic_func.__name__)

    # Add a vertical line for the statistic calculated from our original sample.
    ax.axvline(original_stat, color='black', linestyle='-', linewidth=2,
               label=f'Original {statistic_func.__name__}: {original_stat:.2f}')

    # Shade the 95% confidence interval area on the plot.
    ax.axvspan(ci_low, ci_high, color='red', alpha=0.2,
               label=f'95% Bootstrap CI ({ci_low:.2f} to {ci_high:.2f})')

    # If the statistic is the mean, we can compare the bootstrap CI to the
    # traditional formula-based CI (using the t-distribution). This is a great check.
    if statistic_func == np.mean:
        # Calculate the standard error and the t-critical value.
        std_error = original_series.std() / np.sqrt(original_size)
        degrees_of_freedom = original_size - 1
        t_critical = stats.t.ppf(0.975, df=degrees_of_freedom)
        formula_ci_low = original_stat - t_critical * std_error
        formula_ci_high = original_stat + t_critical * std_error

        # Add orange dotted vertical lines for the formula-based CI to the plot.
        ax.axvline(formula_ci_low, color='orange', linestyle=':', linewidth=2.5,
                   label=f'Formula CI ({formula_ci_low:.2f} to {formula_ci_high:.2f})')
        ax.axvline(formula_ci_high, color='orange', linestyle=':', linewidth=2.5)

    # --- Step 5.5: Finalize and show the plot ---
    ax.set_title(f'Bootstrap Distribution of "{variable}" ({statistic_func.__name__})', fontsize=16)
    ax.set_xlabel(f'Simulated {statistic_func.__name__} Values', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.legend()
    plt.show()

    # --- Step 5.6: Print a final summary ---
    print("\n--- Results Summary ---")
    print(f"Statistic from Original Data: {original_stat:.4f}")
    print(f"Bootstrap 95% Confidence Interval: ({ci_low:.4f}, {ci_high:.4f})")
    if statistic_func == np.mean:
        print(f"Formula-based 95% CI (for mean): ({formula_ci_low:.4f}, {formula_ci_high:.4f})")
    print("-" * 30 + "\n")

    return bootstrap_stats_series, (ci_low, ci_high)


# =============================================================================
# 6. EXAMPLE USAGE OF THE BOOTSTRAP FUNCTION
# =============================================================================
# This special block `if __name__ == '__main__':` ensures that the code inside it
# only runs when you execute this file directly (not when you import it into another script).


# --- Example 1: Bootstrap the MEAN ---
# We pass the DataFrame, the column name 'realinc', and the function `np.mean`.
# The plot for this example will include the orange dotted lines for the formula-based CI.
mean_stats, mean_ci = bootstrap_simulation(data=gss,
                                           variable='realinc',
                                           statistic_func=np.mean,
                                           num_iterations=1000)

# --- Example 2: Bootstrap the MEDIAN ---
# The power of our function is that we can just switch out the statistic.
# A formula-based CI for the median is complex, showing why bootstrapping is so useful.
median_stats, median_ci = bootstrap_simulation(data=gss,
                                                   variable='realinc',
                                                   statistic_func=np.median,
                                                   num_iterations=1000)

# --- Example 3: Bootstrap the STANDARD DEVIATION ---
# We can even get a confidence interval for the standard deviation itself.
sd_stats, sd_ci = bootstrap_simulation(data=gss,
                                        variable='realinc',
                                        statistic_func=np.std,
                                        num_iterations=1000)

# --- Example 3: Bootstrap the 10th PERCENTILE ---
# We can even get a confidence interval for the 10th percentile itself.
percentile10_stats, percentile10_ci = bootstrap_simulation(data=gss,
                                        variable='realinc',
                                        statistic_func=lambda x: np.percentile(x, 10),
                                        num_iterations=1000)


# What is a lambda function?
# A lambda function is a small, anonymous (unnamed) function. It's a shorthand for defining a simple function right where you need it.
# Your line: lambda x: np.percentile(x, 10)
# Is exactly the same as writing this:
# code
# Python
# def calculate_10th_percentile(x):
#     return np.percentile(x, 10)

# =============================================================================
# 6. BOOTSTRAP FUNCTION ON MOVIE RATINGS
# =============================================================================
movies = pd.read_csv('data/movies.csv')

mean_rating_stats, mean_rating_ci = bootstrap_simulation(
        data=movies.query("genre == 'Animation'"),
        variable='rating',
        statistic_func=np.mean,
        num_iterations=1000
    )

mean_rating_stats, mean_rating_ci = bootstrap_simulation(
    data=movies.query("genre == 'Documentary'"),
    variable='rating',
    statistic_func=np.mean,
    num_iterations=1000
)