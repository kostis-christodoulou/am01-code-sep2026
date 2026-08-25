# =============================================================================
# FILE: bootstrap_function.py
# PURPOSE: A reusable module for performing bootstrap simulations.
# AUTHOR: Kostis Christodoulou
# DATE: 2025-08-23
# =============================================================================

# --- Step 1: Import all necessary libraries for the function to work ---
# By placing imports here, anyone who imports our function doesn't have to
# worry about importing these themselves.
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# --- Step 2: Define the main, reusable function ---
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
    print("--- Starting Bootstrap Simulation ---")
    print(f"Variable: '{variable}'")
    print(f"Statistic: {statistic_func.__name__}")
    print(f"Number of Iterations: {num_iterations}\n")

    # Prepare the original data
    original_series = data[variable].dropna()
    original_size = len(original_series)

    # Run the bootstrap resampling
    bootstrap_stats = [statistic_func(original_series.sample(n=original_size, replace=True))
                       for _ in range(num_iterations)]
    bootstrap_stats_series = pd.Series(bootstrap_stats)

    # Calculate the percentile confidence interval
    ci_low = bootstrap_stats_series.quantile(0.025)
    ci_high = bootstrap_stats_series.quantile(0.975)

    # Calculate the statistic on the original data
    original_stat = statistic_func(original_series)

    # Plot the results
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(bootstrap_stats_series, ax=ax, kde=True, stat="density",
                 label="Bootstrap Distribution of " + statistic_func.__name__)
    ax.axvline(original_stat, color='black', linestyle='-', linewidth=2,
               label=f'Original {statistic_func.__name__}: {original_stat:.2f}')
    ax.axvspan(ci_low, ci_high, color='red', alpha=0.2,
               label=f'95% Bootstrap CI ({ci_low:.2f} to {ci_high:.2f})')

    if statistic_func == np.mean:
        std_error = original_series.std() / np.sqrt(original_size)
        degrees_of_freedom = original_size - 1
        t_critical = stats.t.ppf(0.975, df=degrees_of_freedom)
        formula_ci_low = original_stat - t_critical * std_error
        formula_ci_high = original_stat + t_critical * std_error
        ax.axvline(formula_ci_low, color='orange', linestyle=':', linewidth=2.5,
                   label=f'Formula CI ({formula_ci_low:.2f} to {formula_ci_high:.2f})')
        ax.axvline(formula_ci_high, color='orange', linestyle=':', linewidth=2.5)

    # Finalize and show the plot
    ax.set_title(f'Bootstrap Distribution of "{variable}" ({statistic_func.__name__})', fontsize=16)
    ax.set_xlabel(f'Simulated {statistic_func.__name__} Values', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.legend()
    plt.show()

    # Print a final summary
    print("\n--- Results Summary ---")
    print(f"Statistic from Original Data: {original_stat:.4f}")
    print(f"Bootstrap 95% Confidence Interval: ({ci_low:.4f}, {ci_high:.4f})")
    if statistic_func == np.mean:
        print(f"Formula-based 95% CI (for mean): ({formula_ci_low:.4f}, {formula_ci_high:.4f})")
    print("-" * 30 + "\n")

    return bootstrap_stats_series, (ci_low, ci_high)


# --- Step 3: Create a demonstration block ---
# This is a special block in Python. The code inside `if __name__ == '__main__':`
# will ONLY run when you execute this file directly (e.g., `python bootstrap_function.py`).
# It will NOT run when you import this file into another script.
# This makes it the perfect place to put examples and tests for your function.

if __name__ == '__main__':
    print("--- Running Demonstration for bootstrap_function.py ---")
    print("This code is for example purposes and will not run when imported.\n")

    # Create some sample data for the demonstration
    np.random.seed(42)
    simulated_data = pd.DataFrame({
        'income': np.random.lognormal(mean=10.5, sigma=0.8, size=500),
        'age': np.random.randint(18, 70, size=538)
    })

    # --- Example 1: Bootstrap the MEAN of income ---
    print("--- DEMO 1: Bootstrapping the Mean ---")
    bootstrap_simulation(data=simulated_data,
                         variable='income',
                         statistic_func=np.mean,
                         num_iterations=1000)

    # --- Example 2: Bootstrap the MEDIAN of age ---
    print("--- DEMO 2: Bootstrapping the Median ---")
    bootstrap_simulation(data=simulated_data,
                         variable='age',
                         statistic_func=np.median,
                         num_iterations=1000)