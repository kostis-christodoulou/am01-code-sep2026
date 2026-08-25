# =============================================================================
# 1. IMPORT LIBRARIES
# =============================================================================
# - numpy: The core library for numerical operations. We'll use it to simulate the random dice rolls.
# - pandas: The primary library for data manipulation and analysis. We will use it to organize our simulation results into a clean table (a DataFrame).
# - seaborn: A powerful and easy-to-use library for statistical plotting. It's built on top of matplotlib.
# - matplotlib.pyplot: The foundational plotting library. We use it to customize our plot, like setting the size and title.

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# =============================================================================
# 2. SET UP AND RUN THE SIMULATIONS
# =============================================================================

# --- Configuration ---
# Define how many times we want to simulate the dice rolls.
# A higher number gives a smoother, more accurate result.
num_simulations = 10000

# Define the different numbers of dice we want to test in each experiment.
dice_to_roll = [1, 4, 12]


# --- Simulation ---
# We will store the results of our simulations in a dictionary.
# The keys will be the column names (like "1 Die") and the values will be the list of outcomes.
simulation_results = {}

print("Running simulations...")

# Loop through each number in our list of dice to roll.
for num_dice in dice_to_roll:
    # Create a descriptive name for our column, e.g., "1 Die", "2 Dice".
    label = f"{num_dice} {'Die' if num_dice == 1 else 'Dice'}"
    print(f"  - Simulating rolling {label}...")

    # Simulate the dice rolls using NumPy.
    # np.random.randint(1, 7, size=(num_simulations, num_dice))
    # This creates a 2D array (a grid of numbers).
    # - The numbers will be between 1 (inclusive) and 7 (exclusive), i.e., 1, 2, 3, 4, 5, 6.
    # - The `size` will be (10000 rows, num_dice columns). Each row is one simulation.
    rolls = np.random.randint(1, 7, size=(num_simulations, num_dice))

    # Sum the dice for each simulation.
    # `axis=1` tells NumPy to sum the numbers horizontally (across the columns) for each row.
    # The result is a 1D array containing 10,000 sums.
    sums = rolls.sum(axis=1)

    # Store the array of sums in our dictionary.
    simulation_results[label] = sums

print("Simulations complete!")


# =============================================================================
# 3. ORGANIZE DATA INTO A PANDAS DATAFRAME
# =============================================================================
# Now, we convert our dictionary of results into a pandas DataFrame.
# This creates a beautiful, labeled table that is perfect for inspection and plotting.
df_sums = pd.DataFrame(simulation_results)

# Let's look at the first few rows of our new DataFrame.
print("\n--- First 5 Simulation Results ---")
print(df_sums.head())

# We can also get a quick statistical summary of our results.
print("\n--- Statistical Summary of Sums ---")
print(df_sums.describe())


# =============================================================================
# 4. VISUALIZE THE RESULTS WITH SEABORN
# =============================================================================
# Now for the fun part! With our data neatly organized in a DataFrame, plotting is easy.

# Set the style for our plot for a nice appearance.
sns.set_style("whitegrid")

# Create a figure and axes for our plot. `figsize` controls the size in inches.
plt.figure(figsize=(14, 8))

# Create the main plot using seaborn's histplot.
# A histogram is perfect for showing the distribution of outcomes.
plot = sns.histplot(
    data=df_sums,              # The DataFrame containing all our data. Seaborn knows how to handle it.
    stat="probability",        # This is key! It changes the y-axis from a raw count to probability, creating a PMF.
    common_norm=False,         # Ensures each distribution is normalized independently.
    discrete=True,             # Treats the outcomes as distinct integers (e.g., you can't roll a 7.5).
    # kde=True,                  # Adds a smoothed line (Kernel Density Estimate) over the bars to help see the shape.
    linewidth=0.5              # Adds a thin line around the bars for better visibility.
)

# --- Add labels and a title to make the plot easy to understand ---
plt.title('Probability Distribution of the Sum of Dice Rolls', fontsize=16)
plt.xlabel('Sum of Dice', fontsize=12)
plt.ylabel('Probability', fontsize=12)

# Set the x-axis ticks to be integers for clarity.
# Find the maximum possible sum (10 dice * 6) to set the plot limits.
max_sum = df_sums['10 Dice'].max()
plt.xticks(np.arange(1, max_sum + 2, 2)) # Show ticks every 2 numbers

# Show the final plot.
plt.show()