# =============================================================================
# 1. IMPORT LIBRARIES
# =============================================================================
# - numpy: For performing the binomial simulation efficiently.
# - pandas: To organize our results into a tidy DataFrame.
# - seaborn: To create the FacetGrid visualization.
# - matplotlib.pyplot: To customize the final plot (e.g., add a main title).

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# =============================================================================
# 2. SET UP THE SIMULATION PARAMETERS
# =============================================================================
# --- User-Specified Parameters ---
# A list of the different probabilities (p) of success for each trial.
probabilities = [0.01, 0.1, 0.5, 0.7]

# A list of the different numbers of trials (n) to run in each experiment.
num_trials = [1, 5, 10, 100, 1000]

# --- Simulation Configuration ---
# The number of times we repeat each experiment to get a smooth distribution.
num_simulations = 10000

# =============================================================================
# 3. RUN THE SIMULATIONS AND COLLECT RESULTS
# =============================================================================
# This section is unchanged. We generate the data in the same way.
all_results = []

print("Running binomial simulations...")

for p in probabilities:
    for n in num_trials:
        print(f"  - Simulating for p={p}, n={n}...")
        success_counts = np.random.binomial(n, p, size=num_simulations)
        temp_df = pd.DataFrame({
            'num_successes': success_counts,
            'probability_p': p,
            'num_trials_n': n
        })
        all_results.append(temp_df)

df_results = pd.concat(all_results, ignore_index=True)

print("Simulations complete!")
print("\n--- Tidy DataFrame Structure ---")
df_results.info()


# =============================================================================
# 4. VISUALIZE THE RESULTS WITH A CONDITIONAL DENSITY PLOT
# =============================================================================
# Now, we create the grid of plots with our custom logic.

# Set the visual style.
sns.set_style("ticks")

# Step 1: Create the FacetGrid structure.
g = sns.FacetGrid(
    data=df_results,
    row="probability_p",
    col="num_trials_n",
    hue="probability_p",
    sharex=False,
    sharey=False
)

# Step 2: Draw the base histogram on ALL facets.
# We make it slightly transparent so the potential density line will be visible.
g.map_dataframe(
    sns.histplot,
    x="num_successes",
    stat="probability",
    discrete=True,
    alpha=0.7
)

# Step 3: Iterate through each facet and add a smooth density plot only if the condition is met.
# The `g.axes_dict` is a dictionary where keys are the (row, col) facet values
# (e.g., (0.1, 50)) and values are the matplotlib subplot objects (the Axes).
print("\nAdding smoothed density plots where n*p > 5...")
for (p, n), ax in g.axes_dict.items():
    # Check our condition for this specific facet.
    if n * p > 5 and n * (1 - p) > 5:
        print(f"  - Adding density line for p={p}, n={n} (since n*p = {n*p:.1f})")

        # To plot on this specific facet, we need to filter the main DataFrame
        # to get only the data that belongs here.
        facet_data = df_results[(df_results['probability_p'] == p) &
                                (df_results['num_trials_n'] == n)]

        # Now, draw the density plot (the smoothed line) on the current axis (`ax`).
        # The function `sns.kdeplot` is seaborn's tool for creating this smooth density line.
        # The `ax=ax` argument is crucial—it tells seaborn exactly where to draw.
        sns.kdeplot(
            data=facet_data,
            x='num_successes',
            color='black',
            linewidth=2,
            ax=ax  # This ensures the plot is drawn on the correct subplot
        )

# --- Customize Titles and Labels ---
# Add a main title for the entire figure.
g.fig.suptitle('Binomial Distribution (Density Plot Overlay when n*p > 5)', fontsize=16, y=1.03)

# Set the titles for each individual facet.
g.set_titles("n = {col_name} | p = {row_name}")

# Set the labels for the outer axes.
g.set_axis_labels("Number of Successes", "Probability / Density")

# Adjust the layout to prevent titles from overlapping.
plt.tight_layout()

# Show the final plot.
plt.show()
