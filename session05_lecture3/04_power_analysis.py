# ==============================================================================
# SCRIPT: Power Analysis for Two-Sample T-Tests using statsmodels
# ==============================================================================

# This script demonstrates how to use the `TTestIndPower` class from the
# `statsmodels` library to perform power analysis. It covers the three most
# common use cases for the `solve_power` method and shows how to visualize
# a power curve for multiple alpha levels using seaborn.

# ------------------------------------------------------------------------------
# Section 1: Setup and Imports
# ------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.power import TTestIndPower

print("--- Power Analysis Script Initialized ---")

# Set seaborn style for professional-looking plots
sns.set_theme(style="whitegrid")


# ------------------------------------------------------------------------------
# Section 2: Calculating Required Sample Size (A Priori Analysis)
# ------------------------------------------------------------------------------
# This is the most common use case: determining the number of participants
# needed *before* conducting a study to have a high probability of detecting
# a meaningful effect.

print("\n--- Section 2: Calculating Required Sample Size ---")
print("Scenario: 'How many participants do I need for my A/B test?'")

# --- Parameters ---
effect_size_medium = 0.35  # We want to detect a 'medium' effect (Cohen's d)
alpha = 0.05              # Standard significance level (p-value threshold)
target_power = 0.80       # Desired power (80% chance to detect a real effect)

# Instantiate the power analysis object
power_analysis = TTestIndPower()

# --- Calculation ---
# To find the sample size, we set the 'nobs1' parameter to None.
# The 'ratio' parameter is 1.0 by default, meaning we assume equal group sizes.
required_n = power_analysis.solve_power(
    effect_size=effect_size_medium,
    alpha=alpha,
    power=target_power,
    nobs1=None,  # This is the parameter we want to solve for
    alternative='two-sided'
)

# Use np.ceil to round up, since you can't have a fraction of a participant.
required_n_rounded = np.ceil(required_n)

print(f"To detect a medium effect (d={effect_size_medium}) with {target_power:.0%} power and an alpha of {alpha}:")
print(f"You would need approximately {required_n_rounded:.0f} participants IN EACH GROUP.")
print(f"Total participants needed: {required_n_rounded * 2:.0f}")


# ------------------------------------------------------------------------------
# Section 3: Calculating Statistical Power (Post-Hoc Analysis)
# ------------------------------------------------------------------------------
# This is useful *after* a study is complete, especially if you got a
# non-significant result. It helps you determine if your study was
# sensitive enough ("powered") to find an effect if one existed.

print("\n--- Section 3: Calculating Achieved Power ---")
print("Scenario: 'My study is done. What was my chance of finding an effect?'")

# --- Parameters ---
# Let's assume the true effect size was medium (d=0.5)
effect_size_medium = 0.5
alpha = 0.05
actual_nobs1 = 40  # The actual number of participants we managed to get per group

# --- Calculation ---
# To find the power, we set the 'power' parameter to None.
achieved_power = power_analysis.solve_power(
    effect_size=effect_size_medium,
    alpha=alpha,
    nobs1=actual_nobs1,
    power=None,  # This is the parameter we want to solve for
    alternative='two-sided'
)

print(f"With {actual_nobs1} participants per group and an expected effect size of d={effect_size_medium}:")
print(f"The achieved power of the test was approximately {achieved_power:.2f} (or {achieved_power:.0%}).")
print("Interpretation: This test was likely underpowered. It only had about a coin-flip's chance of detecting a true medium-sized effect.")


# ------------------------------------------------------------------------------
# Section 4: Calculating Minimum Detectable Effect Size
# ------------------------------------------------------------------------------
# This is a practical analysis when your resources (and thus sample size)
# are fixed. It answers the question: "Given my constraints, how big must the
# effect be for me to have a good chance of detecting it?"

print("\n--- Section 4: Calculating Minimum Detectable Effect Size ---")
print("Scenario: 'My budget is fixed. What's the smallest effect I can hope to find?'")

# --- Parameters ---
alpha = 0.05
target_power = 0.80
fixed_nobs1 = 30  # Our fixed sample size per group due to budget constraints

# --- Calculation ---
# To find the minimum effect size, we set 'effect_size' to None.
min_effect_size = power_analysis.solve_power(
    alpha=alpha,
    power=target_power,
    nobs1=fixed_nobs1,
    effect_size=None,  # This is the parameter we want to solve for
    alternative='two-sided'
)

print(f"With only {fixed_nobs1} participants per group and a target of {target_power:.0%} power:")
print(f"The smallest effect size you could reliably detect is d ≈ {min_effect_size:.3f}.")
print("Interpretation: This study is only sensitive to large effects. Any true effect smaller than this would likely be missed.")


# ------------------------------------------------------------------------------
# Section 5: Visualizing Power Curves with Multiple Alpha Levels
# ------------------------------------------------------------------------------
# A power curve is the best way to visualize the relationship between sample
# size, power, and significance level (alpha).

print("\n--- Section 5: Visualizing the Power Curve for Different Alphas ---")

# --- Parameters for Visualization ---
effect_size_medium = 0.15
# Define the alpha levels to compare
alphas_to_plot = [0.05, 0.01] # Plot 0.05 first
target_power = 0.90
# Create a range of sample sizes to plot on the x-axis
sample_sizes = np.arange(start=20, stop=1500, step=5)

# --- Calculation for Plotting ---
# Calculate power for each alpha level and combine into one DataFrame
all_power_data = []
for alpha_level in alphas_to_plot:
    powers = power_analysis.power(
        effect_size=effect_size_medium,
        nobs1=sample_sizes,
        alpha=alpha_level,
        alternative='two-sided'
    )
    temp_df = pd.DataFrame({
        'sample_size_per_group': sample_sizes,
        'power': powers,
        'alpha': f"alpha = {alpha_level}"  # Create a formatted label for the legend
    })
    all_power_data.append(temp_df)

# Combine the list of DataFrames into a single one
power_data = pd.concat(all_power_data, ignore_index=True)


# --- Plotting with Seaborn ---
fig, ax = plt.subplots(figsize=(12, 8))
palette = sns.color_palette("viridis", len(alphas_to_plot))

# Use seaborn's 'hue' to plot separate lines for each alpha
sns.lineplot(
    data=power_data,
    x='sample_size_per_group',
    y='power',
    hue='alpha',
    palette=palette,
    linewidth=2.5,
    ax=ax
)

# Add a horizontal line for the 80% power target
ax.axhline(
    y=target_power,
    color='red',
    linestyle='--',
    linewidth=2,
    label=f'{target_power:.0%} Power Target'
)

# --- NEW: Add vertical lines and non-overlapping annotations for each alpha ---
# Define vertical offsets to prevent text from overlapping
vertical_offsets = [-0.10, 0.05]

for i, alpha_level in enumerate(alphas_to_plot):
    # Calculate the required N for 80% power at this alpha
    required_n = np.ceil(power_analysis.solve_power(
        effect_size=effect_size_medium,
        alpha=alpha_level,
        power=target_power,
        nobs1=None
    ))
    
    # Get the color corresponding to the line plot
    line_color = palette[i]
    
    # Add a vertical line from the x-axis to the power curve
    ax.axvline(
        x=required_n,
        color=line_color,
        linestyle=':',
        linewidth=2,
        ymin=0,
        ymax=target_power / ax.get_ylim()[1] # Scale line to stop at the 80% mark
    )
    
    # Add a text annotation with an arrow
    ax.annotate(
        f'N = {int(required_n)} for alpha={alpha_level}',
        xy=(required_n, target_power), # Arrow tip points to the intersection
        xytext=(required_n + 10, target_power + vertical_offsets[i]), # Text position
        arrowprops=dict(facecolor=line_color, edgecolor=line_color, shrink=0.05),
        fontsize=12,
        fontweight='bold',
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=line_color, lw=1, alpha=0.8)
    )

# --- Formatting ---
ax.set_title(f"Power Curve for a Two-Sample T-Test (Effect Size d={effect_size_medium})", fontsize=16, pad=20)
ax.set_xlabel("Sample Size Per Group", fontsize=12)
ax.set_ylabel("Power (Probability of Detecting Effect)", fontsize=12)
ax.set_ylim(0, 1.05)
# Manually adjust legend to include the red line
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles, labels=labels, title='Significance Level', fontsize=11)

# Show the plot
plt.tight_layout()
plt.show()
