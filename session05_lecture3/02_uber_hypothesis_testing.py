# ==============================================================================
# 1. SETUP AND DATA LOADING
# ==============================================================================

# Import necessary libraries
import pandas as pd
import janitor  # For the .clean_names() method
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# Set plot style to be similar to R's theme_bw()
plt.style.use('seaborn-v0_8-whitegrid')
# Set a default font that is clean and widely available
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

# Load and clean the data using command chaining
# Assumes the data is in a subfolder named 'data'
try:
    uber = (
        pd.read_csv('data/uber.csv')
        .clean_names()  
        .assign(
            period_start=lambda df: pd.to_datetime(df['period_start'], format='%d/%m/%y %H:%M')
        )
    )
    print("Data loaded and cleaned successfully.")
    print("First 5 rows of the dataset:")
    print(uber.head())
 
except FileNotFoundError:
    print("Error: 'data/uber.csv' not found. Please create a 'data' folder and place the CSV file inside.")
    # Create a dummy dataframe to prevent the rest of the script from crashing
    uber = pd.DataFrame() 


uber.info()

# Proceed only if the dataframe was loaded successfully
# ==============================================================================
# 2. PLOT CANCELLATION RATE OVER TIME
# ==============================================================================

# Create the figure and axes for the plot
fig, ax = plt.subplots(figsize=(12, 7))

# Plot cancellation rate over time
sns.scatterplot(
    data=uber,
    x='period_start',
    y='cancellation_rate',
    hue='group',
    ax=ax
)

# Plot target cancellation of 4%
ax.axhline(
    y=4,
    color="#001e62",
    linewidth=1.1,
    linestyle="--"
)

# Set titles and labels
ax.set_title("Cancellation Rate for groups A & B", fontsize=16)
ax.set(
    xlabel=None,
    ylabel="Cancellation Rate"
)


plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
plt.show()



# ==============================================================================
# 3. PLOT CANCELLATION RATE BY GROUP (WITH CONFIDENCE INTERVALS)
# ==============================================================================

# Create the figure and axes for the plot
fig, ax = plt.subplots(figsize=(10, 6))

# Use stripplot for the individual data points (like geom_point)
sns.stripplot(
    data=uber,
    x='cancellation_rate',
    y='group',
    ax=ax,
    alpha=0.7,
    jitter=0.05  # Helps spread out the points
)

# Use pointplot to overlay the mean and 95% CI (like stat_summary)
sns.pointplot(
    data=uber,
    x='cancellation_rate',
    y='group',
    ax=ax,
    join=False,          # Don't connect the points
    errorbar=('ci', 95), # Corresponds to mean_se with mult=1.96
    color='red',
    markersize=20,
    linewidth=3,
    capsize=0.1
)

# --- Add mean value labels (equivalent to stat_summary with geom="text") ---
# 1. Calculate the means
group_means = uber.groupby('group')['cancellation_rate'].mean()

# 2. Iterate and add text to the plot
for i, group in enumerate(group_means.index):
    mean_val = group_means[group]
    ax.text(
        x=mean_val,
        y=i,
        s=f'{mean_val:.2f}',  # Format to 2 decimal places
        color='white',
        fontsize=8,
        fontweight='bold',
        ha='center',          # Horizontal alignment
        va='center'           # Vertical alignment
    )

# Plot target cancellation of 4%
ax.axvline(
    x=4,
    color="#001e62",
    linestyle="--"
)

# Set titles, labels, and theme elements
ax.set_title("Cancellation Rate for groups A & B", loc='left', fontsize=16)
ax.set(
    xlabel="Cancellation Rate",
    ylabel=None
)
ax.tick_params(axis='y', labelsize=14)

plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.show()


# ==============================================================================
# 4. PLOT MATCH RATE BY GROUP (WITH CONFIDENCE INTERVALS)
# ==============================================================================

# Create the figure and axes for the plot
fig, ax = plt.subplots(figsize=(10, 6))

# Use stripplot for the individual data points
sns.stripplot(
    data=uber,
    x='match_rate',
    y='group',
    ax=ax,
    alpha=0.7,
    jitter=0.05
)

# Use pointplot to overlay the mean and 95% CI
sns.pointplot(
    data=uber,
    x='match_rate',
    y='group',
    ax=ax,
    join=False,
    errorbar=('ci', 95),
    color='red',
    markersize=20,
    linewidth=2,
    capsize=0.1
)

# --- Add mean value labels ---
group_means_match = uber.groupby('group')['match_rate'].mean()
for i, group in enumerate(group_means_match.index):
    mean_val = group_means_match[group]
    ax.text(
        x=mean_val,
        y=i,
        s=f'{mean_val:.1f}',  # Format to 1 decimal place
        color='white',
        fontsize=10,
        fontweight='bold',
        ha='center',
        va='center'
    )

# Set titles, labels, and theme elements
ax.set_title("Match Rate for groups A & B", loc='left', fontsize=16)
ax.set(
    xlabel="Match Rate",
    ylabel=None
)
ax.tick_params(axis='y', labelsize=14)

plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.show()


# ==============================================================================
# 5. SUMMARY STATISTICS
# ==============================================================================

# Get quick summary statistics for cancellation_rate by group
cancellation_summary = (
    uber
    .groupby('group')['cancellation_rate']
    .describe().round(2)
    .reset_index()
)

print("--- Summary statistics of cancellation rate ---")
print(cancellation_summary)


# Get quick summary statistics for match_rate by group
match_summary = (
    uber
    .groupby('group')['match_rate']
    .describe().round(2)
    .reset_index()  
)

print("--- Summary statistics of match rate ---")
print(match_summary)

# ==============================================================================
# 6. HYPOTHESIS TESTING
# ==============================================================================

# --- One-Sample T-Tests (Cancellation Rate vs. 4% Target) ---
# The Python 1 sample t-test  is `scipy.stats.ttest_1samp`.


# We must first split the data into the two groups (e.g., `group_a` and `group_b`) before running the test
group_a = uber.query("group == 'A - 5 min'")
group_b = uber.query("group == 'B - 2min'")

# H0: The mean cancellation rate for Group A is 4%
ttest_a = stats.ttest_1samp(
    a=group_a['cancellation_rate'],
    popmean=4
)

# H0: The mean cancellation rate for Group B is 4%
ttest_b = stats.ttest_1samp(
    a=group_b['cancellation_rate'],
    popmean=4
)

print("--- 1-Sample T-Test: Is Group A cancellation rate = 4%? ---")
print(f"T-statistic: {ttest_a.statistic:.4f}, P-value: {ttest_a.pvalue:.4f}\n")

print("--- 1-Sample T-Test: Is Group B cancellation rate = 4%? ---")
print(f"T-statistic: {ttest_b.statistic:.4f}, P-value: {ttest_b.pvalue:.4f}\n")

# --- Two-Sample T-Tests (Comparing Group Means) ---
# The Python t-test of two independent samples is `scipy.stats.ttest_ind`.

# H0: The mean cancellation rates of Group A and Group B are equal
ttest_cancellation = stats.ttest_ind(
    a=group_a['cancellation_rate'],
    b=group_b['cancellation_rate'],
    equal_var=False  # Perform Welch's t-test, assuming unequal variances
)
print("--- 2-Sample T-Test: Is the mean cancellation rate between groups the same? ---")
print(f"T-statistic: {ttest_cancellation.statistic:.4f}, P-value: {ttest_cancellation.pvalue:.4f}\n")

# H0: The mean match rates of Group A and Group B are equal
ttest_match = stats.ttest_ind(
    a=group_a['match_rate'],
    b=group_b['match_rate'],
    equal_var=False  # Perform Welch's t-test, assuming unequal variances
)
print("--- 2-Sample T-Test: Is the mean match rate between groups the same? ---")
print(f"T-statistic: {ttest_match.statistic:.4f}, P-value: {ttest_match.pvalue:.4f}\n")