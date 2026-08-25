# Loading necessary libraries
import pandas as pd  # For data manipulation
import matplotlib.pyplot as plt  # For creating plots
import seaborn as sns  # For enhanced statistical visualizations
import numpy as np  # For numerical operations
from scipy import stats  # For statistical formulas
from scipy.stats import t  # For t-distribution (used in confidence intervals)
import warnings  # For handling warning messages

# --- INITIAL SETUP ---

# Suppress warning messages for a cleaner output
warnings.filterwarnings('ignore')

# Set a consistent style and color palette for all plots
plt.style.use('default')
sns.set_palette("husl")
# Set default font size and family
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'sans-serif'


# --- DATA LOADING ---

# Load the dataset from a CSV file into a pandas DataFrame
# The script will exit if the file is not found.
try:
    survey = pd.read_csv('data/student_survey.csv')
    # Print a summary of the DataFrame (columns, data types, non-null counts)
    survey.info()
except FileNotFoundError:
    print("Error: 'data/student_survey.csv' not found.")
    print("Please ensure the CSV file is in the correct directory.")
    exit()


# --- HELPER FUNCTION ---

# This function calculates key summary statistics for a value, grouped by a category
def favstats_by_group(df, value, group):
    """Calculates grouped summary statistics including a 95% confidence interval."""
    # Group the data and calculate common statistics
    stats = df.groupby(group)[value].agg(['min', 'max', 'mean', 'median', 'std', 'count'])
    # Calculate the standard error of the mean
    stats['se'] = stats['std'] / np.sqrt(stats['count'])
    # Find the t-critical value for a 95% confidence interval
    stats['t_critical'] = stats['count'].apply(lambda n: t.ppf(0.975, df=n - 1) if n > 1 else np.nan)
    # Calculate the lower and upper bounds of the confidence interval
    stats['lower_ci'] = stats['mean'] - stats['t_critical'] * stats['se']
    stats['upper_ci'] = stats['mean'] + stats['t_critical'] * stats['se']
    # Remove the temporary t_critical column and round the output
    stats = stats.drop(columns=['t_critical'])
    return stats.round(2)


# --- DATA ANALYSIS AND VISUALIZATION ---

# Define the list of variables to analyze (column_name, plot_label)
variables = [
    ('haircut_spend', 'Haircut Spend'),
    ('exercise_hrs', 'Exercise Hours'),
    ('online_hrs', 'Online Hours'),
    ('sleep_hrs', 'Sleep Hours'),
    ('facebook_friends', 'Facebook Friends'),
    ('motivated_course_grade', 'Motivation'),
    ('relaxed_stressful', 'Stress During the Day'),
    ('anxiety_about_analytics', 'Anxiety About Analytics'),
    ('homeopathy_works', 'Homeopathy'),
    ('lied_about_age', 'Lied About Age'),
    ('marijuana', 'Marijuana')
]

# Loop through each variable to generate plots
for var, label in variables:
    # Ensure the variable is a number before trying to plot it
    if survey[var].dtype in [np.float64, np.int64]:
        # Create a figure to hold two subplots side-by-side
        plt.figure(figsize=(10, 4))

        # First subplot: Boxplot
        plt.subplot(1, 2, 1)
        sns.boxplot(data=survey, x=var, y='gender')
        plt.title(f'{label} vs. Gender (Boxplot)')
        
        # Second subplot: Density Plot
        plt.subplot(1, 2, 2)
        # Create an overlapping density plot for each gender
        for gender in survey['gender'].unique():
            subset = survey[survey['gender'] == gender]
            # Use density=True to normalize the histogram
            plt.hist(subset[var], alpha=0.3, density=True, label=gender, bins=20)
        plt.xlabel(label)
        plt.ylabel('Density')
        plt.title(f'{label} Distribution by Gender')
        plt.legend()

        # Adjust layout and display the plots
        plt.tight_layout()
        plt.show()

# --- STATISTICAL TESTS ---

# Loop through each variable again to perform statistical tests
for var, label in variables:
    print(f"\n--- {label} ({var}) ---")
    
    # Print summary statistics grouped by gender
    print(favstats_by_group(survey, var, 'gender'))
    
    # Separate the data into two groups for the t-test
    group1 = survey[survey['gender'] == survey['gender'].unique()[0]][var]
    group2 = survey[survey['gender'] == survey['gender'].unique()[1]][var]
    
    # Perform an independent t-test, ignoring any missing (NaN) values
    t_stat, p_value = stats.ttest_ind(group1, group2, nan_policy='omit')
    
    # Print the t-test results, formatted to 3 decimal places
    print(f"T-test for {label} by gender:")
    print(f"T-statistic: {t_stat:.3f}")
    # The p-value indicates statistical significance (p <= 0.05 is a common threshold)
    print(f"P-value: {p_value:.3f}")