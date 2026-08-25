# Import all the libraries we need for data analysis and visualization
import pandas as pd # Used for data manipulation and analysis - think Excel but in Python
import seaborn as sns # Used for statistical data visualization - makes pretty charts
import matplotlib.pyplot as plt # Used for creating static, animated, and interactive visualizations
import matplotlib.dates as mdates # Used for formatting dates on plots
from matplotlib.ticker import FuncFormatter # Used for formatting tick labels on plot axes (like adding commas to numbers)
import numpy as np # Used for numerical operations and arrays
from skimpy import skim # Used for generating summary statistics of DataFrames
from datetime import datetime # Used for working with dates and times
import warnings # Used to control warning messages
warnings.filterwarnings('ignore') # Hide warning messages to keep output clean

# Set style for better looking plots - makes charts look more professional
plt.style.use('seaborn-v0_8') # Apply a seaborn style template
sns.set_palette("husl") # Set a color palette for consistent colors across charts

# Read the CSV file into a pandas DataFrame (like loading an Excel file)
bike = pd.read_csv('data/london_bikes.csv')
skim(bike)

# DATA PREPARATION SECTION
# Fix dates and generate new variables for year, month, month_name, day, and day_of_week

# Convert the 'date' column from text to actual date format so Python knows it's dates
bike['date'] = pd.to_datetime(bike['date'])

# Extract just the month number (1-12) from each date
bike['month'] = bike['date'].dt.month

# Extract month names as 3-letter abbreviations (Jan, Feb, etc.) from each date
bike['month_name'] = bike['date'].dt.strftime('%b')

# Create a list with the correct order of weekday abbreviations
# This ensures days appear in logical order (Mon-Sun) instead of alphabetical
day_order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

# Convert the 'wday' column to a categorical type with the specified order
# Categorical = tells pandas this column has a specific order, not just random text
bike['day_of_week'] = pd.Categorical(bike['wday'], categories=day_order, ordered=True)



# Create a list with the correct order of month abbreviations
# This ensures months appear in calendar order instead of alphabetical
month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# Convert the 'month_name' column to a categorical type with the specified order
bike['month_name'] = pd.Categorical(bike['month_name'], categories=month_order, ordered=True)

# Generate new variable season_name to turn seasons from numbers to Winter, Spring, etc
# This function takes a month name and returns which season it belongs to
def get_season(month_name):
    if month_name in ['Dec', 'Jan', 'Feb']:  # Winter months
        return 'Winter'
    elif month_name in ['Mar', 'Apr', 'May']:  # Spring months
        return 'Spring'
    elif month_name in ['Jun', 'Jul', 'Aug']:  # Summer months
        return 'Summer'
    else:  # Autumn months (Sep, Oct, Nov)
        return 'Autumn'

# Apply the season function to each row to create a new 'season_name' column
bike['season_name'] = bike['month_name'].apply(get_season)

# Make season_name an ordered categorical so seasons appear in logical order
bike['season_name'] = pd.Categorical(bike['season_name'], 
                                    categories=['Winter', 'Spring', 'Summer', 'Autumn'], 
                                    ordered=True)

# Create a weekend/weekday indicator column

# Lambda is a quick way to write a mini-function: if day is Sat/Sun, label as Weekend, else Weekday
bike['weekend'] = bike['wday'].apply(lambda x: 'Weekend' if x in ['Sat', 'Sun'] else 'Weekday')
# np.where is a vectorized way to achieve the same result: faster for large datasets
bike['weekend'] = np.where(bike['wday'].isin(['Sat', 'Sun']), 'Weekend', 'Weekday')

# EXPLORE THE DATA
# Examine what the resulting data frame looks like
bike.info()  # Shows column names, data types, and memory usage
bike.head()  # Shows the first 5 rows of data

print("\nSummary statistics:")
bike.describe()  # Shows count, mean, std dev, min, max for numerical columns

skim(bike)  # More detailed summary statistics from the skimpy library

# TIME SERIES VISUALIZATION SECTION
# Time series plot of bikes rented, with two means

# ---  Style Setup ---
# Set the overall plot style to "whitegrid" which adds a subtle grid background
sns.set_theme(style="whitegrid")

# --- Calculate Averages ---
# We want to compare bike usage before and after 2023, so calculate separate means

# Define the start and end dates for the first period (2014-2022).
# We add `utc=True` to make these Timestamps timezone-aware, matching the 'date' column.
start_date_period1 = pd.to_datetime("2014-01-01", utc=True)
end_date_period1 = pd.to_datetime("2022-12-31", utc=True)

# Define the start and end dates for the second period (2023 onwards).
start_date_period2 = pd.to_datetime("2023-01-01", utc=True)
end_date_period2 = pd.to_datetime("2024-12-31", utc=True)

# Calculate the mean of 'bikes_hired' for the 2014-2022 period.
# This filters the DataFrame for dates within the range and then computes the mean.
mean_2014_2022 = bike[(bike['date'] >= start_date_period1) & (bike['date'] <= end_date_period1)]['bikes_hired'].mean()

# Calculate the mean of 'bikes_hired' for the 2023-2024 period.
mean_2023_on = bike[(bike['date'] >= start_date_period2) & (bike['date'] <= end_date_period2)]['bikes_hired'].mean()

# --- Data Filtering for Plotting ---
# Create a filtered DataFrame containing only the data we want to plot (2014-2024)
bike_filtered = bike[(bike['date'] >= start_date_period1) & (bike['date'] < pd.to_datetime("2025-01-01", utc=True))].copy()

# --- Plotting ---
# Create a figure and axes object for the plot with a specified size (12 inches wide, 8 inches tall)
fig, ax = plt.subplots(figsize=(12, 8))

# Create a scatter plot of 'bikes_hired' against 'date'.
# 'alpha=0.3' makes the points semi-transparent so overlapping points don't hide each other
sns.scatterplot(data=bike_filtered, x='date', y='bikes_hired', alpha=0.3, ax=ax)

# --- Add Mean Lines (Segments) ---
# Plot horizontal lines showing the average for each time period

# Plot the horizontal line for the 2014-2022 mean (blue line)
ax.plot([start_date_period1, end_date_period1], [mean_2014_2022, mean_2014_2022], color='blue', linewidth=2)

# Plot the horizontal line for the 2023-2024 mean (red line)
ax.plot([start_date_period2, end_date_period2], [mean_2023_on, mean_2023_on], color='red', linewidth=2)

# --- Add Text Labels for Means ---
# Add text labels showing the actual mean values

# Add a text label for the 2014-2022 mean
# Position it at 2018 on x-axis and slightly above the mean line on y-axis
ax.text(x=pd.to_datetime("2018-01-01"), y=mean_2014_2022 + 20000, 
        s=f"Mean: {mean_2014_2022:,.0f}", color="blue", fontsize=14)  # :,.0f formats number with commas, no decimals

# Add a text label for the 2023-2024 mean
ax.text(x=pd.to_datetime("2023-08-01"), y=mean_2023_on + 15000, 
        s=f"Mean: {mean_2023_on:,.0f}", color="red", fontsize=14)

# --- Customize Plot Appearance ---
# Make the plot look professional and easy to read

# Set the title of the plot
ax.set_title("Tfl bikes hired vs mean", fontsize=16)

# Remove the x and y axis labels (set to None means no label)
ax.set_xlabel(None)
ax.set_ylabel(None)

# Set the x-axis to show tick marks every year
ax.xaxis.set_major_locator(mdates.YearLocator())
# Format the x-axis tick labels to show abbreviated month and full year (e.g., "Jan 2014")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

# Format the y-axis tick labels to include commas for thousands (e.g., "10,000" instead of "10000")
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{int(x):,}'))

# Set the limits for the axes to remove padding and show exact date range
ax.set_xlim(pd.to_datetime("2014-01-01"), pd.to_datetime("2025-01-01"))
ax.set_ylim(0) # Start y-axis at 0 (don't show negative bike rentals)

# Adjust layout to prevent labels from being cut off
plt.tight_layout()

# Display the plot on screen
plt.show()

# SUMMARY STATISTICS SECTION
# Generate various summary statistics to understand the data patterns

print("Summary statistics for bikes_hired:")
print(bike['bikes_hired'].describe())  # Basic stats: count, mean, std, min, 25%, 50%, 75%, max

# Summary statistics grouped by different categories
print("\nSummary by year:")
bike.groupby('year')['bikes_hired'].describe()  # Stats for each year separately

print("\nSummary by weekend/weekday:")
bike.groupby('weekend')['bikes_hired'].describe()  # Compare weekend vs weekday usage

print("\nSummary by wday- this is sorted alphabetically:") 
bike.groupby('wday')['bikes_hired'].describe()  # Stats by day (note: alphabetical order)

print("\nSummary by day of week:") 
bike.groupby('day_of_week')['bikes_hired'].describe()  # Stats by day (proper Mon-Sun order)

print("\nSummary by month:")
bike.groupby('month_name')['bikes_hired'].describe()  # Stats for each month

print("\nSummary by season:")
bike.groupby('season_name')['bikes_hired'].describe()  # Stats for each season

# HISTOGRAM SECTION
# Histograms show the distribution of data (how many observations fall in each range)

# Set a style for the plot. 'whitegrid' adds a subtle grid background
sns.set_style("whitegrid")

# Create a figure with specified size (10 inches wide, 6 inches tall)
plt.figure(figsize=(10, 6))

# Create the histogram using seaborn's histplot function
sns.histplot(
    data=bike,           # Specify the DataFrame to use
    x='bikes_hired',     # Specify the column to plot
    bins=30,             # Divide the data into 30 bins (bars)
    alpha=0.7,           # Set transparency (0=invisible, 1=solid)
    color='black',       # Set the fill color of the bars
    edgecolor='grey',    # Add grey lines around each bar for better definition
    kde=True             # Add a smooth density curve on top of the histogram
)

# Customize plot labels and title
plt.title('Histogram of Bikes Rented', fontsize=16)  # Main title
plt.xlabel('Bikes Hired', fontsize=12)               # X-axis label
plt.ylabel('Frequency', fontsize=12)                 # Y-axis label

# Ensure the layout is clean and labels don't overlap
plt.tight_layout()

# Display the plot
plt.show()

# FACETED HISTOGRAM BY SEASON
# Create separate histograms for each season to compare distributions

sns.set_style("whitegrid")  # Set consistent style
# Create a grid of histograms, one for each season
g = sns.displot(
    data=bike,           # The DataFrame to use
    x='bikes_hired',     # Variable to plot
    col='season_name',   # Create separate plots for each season
    col_wrap=2,          # Arrange plots in 2 columns
    kind='hist',         # Make histogram plots
    bins=20,             # Use 20 bins per histogram
    color='black',       # Bar color
    edgecolor='black',   # Edge color
    alpha=0.7            # Transparency
)

# Customize the grid of plots
g.set_axis_labels('Bikes Hired', 'Frequency')  # Set labels for all plots
g.set_titles('{col_name}')  # Use season name as title for each subplot
g.fig.suptitle('Distribution of Bikes Hired by Season', y=1.03)  # Overall title

plt.show()

# FACETED HISTOGRAM BY MONTH
# Create separate histograms for each month

# Create a grid structure for the plots
g = sns.FacetGrid(bike, 
                  col='month_name',  # Create one plot per month
                  col_wrap=3         # Arrange in 3 columns
                  )

# Create histogram for each month
g.map(sns.histplot, 'bikes_hired', bins=20, alpha=0.7, edgecolor='black')
g.set_titles("{col_name}")  # Use month name as title
g.set_axis_labels('Bikes Hired', 'Frequency')  # Set axis labels
plt.tight_layout()  # Clean layout
plt.show()

# DENSITY PLOTS SECTION
# Density plots show smooth curves representing data distribution (like smoothed histograms)

# Simple density plot
plt.figure(figsize=(10, 6))
sns.kdeplot(data=bike, x='bikes_hired', fill=True, alpha=0.6)  # kde = kernel density estimate
plt.title('Density Plot of Bikes Rented')
plt.xlabel('Bikes Hired')
plt.ylabel('Density')  # Density = probability per unit
plt.grid(True, alpha=0.3)  # Add light grid
plt.show()

# Density plot with different colors for each season
plt.figure(figsize=(10, 6))
sns.kdeplot(data=bike, x='bikes_hired', hue='season_name', fill=True, alpha=0.3, common_norm=False)
# hue='season_name' creates different colored curves for each season
# common_norm=False means each curve represents that season's distribution independently
plt.title('Density Plot of Bikes Rented by Season')
plt.xlabel('Bikes Hired')
plt.ylabel('Density')
plt.show()

# Separate density plot for each season (faceted)
g = sns.FacetGrid(bike, col='season_name', col_wrap=2, sharex=False, sharey=False)
# sharex=False, sharey=False allows each plot to have its own scale
g.map(sns.kdeplot, 'bikes_hired', fill=True, alpha=0.6)
g.set_titles("{col_name}")
g.set_axis_labels('Bikes Hired', 'Density')
plt.tight_layout()
plt.show()

# Separate density plot for each month (faceted)
g = sns.FacetGrid(bike, col='month_name', col_wrap=3, 
                 col_order=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], 
                 sharex=False, sharey=False)
# col_order ensures months appear in calendar order
g.map(sns.kdeplot, 'bikes_hired', fill=True, alpha=0.6)
g.set_titles("{col_name}")
g.set_axis_labels('Bikes Hired', 'Density')
plt.tight_layout()
plt.show()

# BOXPLOT SECTION
# Boxplots show the median, quartiles, and outliers for different groups

# Boxplot by day of week
plt.figure(figsize=(10, 6))
sns.boxplot(data=bike, 
            x='day_of_week',  # X-axis: days of week
            y='bikes_hired')  # Y-axis: number of bikes hired
# Boxplot shows: median (middle line), 25th-75th percentiles (box), whiskers, outliers (dots)
plt.title('Boxplot of Bikes Hired by Day of the week')
plt.xlabel('Day of week')
plt.ylabel('Bikes Hired')
plt.grid(True, alpha=0.3)
plt.show()

# Boxplot by month (using month numbers)
plt.figure(figsize=(10, 6))
sns.boxplot(data=bike, x='month', y='bikes_hired')
plt.title('Boxplot of Bikes Hired by Month')
plt.xlabel('Month')
plt.ylabel('Bikes Hired')
plt.grid(True, alpha=0.3)
plt.show()

# Boxplot by month names (better labels)
month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
# Ensure months appear in calendar order (already done above, but repeated for clarity)
bike['month_name'] = pd.Categorical(bike['month_name'], categories=month_order, ordered=True)

plt.figure(figsize=(12, 6))
sns.boxplot(data=bike, x='month_name', y='bikes_hired')
plt.title('Boxplot of Bikes Hired by Month')
plt.xlabel('Month')
plt.ylabel('Bikes Hired')
plt.xticks(rotation=45)  # Rotate month labels 45 degrees for better readability
plt.grid(True, alpha=0.3)
plt.show()

# Boxplot by month with season coloring
plt.figure(figsize=(12, 6))
sns.boxplot(data=bike, x='month_name', y='bikes_hired', hue='season_name')
# hue='season_name' colors each box according to its season
plt.title('Boxplot of Bikes Hired by Month and Season')
plt.xlabel('Month')
plt.ylabel('Bikes Hired')
plt.xticks(rotation=45)
plt.legend(title='Season')  # Add legend showing season colors
plt.grid(True, alpha=0.3)
plt.show()

# VIOLIN PLOTS
# Violin plots combine boxplot info with density curves (shape shows distribution)

plt.figure(figsize=(12, 6))
sns.violinplot(data=bike, x='month_name', y='bikes_hired')
# Violin plot shows distribution shape (wide = more data, narrow = less data) plus quartiles
plt.title('Violin Plot of Bikes Hired by Month')
plt.xlabel('Month')
plt.ylabel('Bikes Hired')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.show()

# CORRELATION ANALYSIS SECTION
# Explore relationships between bike rentals and weather variables

# Bikes hired vs temperature with regression line
plt.figure(figsize=(10, 6))
sns.regplot(data=bike, 
            x='temp',          # X-axis: temperature
            y='bikes_hired',   # Y-axis: bikes hired
            scatter_kws={'alpha':0.5, 'color': 'black'})  # Make points semi-transparent and black
# regplot automatically fits and displays a regression line showing the trend
plt.title('Bikes Hired vs Mean Temperature')
plt.xlabel('Mean Temperature')
plt.ylabel('Bikes Hired')
plt.grid(True, alpha=0.3)
plt.show()

# Temperature vs bikes hired, colored by season
sns.lmplot(data=bike, 
           x='temp', 
           y='bikes_hired', 
           hue='season_name',  # Different colors and regression lines for each season
           scatter_kws={'alpha':0.6}, 
           height=6,    # Plot height
           aspect=1.5)  # Width = height * aspect
plt.title('Bikes Hired vs Mean Temperature by Season')
plt.xlabel('Mean Temperature')
plt.ylabel('Bikes Hired')
plt.grid(True, alpha=0.3)
plt.show()

# Separate temperature plots for each season (faceted)
g = sns.lmplot(data=bike, 
               x='temp', 
               y='bikes_hired', 
               col='season_name',  # Create separate plot for each season
               col_wrap=2,         # Arrange in 2 columns
               height=5, 
               aspect=1, 
               scatter_kws={'alpha':0.3})
g.set_axis_labels('Mean Temperature', 'Bikes Hired')
g.set_titles("{col_name}")
plt.tight_layout()
plt.show()

# Bikes hired vs humidity
plt.figure(figsize=(10, 6))
sns.regplot(data=bike, 
            x='humidity', 
            y='bikes_hired', 
            scatter_kws={'alpha':0.5, 'color': 'black'})
plt.title('Bikes Hired vs Humidity')
plt.xlabel('Humidity')
plt.ylabel('Bikes Hired')
plt.grid(True, alpha=0.3)
plt.show()

# Humidity vs bikes hired, colored by season
sns.lmplot(data=bike, 
           x='humidity', 
           y='bikes_hired', 
           hue='season_name',  # Show different seasons in different colors
           scatter_kws={'alpha':0.4}, 
           height=6, 
           aspect=1.5)
plt.title('Bikes Hired vs Humidity by Season')
plt.xlabel('Humidity')
plt.ylabel('Bikes Hired')
plt.grid(True, alpha=0.3)
plt.show()

# Bikes hired vs atmospheric pressure
plt.figure(figsize=(10, 6))
sns.regplot(data=bike, x='pressure', y='bikes_hired', scatter_kws={'alpha':0.5})
plt.title('Bikes Hired vs Pressure')
plt.xlabel('Pressure')
plt.ylabel('Bikes Hired')
plt.grid(True, alpha=0.3)
plt.show()