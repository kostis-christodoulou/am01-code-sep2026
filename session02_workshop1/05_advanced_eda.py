# Advanced Pandas EDA with Method Chaining - Beginner's Guide
# This script demonstrates "dplyr-like" method chaining in pandas, similar to R's tidyverse

# =============================================================================
# IMPORTING LIBRARIES
# =============================================================================

import pandas as pd              # Main data manipulation library
import matplotlib.pyplot as plt  # Basic plotting
import matplotlib.ticker as mticker  # For formatting plot axes
import seaborn as sns           # Statistical visualization (makes pretty plots)
import numpy as np              # Numerical operations
from gapminder import gapminder # Sample dataset about countries over time
import janitor                  # For cleaning column names (like R's janitor::clean_names())
from skimpy import skim         # For generating comprehensive summary statistics (like R's skimr)
import warnings
warnings.filterwarnings('ignore')  # Hide warning messages for cleaner output

# Set plotting style for better looking graphs
plt.style.use('seaborn-v0_8')   # Use seaborn's clean styling
sns.set_palette("husl")         # Use a nice color palette

# =============================================================================
# METHOD CHAINING BASICS - THREE EQUIVALENT APPROACHES
# =============================================================================

# METHOD CHAINING is like R's pipe operator (%>%) - it lets you chain operations
# together in a readable, step-by-step manner instead of creating intermediate variables

# APPROACH 1: Step-by-step (creates intermediate variables)
# This is like doing each operation separately - easy to debug but verbose
gapminder_2007_step1 = gapminder[gapminder['year'] == 2007]                    # Filter for 2007
gapminder_2007_step2 = gapminder_2007_step1[['country', 'continent', 'lifeExp']]  # Select columns
gapminder_2007_step3 = gapminder_2007_step2.sort_values('lifeExp', ascending=False)  # Sort by life expectancy
gapminder_2007_step3.head()


# APPROACH 2: Nested functions (everything in one line)
# This is compact but can be hard to read when it gets complex
gapminder_2007_nested = gapminder[gapminder['year'] == 2007][['country', 'continent', 'lifeExp']].sort_values('lifeExp', ascending=False)
gapminder_2007_nested.head()

# APPROACH 3: Method chaining (RECOMMENDED - like R's dplyr)
# This is readable, compact, and easy to modify
# Each line represents one data transformation step
gapminder_2007_chained = (
    gapminder
    .query('year == 2007')                           # Filter: keep only 2007 data
    [['country', 'continent', 'lifeExp']]            # Select: keep only these columns
    .sort_values('lifeExp', ascending=False)         # Sort: highest life expectancy first
)
gapminder_2007_chained.head()


# --- AGGREGATE STATISTICS BY CONTINENT ---

# The goal here is to calculate summary statistics for life expectancy ('lifeExp')
# for each continent. We want to find the average, minimum, maximum, and
# standard deviation of life expectancy across all years and countries within a continent.

# We start with the 'gapminder' DataFrame and chain our operations together.
agg_stats_continent = (
    gapminder
    
    # 1. GROUP THE DATA
    # .groupby(['continent']) tells pandas to collect all rows with the same
    # 'continent' value into separate groups (e.g., one group for 'Asia', one for 'Europe').
    # All future calculations will be performed on each of these groups independently.
    .groupby(['continent'])
    
    # 2. AGGREGATE (SUMMARIZE) THE GROUPS
    # .agg() is used to perform multiple calculations on the grouped data at once.
    # We are creating four new columns with specific calculations:
    .agg(
        # New column 'mean_lifexp' = the 'mean' of the 'lifeExp' column for that group.
        mean_lifexp=('lifeExp', 'mean'),
        
        # New column 'min_lifexp' = the 'min' (minimum) value of 'lifeExp'.
        min_lifexp=('lifeExp', 'min'),
        
        # New column 'max_lifexp' = the 'max' (maximum) value of 'lifeExp'.
        max_lifexp=('lifeExp', 'max'),
        
        # New column 'std_lifexp' = the 'std' (standard deviation) of 'lifeExp'.
        std_lifexp=('lifeExp', 'std')
    )
    
    # 3. EXPLAINING .reset_index()
    # When you use .groupby(), pandas automatically makes the grouping column ('continent')
    # the "index" of the new DataFrame. The index acts like a row label.
    # This means 'continent' is not a regular data column, which can be confusing.
    #
    # .reset_index() converts that special index back into a regular column.
    # So, the 'continent' labels move from being the index on the side, into the
    # first data column of our table. This makes the DataFrame "tidy" and easier to use.
).reset_index()

# Display the final summary table for continents.
# It will have 5 columns: 'continent', 'mean_lifexp', 'min_lifexp', 'max_lifexp', 'std_lifexp'.
print("--- Aggregate Statistics by Continent ---")
print(agg_stats_continent)

# skim() is a useful function from the 'skimpy' library that provides a quick
# statistical summary of the new DataFrame we just created.
# skim(agg_stats_continent)

# --- AGGREGATE STATISTICS BY CONTINENT and YEAR ---

# This is very similar to the first example, but we want to be more specific.
# We will calculate the same statistics, but for each combination of continent AND year.
# For example, we'll get stats for 'Asia in 1952', 'Asia in 1957', 'Europe in 1952', etc.

agg_stats_continent_year = (
    gapminder
    
    # 1. GROUP THE DATA BY TWO COLUMNS
    # By passing a list of columns ['continent', 'year'], we tell pandas to create
    # a group for every unique combination of these two values.
    .groupby(['continent', 'year'])
    
    # 2. AGGREGATE THE GROUPS
    # The .agg() function works exactly the same way, but now it performs the
    # calculations on these smaller, more specific groups (e.g., on all data for Asia in 1952).
    .agg(
        mean_lifexp=('lifeExp', 'mean'),
        min_lifexp=('lifeExp', 'min'),
        max_lifexp=('lifeExp', 'max'),
        std_lifexp=('lifeExp', 'std')
    )
    
    # 3. RESET THE INDEX (AGAIN)
    # Since we grouped by two columns, pandas creates a "MultiIndex" using both
    # 'continent' and 'year'.
    # .reset_index() is smart enough to handle this too. It converts both index levels
    # ('continent' and 'year') back into regular data columns at the front of the table.
).reset_index()

# Display the final summary table for continents and years.
# It will have 6 columns: 'continent', 'year', and the four calculated statistics.
agg_stats_continent_year

# Use skim() to get a quick overview of this more detailed DataFrame.
# skim(agg_stats_continent_year)


# =============================================================================
# LOADING AND CLEANING DATA WITH PYJANITOR
# =============================================================================

print("\n=== LOADING AND CLEANING CA CONTRIBUTIONS DATA ===\n")

# Load California political contributions data
# Method chaining starts here - each line does one transformation
CA_contributions = (
    pd.read_csv("data/CA_contributors_2016.csv")     # Load the CSV file
    .clean_names()                                   # Clean column names (pyjanitor function)
    # .clean_names() converts "Column Name" to "column_name" 
    # - Makes all lowercase
    # - Replaces spaces with underscores
    # - Removes special characters
    # This is equivalent to R's janitor::clean_names()
)

print("Data shape:", CA_contributions.shape)
print("Clean column names:", list(CA_contributions.columns))
print("\nFirst few rows:")
CA_contributions.head()

# =============================================================================
# COMPREHENSIVE DATA EXPLORATION WITH SKIMPY
# =============================================================================

print("\n=== COMPREHENSIVE DATA EXPLORATION WITH SKIMPY ===")

# skim() provides a comprehensive overview similar to R's skimr package
# It shows for each column:
# - Data type and missing values
# - Summary statistics (mean, std, min, max, quartiles)
# - Unique value counts
# - Distribution histograms (for numeric columns)
# - Top frequent values (for categorical columns)
print("\nComprehensive summary of CA Contributions data:")
print("=" * 60)
skim(CA_contributions)

# =============================================================================
# CALCULATING AVERAGES WITH METHOD CHAINING
# =============================================================================

print("\n=== CALCULATING DONATION AVERAGES ===\n")

# Calculate Trump's average donation using method chaining
trump_avg = (
    CA_contributions
    .query("cand_nm == 'Trump, Donald J.'")         # Filter: only Trump donations
    ['contb_receipt_amt']                           # Select: only the amount column
    .mean()                                         # Calculate: mean of the amounts
)
print(f"Average donation for Trump: ${trump_avg:.2f}")

# Calculate average donation by ALL candidates using method chaining
candidate_avg = (
    CA_contributions
    .groupby('cand_nm')                             # Group: by candidate name
    ['contb_receipt_amt']                           # Select: the amount column
    .mean()                                         # Calculate: mean for each group
    .sort_values(ascending=False)                   # Sort: highest average first
    .round(2)
).reset_index()

print("\nTop 10 candidates by average donation:")
candidate_avg.head(10)

# Calculate multiple donation statistics by candidate using method chaining
candidate_summary = (
    CA_contributions

    .groupby('cand_nm')
    .agg(
        # New Column 'avg_donation': Calculate the 'mean' (average) of donation amounts.
        avg_donation=('contb_receipt_amt', 'mean'),

        # New Column 'num_donations': 'count' the number of individual donations.
        num_donations=('contb_receipt_amt', 'count'),

        # New Column 'total_raised': Calculate the 'sum' of all donation amounts.
        total_raised=('contb_receipt_amt', 'sum')
    )
    .reset_index()

    # 3. SORT THE RESULTS
    .sort_values(by='total_raised', ascending=False)
).round(2) # Round all numeric columns to 2 decimal places

# --- Display the Results ---

# The result 'candidate_summary' is now a DataFrame with columns for each statistic.
print("\nTop 10 candidates by total amount raised:")
candidate_summary.head(10)

# You could also easily sort by a different column, like the average donation:
# print("\nTop 10 candidates by average donation:")
# print(candidate_summary.sort_values(by='avg_donation', ascending=False).head(10))

# =============================================================================
# CREATING VISUALIZATIONS - TOP FUNDRAISERS
# =============================================================================


# --- Create the Plot ---
plt.figure(figsize=(12, 8))

# Create the bar plot and store the Axes object in the 'ax' variable
ax = sns.barplot(
    data=candidate_summary.head(10),
    x='total_raised',
    y='cand_nm',
    orient='h',
    color='skyblue',
    edgecolor='black'
)

# --- Add Text Annotations to the Right of Each Bar ---

# We loop through each of the bars ('patches') in our plot.
for bar in ax.patches:
    # Get the width of the bar (the total amount raised).
    bar_width = bar.get_width()
    
    # Get the y-coordinate of the bar's center.
    y_position = bar.get_y() + bar.get_height() / 2
    
    # Create the text label in millions shorthand (e.g., "$25.7M").
    label_text = f"${bar_width / 1_000_000:.1f}M"
    
    # Define a small horizontal offset to create a gap between the bar and the text.
    # This offset is a small percentage of the total x-axis range.
    offset = ax.get_xlim()[1] * 0.01
    
    # Place the text on the plot.
    ax.text(
        bar_width + offset,      # x-position: end of the bar PLUS a small offset.
        y_position,              # y-position: the center of the bar.
        label_text,              # The text to display.
        ha='left',               # MODIFIED: Horizontal alignment is now 'left'.
        va='center',             # Vertical alignment: center it vertically.
        color='black',           # Text color.
        fontweight='bold',       # Make the text bold.
        size=10                  # Set the font size.
    )

# --- Customize the Plot ---
plt.xlabel('Total Donation Amount ($)')
plt.ylabel('Candidate')
plt.title('Top 10 Fundraisers in California (2016)')

# Format x-axis ticks to show dollar amounts.
ax.xaxis.set_major_formatter(mticker.StrMethodFormatter('${x:,.0f}'))
ax.ticklabel_format(style='plain', axis='x')

# IMPORTANT: Increase the right-side limit of the x-axis to make space for the labels.
# A larger multiplier (e.g., 1.15) is needed now that text is outside the bars.
ax.set_xlim(right=ax.get_xlim()[1] * 1.15)

plt.tight_layout()  # Adjust layout to prevent text cutoff
plt.show()

# =============================================================================
# FROM ZIPCODES TO CITIES
# =============================================================================


#  Load, clean, and filter the zipcodes data
zipcodes_ca = (
    pd.read_csv("data/zip_code_database.csv")
    .clean_names()
    .query("state == 'CA'")

    # Ensure the 'zip' column in this DataFrame is also a string.
    .assign(zip=lambda df: df['zip'].astype(str))
)

top_50_cities = (
    CA_contributions
    # Ensure the 'zip' column in this DataFrame is also a string.
    .assign(zip=lambda df: df['zip'].astype(str))

    # .merge() is pandas' join function. 'how="left"' specifies the join type.
    .merge(zipcodes_ca, on='zip', how='left')
    
    # .groupby() groups the DataFrame by the specified columns.
    .groupby(['primary_city', 'cand_nm'])
     
    # .agg() calculates summary statistics for each group.
    # We create a new column 'total' by summing 'contb_receipt_amt'.
    .agg(total=('contb_receipt_amt', 'sum'))

    # .reset_index() converts the grouped DataFrame back to a regular DataFrame.
    .reset_index()

    # .sort_values() orders the result. 'ascending=False' is equivalent to desc().
    .sort_values(by='total', ascending=False)
    
    # This step is crucial. .groupby() makes the grouping columns the index.
    # .reset_index() converts them back to regular columns.
    .reset_index()
    
    # .head() selects the first 50 rows of the sorted DataFrame.
    .head(50)
)

# Display the final result
top_50_cities

# =============================================================================
# MOVIE DATA ANALYSIS
# =============================================================================

print("\n=== ANALYZING MOVIE BOX OFFICE DATA ===\n")

# Load movie data with method chaining
movies = (
    pd.read_csv("data/movies.csv")
    .clean_names()                                   # Clean column names
)

print("Movies data shape:", movies.shape)
print("Clean column names:", list(movies.columns))

# =============================================================================
# COMPREHENSIVE MOVIE DATA EXPLORATION WITH SKIMPY
# =============================================================================

print("\n=== COMPREHENSIVE MOVIE DATA EXPLORATION ===")

# Use skimpy to get a detailed overview of the movie dataset
# This will show us:
# - Missing values in each column
# - Distribution of numerical variables (gross, budget, etc.)
# - Most frequent categories (genres, directors, etc.)
# - Data quality issues to be aware of
print("\nComprehensive summary of Movies data:")
print("=" * 60)
skim(movies)

# After seeing the skim output, we can make informed decisions about:
# - Which columns have too many missing values to use
# - Whether we need to clean or transform certain variables
# - What the typical ranges are for numerical variables
print("\n" + "=" * 60)

# Create top 20 movies by gross earnings using method chaining
top_20_movies = (
    movies
    .nlargest(20, 'gross')                          # Get top 20 by gross earnings
    [['title', 'gross']]                            # Select only title and gross columns
    .sort_values('gross', ascending=False)          # Sort from highest to lowest
)

print("Top 20 movies by gross earnings:")
print(top_20_movies)

# Create horizontal bar chart
plt.figure(figsize=(12, 10))

sns.barplot(
    data=top_20_movies,
    x='gross',                                      # Gross earnings on x-axis
    y='title',                                      # Movie titles on y-axis
    color='darkblue'                                # Professional dark blue color
)

# Format the plot professionally
ax = plt.gca()
# Format x-axis as millions of dollars (e.g., "$500.2M")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f'${x/1e6:.1f}M'))

plt.xlabel('Gross Earnings (Millions USD)')
plt.ylabel('')  # Remove y-axis label (titles are self-explanatory)
plt.title('Top 20 Movies by Box Office Gross Earnings')
plt.tight_layout()
plt.show()

# =============================================================================
# ADVANCED VISUALIZATION - MOVIES BY GENRE
# =============================================================================

print("\n=== MOVIES BY GENRE ANALYSIS ===\n")

# Create top 20 movies with genre information
top_20_movies_with_genre = (
    movies
    .nlargest(20, 'gross')                          # Get top 20 by earnings
    [['title', 'gross', 'genre']]                   # Include genre information
    .sort_values('gross', ascending=False)          # Sort by earnings
)

# Create colorful bar chart by genre
plt.figure(figsize=(12, 10))

sns.barplot(
    data=top_20_movies_with_genre,
    x='gross',
    y='title', 
    hue='genre',                                    # Color bars by genre
    dodge=False                                     # Don't separate bars by genre
)

# Professional formatting
ax = plt.gca()
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f'${x/1e6:.1f}M'))

plt.xlabel('Gross Earnings (Millions USD)')
plt.ylabel('')
plt.title('Top 20 Movies by Genre\nAction and Adventure Dominate the Box Office')

# Position legend outside plot area
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.show()

# =============================================================================
# CREATING HEATMAPS WITH EUROPEAN LIFE EXPECTANCY
# =============================================================================

print("\n=== CREATING LIFE EXPECTANCY HEATMAP ===\n")

# Prepare European life expectancy data using method chaining
europe_data = (
    gapminder
    .query("continent == 'Europe'")                 # Filter: only European countries
    .copy()                                         # Make a copy to avoid warnings
    .assign(
        # Reorder countries by median life expectancy (lowest to highest)
        country=lambda df: pd.Categorical(
            df['country'],
            categories=df.groupby('country')['lifeExp'].median().sort_values().index,
            ordered=True
        )
    )
)

# Create pivot table for heatmap (countries as rows, years as columns)
heatmap_data = europe_data.pivot(index='country', 
                                 columns='year', 
                                 values='lifeExp')

# Calculate the median life expectancy for each country to determine the order
country_order = heatmap_data.median(axis=1).sort_values(ascending=False).index

# Reorder the heatmap_data based on the calculated median life expectancy
heatmap_data = heatmap_data.loc[country_order]

# Create the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(
    heatmap_data, 
    cmap='viridis',                                 # Use viridis colormap (colorblind-friendly)
    cbar_kws={'label': 'Life Expectancy (years)'}  # Label the color bar
)
plt.title('Life Expectancy in European Countries (1952-2007)')
plt.xlabel('Year')
plt.ylabel('Country (ordered by median life expectancy)')
plt.tight_layout()
plt.show()

# =============================================================================
# CREATING FACETED PLOTS (SMALL MULTIPLES)
# =============================================================================

print("\n=== CREATING FACETED TIME SERIES PLOTS ===\n")

# 1. Determine the order of countries based on median life expectancy (min to max)
country_order = europe_data.groupby('country')['lifeExp'].median().sort_values(ascending=True).index

# 2. Create a FacetGrid with the specified order and smaller plot sizes
g = sns.FacetGrid(
    europe_data, 
    col='country', 
    col_order=country_order,  # Apply the calculated order
    col_wrap=7,               # Wrap after 7 plots to create rows
    height=2.5,               # Reduce the height of each plot
    aspect=1.1,               # Adjust aspect ratio (width/height)
    sharex=True, 
    sharey=True
)

# Map a line plot onto each facet (subplot)
g.map(sns.lineplot, 'year', 'lifeExp', color='steelblue', linewidth=2)

# Set the titles for each subplot
g.set_titles("{col_name}")

# Add a light grid to each subplot for better readability
g.map(plt.grid, True, alpha=0.3)

# Set the axis labels for the entire grid
g.set_axis_labels("Year", "Life Expectancy")

# Add an overall title for the entire figure
plt.suptitle('Life Expectancy Trends in Europe (1952-2007)\nOrdered by Median Life Expectancy', 
             y=1.03, fontsize=14)

# Adjust the layout to ensure titles and labels don't overlap
plt.tight_layout()

# Display the plot
plt.show()

print("\n=== ANALYSIS COMPLETE ===")
print("This script demonstrated:")
print("1. Method chaining (dplyr-like operations)")
print("2. Data cleaning with pyjanitor")  
print("3. Creating new variables with .assign()")
print("4. Professional data visualizations")
print("5. Faceted plots and heatmaps")

# =============================================================================
# KEY CONCEPTS RECAP FOR BEGINNERS:
# =============================================================================

# METHOD CHAINING BENEFITS:
# - Readable: Each line represents one transformation
# - No intermediate variables cluttering your namespace  
# - Easy to modify: Add/remove/reorder steps easily
# - Similar to R's dplyr pipe operator (%>%)

# PYJANITOR BENEFITS:
# - .clean_names(): Automatically creates consistent column names
# - Many other useful functions for data cleaning

# SKIMPY BENEFITS:
# - skim(): Provides comprehensive data overview similar to R's skimr
# - Shows data types, missing values, summary stats, and distributions
# - Helps identify data quality issues early in analysis
# - More informative than basic .info() and .describe()

# ASSIGN() FUNCTION:
# - Creates new columns without breaking the chain
# - Can reference previously created columns in the same assign()
# - Uses lambda functions for transformations

# QUERY() FUNCTION:  
# - Filters data using string expressions
# - More readable than boolean indexing for complex conditions
# - Similar to SQL WHERE clauses