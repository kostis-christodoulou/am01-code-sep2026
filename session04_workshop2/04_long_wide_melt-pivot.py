# Import the libraries we need for data manipulation and visualization
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter

# =============================================================================
# SECTION 1: GRADES DATASET - TRANSFORMING DATA FROM WIDE TO LONG FORMAT
# =============================================================================

# Create a sample dataset of student grades
# Each row represents one student, each column represents a different assignment type
data = {
    'student_id': [2457625, 1758293, 1622247],      # Unique identifier for each student
    'final_exam': [79, 92, 71],                     # Final exam scores
    'midterm': [68, 73, 87],                        # Midterm exam scores  
    'individual_project': [71, 67, 74],             # Individual project scores
    'group_project': [83, 56, 77]                   # Group project scores
}

# Convert our dictionary into a pandas DataFrame
# This creates a "wide" format where each assignment type is its own column
grades_df = pd.DataFrame(data)

# Display the original wide format for comparison
print("--- Original (Wide) DataFrame ---")
print(grades_df)

# Transform from "wide" to "long" format using pd.melt()
# Wide format: each assignment type is a separate column
# Long format: all assignment types go into one column, scores into another
long_grades_df = pd.melt(
    grades_df,                          # The DataFrame to transform
    id_vars=['student_id'],             # Columns to keep as identifiers (don't melt these)
    var_name='assignment_type',         # Name for the new column that will contain the old column names
    value_name='score'                  # Name for the new column that will contain the values
)

# Display the transformed long format
print("\n--- Transformed (Long) DataFrame ---")
print(long_grades_df)

# Calculate summary statistics for each assignment type
# Group all students by assignment type, then calculate stats for the scores
summary_stats = (long_grades_df
                 .groupby('assignment_type')['score']    # Group by assignment, focus on score column
                 .agg(                                   # Apply multiple aggregation functions
                     min_score='min',                    # Minimum score for this assignment type
                     max_score='max',                    # Maximum score for this assignment type  
                     mean_score='mean',                  # Average score for this assignment type
                     std_dev='std'                       # Standard deviation (spread of scores)
                 )
                 .reset_index()                          # Convert back to regular DataFrame
                )

# Display the summary statistics
print("\n--- Summary Statistics by Assignment Type ---")
print(summary_stats)

# =============================================================================
# SECTION 2: UK BABY NAMES DATASET - DATA ANALYSIS AND VISUALIZATION
# =============================================================================

# Load the UK baby names dataset from a CSV file
# This dataset contains baby name popularity data over time
babynames = pd.read_csv('data/ukbabynames.csv')

# Display the first 10 rows to understand the data structure
print(babynames.head(10))

# Show information about the dataset structure and data types
print("\n--- Dataset Information ---")
print(babynames.info())

# ANALYSIS 1: Calculate total births across all years in the dataset
total_births_all_time = babynames['n'].sum()
print(f"\nTotal births across all years: {total_births_all_time:,}")

# ANALYSIS 2: Calculate total births for each year
# Group all records by year, then sum the 'n' column (number of births)
yearly_births = (babynames
                 .groupby('year')                       # Group by year
                 .agg({'n': 'sum'})                     # Sum the birth counts
                 .reset_index()                         # Convert back to DataFrame
                )

print("\n--- Total Births by Year ---")
print(yearly_births)

# ANALYSIS 3: Calculate total births by year AND sex (boys vs girls)
# This gives us separate totals for male and female births each year
births_by_year_sex = (babynames
                      .groupby(['year', 'sex'])         # Group by both year and sex
                      .agg(total_births=('n', 'sum'))   # Sum births, name the result 'total_births'
                      .reset_index()                    # Convert back to DataFrame
                     )




print("\n--- Total Births by Year and Sex ---")
print(births_by_year_sex)

# ANALYSIS: From Long to Wide Format
# This involves reshaping the data to get male/female counts in separate columns

# ANALYSIS 3a: Calculate the percentage of boys born each year
# This involves reshaping the data to get male/female counts in separate columns
births_by_year_sex_wide =(births_by_year_sex
    # Reshape data so 'F' and 'M' become separate columns
    # pivot() transforms rows into columns based on the 'sex' values
    .pivot(index='year',                     # Rows will be years
            columns='sex',                    # Columns will be 'F' and 'M' 
            values='total_births')
    ) 


pct_boys_df = (births_by_year_sex_wide
               #  Calculate percentage of boys using the new F and M columns
               # lambda df creates a temporary function that calculates the percentage
               .assign(pct_boys=lambda df: (df['M'] / (df['F'] + df['M'])) * 100)
               .reset_index()                           # Convert back to regular DataFrame
              )

print("\n--- Percentage of Boys by Year ---")
print(pct_boys_df.head())

# VISUALIZATION 1: Plot the percentage of boys born over time
# Set up the plot size and style
plt.figure(figsize=(12, 7))                          # Create a wide figure for time series
sns.set_style("whitegrid")                           # Use seaborn's clean grid style

# Create the line plot
ax = sns.lineplot(data=pct_boys_df,                  # Use our calculated percentage data
                  x='year',                          # Years on x-axis
                  y='pct_boys',                      # Percentage of boys on y-axis
                  color='blue')                      # Blue line color

# Customize the plot appearance
ax.set_title('Percentage of Boys Born Each Year',    # Main title
             fontsize=16, 
             fontweight='bold',
             loc='left')                             # Left-align title
ax.set_ylabel("Percentage (%)")                     # Y-axis label
ax.set_xlabel("Year")                               # X-axis label

# Format the y-axis to show percentage signs
ax.yaxis.set_major_formatter(PercentFormatter())

# Adjust layout to prevent any text cutoff
plt.tight_layout()
plt.show()

# ANALYSIS 4: Count unique names by year and sex
# This tells us how many different names were used each year for boys vs girls
unique_names_df = (babynames
                   .groupby(['year', 'sex'])            # Group by year and sex
                   .agg(unique_names=('name', 'nunique')) # Count unique names (nunique = number of unique)
                   .reset_index()                       # Convert back to DataFrame
                  )

print("\n--- Unique Names by Year and Sex ---")
print(unique_names_df.head())

# VISUALIZATION 2: Plot unique names over time for both sexes
# Create a color mapping to control the line colors
color_palette = {'M': 'blue', 'F': 'red'}          # Boys = blue, Girls = red

# Set up the plot
plt.figure(figsize=(12, 7))
sns.set_style("whitegrid")

# Create line plot with separate lines for male and female
ax = sns.lineplot(data=unique_names_df, 
                  x='year',                          # Years on x-axis
                  y='unique_names',                  # Number of unique names on y-axis
                  hue='sex',                         # Create separate lines for M and F
                  style='sex',                       # Give lines different styles (optional)
                  palette=color_palette)             # Use our custom colors

# Customize the plot appearance
ax.set_title('Number of Unique Male and Female Names in the UK Over Time', 
             fontsize=16, 
             fontweight='bold',
             loc='left')                             # Left-align title
ax.set_ylabel("Number of Unique Names")              # Y-axis label
ax.set_xlabel("Year")                               # X-axis label
ax.legend(title='Sex')                              # Improve legend title

# Adjust layout and display
plt.tight_layout()
plt.show()

# ANALYSIS 5: Calculate the ratio of female to male unique names
# This shows whether there are more unique girl names or boy names each year
ratio_df = (unique_names_df
            # Step 1: Reshape data so F and M counts are in separate columns
            .pivot(index='year',                     # Rows will be years
                   columns='sex',                    # Columns will be 'F' and 'M'
                   values='unique_names')            # Values will be unique name counts
            
            # Step 2: Calculate the ratio of female to male unique names
            # A ratio > 1 means more unique female names, < 1 means more unique male names
            .assign(ratio=lambda df: df['F'] / df['M'])
            .reset_index()                           # Convert back to regular DataFrame
           )

print("\n--- Ratio of Female to Male Unique Names ---")
print(ratio_df.head())

# VISUALIZATION 3: Plot the ratio over time
plt.figure(figsize=(12, 7))
sns.set_style("whitegrid")

# Create the line plot showing the ratio trend
ax = sns.lineplot(data=ratio_df, 
                  x='year',                          # Years on x-axis
                  y='ratio',                         # Female/Male ratio on y-axis
                  color='hotpink')                   # Pink color for the line

# Customize the plot appearance
ax.set_title('Ratio of Unique UK Female to Male Names Over Time', 
             fontsize=16, 
             fontweight='bold',
             loc='left')                             # Left-align title
ax.set_ylabel("Ratio (Unique Female Names / Unique Male Names)")  # Descriptive y-axis label
ax.set_xlabel("Year")                               # X-axis label

# Adjust layout and display
plt.tight_layout()
plt.show()

# =============================================================================
# KEY CONCEPTS DEMONSTRATED:
# 
# 1. DATA RESHAPING:
#    - pd.melt(): Transforms wide format to long format
#    - .pivot(): Transforms long format to wide format
#
# 2. DATA AGGREGATION:
#    - .groupby(): Groups data by one or more columns
#    - .agg(): Applies functions (sum, mean, count, etc.) to grouped data
#    - .nunique(): Counts unique values in a column
#
# 3. DATA MANIPULATION:
#    - .assign(): Adds new calculated columns
#    - .reset_index(): Converts grouped data back to regular DataFrame
#    - lambda functions: Create simple calculations inline
#
# 4. VISUALIZATION WITH SEABORN:
#    - sns.lineplot(): Creates line plots with automatic styling
#    - hue parameter: Creates separate lines for different categories
#    - palette parameter: Controls colors used in the plot
#    - Integration with matplotlib for additional customization
# =============================================================================