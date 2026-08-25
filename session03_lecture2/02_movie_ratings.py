# =============================================================================
# MOVIE DATA ANALYSIS - A BEGINNER'S GUIDE
# =============================================================================
"""
This script analyzes a dataset of movies and their IMDB ratings by genre.
We'll learn how to:
1. Load and explore data
2. Filter data based on conditions
3. Create various types of plots
4. Calculate confidence intervals for statistical analysis

Think of this as a step-by-step tutorial for data analysis in Python!
"""

# =============================================================================
# 1. IMPORT LIBRARIES (Like getting your tools ready)
# =============================================================================

# Think of libraries as toolboxes with specialized tools for different jobs:

import pandas as pd          # THE data analysis library - like Excel but much more powerful
import seaborn as sns        # Makes beautiful statistical plots easily
import matplotlib.pyplot as plt  # The basic plotting library (seaborn uses this underneath)
import numpy as np           # Math operations and working with arrays of numbers
from scipy import stats     # Advanced statistical functions
from skimpy import skim     # Gives us nice data summaries (like R's skim function)

# What each library does:
# - pandas: Imagine Excel, but programmable and much more powerful
# - seaborn: Like having a professional graphic designer for your charts
# - matplotlib: The basic drawing canvas that seaborn paints on
# - numpy: A calculator on steroids for working with lots of numbers at once
# - scipy.stats: Your statistics textbook turned into code
# - skimpy: Gives you a "quick health check" of your dataset

# =============================================================================
# 2. INITIAL SETUP AND DATA LOADING
# =============================================================================

# Set up how we want our plots to look (like choosing a theme)
sns.set_style("whitegrid")  # Clean white background with helpful grid lines
sns.set_style("ticks")      # This overwrites the previous line - keeps clean look with borders

# Load our data from a CSV file
# CSV = Comma Separated Values (think of it as a simple Excel file)
# A DataFrame is pandas' version of a spreadsheet - rows and columns of data
try:
    # Try to load the file - this is called "error handling"
    movies = pd.read_csv('data/movies.csv')
    print("✅ Data loaded successfully!")
except FileNotFoundError:
    # If the file doesn't exist, tell the user instead of crashing
    print("❌ Error: 'data/movies.csv' not found. Make sure the CSV file is in a 'data' directory.")
    print("   The file should contain columns like: title, genre, rating, year, etc.")

# =============================================================================
# 3. EXPLORING OUR DATA (Getting to know what we're working with)
# =============================================================================

print("\n" + "="*50)
print("📊 EXPLORING OUR DATASET")
print("="*50)

# .info() is like getting a "summary sheet" of your data
print("\n--- Dataset Overview ---")
print("This tells us:")
print("- How many rows (movies) and columns (features) we have")
print("- What type of data is in each column (numbers, text, etc.)")
print("- If there are any missing values")
movies.info()

# .head() shows us the first few rows - like peeking at the top of a spreadsheet
print("\n--- First 5 Movies in Our Dataset ---")
print("This gives us a preview of what each row looks like:")
print(movies.head())

# skim() gives us detailed statistics - like having a data analyst summarize everything
print("\n--- Detailed Data Summary ---")
print("This shows statistics like averages, ranges, and data quality:")
skim(movies)

# Count movies by genre - this is like making a tally chart
print("\n--- How Many Movies Per Genre? ---")
print("Before we do analysis, let's see what genres we have:")
genre_counts = movies['genre'].value_counts()  # Counts how many times each genre appears
print(genre_counts)
print(f"\nWe have {len(genre_counts)} different genres in total")

# =============================================================================
# 4. FILTERING THE DATA (Cleaning up for better analysis)
# =============================================================================

print("\n" + "="*50)
print("🧹 CLEANING OUR DATA")
print("="*50)

# Why filter? If a genre only has 1-2 movies, we can't draw meaningful conclusions
# It's like trying to judge all Italian restaurants based on just one meal!

print("We're going to keep only genres with more than 5 movies.")
print("This ensures our analysis is based on enough data to be meaningful.\n")

# Step 1: Find genres with more than 5 movies
# This creates a True/False list: True for genres with >5 movies, False otherwise
genres_to_keep_mask = genre_counts > 5
print("Genres with more than 5 movies:")
print(genres_to_keep_mask[genres_to_keep_mask == True])  # Show only the True ones

# Step 2: Get the actual genre names (not just True/False)
genres_to_keep = genre_counts[genres_to_keep_mask].index.tolist()
print(f"\nKeeping these {len(genres_to_keep)} genres: {genres_to_keep}")

# Step 3: Filter our main dataset
# .isin() checks if each movie's genre is in our "keep" list
original_movie_count = len(movies)  # Remember how many we started with
movies = movies[movies['genre'].isin(genres_to_keep)].copy()

print("\n📈 Filtering Results:")
print(f"   Started with: {original_movie_count} movies")
print(f"   Kept: {len(movies)} movies")
print(f"   Removed: {original_movie_count - len(movies)} movies")

# =============================================================================
# 5. CREATING VISUALIZATIONS (Making our data tell stories)
# =============================================================================

print("\n" + "="*50)
print("📈 CREATING VISUALIZATIONS")
print("="*50)
print("Now we'll create different types of plots to understand our data...")

# PLOT 1: BOXPLOTS
# Boxplots show the "five-number summary": min, 25th percentile, median, 75th percentile, max
# Plus any outliers (unusual values)
print("\n🎯 Creating boxplots...")
print("Boxplots show:")
print("- The middle line is the median (middle value)")
print("- The box shows where 50% of the data falls")
print("- The 'whiskers' (lines) show the range")
print("- Dots are outliers (unusually high/low values)")

# FacetGrid creates multiple subplots - one for each genre
g_box = sns.FacetGrid(movies, 
                      col="genre",           # Make one plot per genre
                      hue="genre",          # Color each genre differently
                      col_wrap=4,           # Put 4 plots per row
                      sharex=True,          # All plots have same x-axis
                      sharey=True)          # All plots have same y-axis
g_box.map(sns.boxplot, 'rating')            # Create boxplot of ratings for each genre
g_box.fig.suptitle("Distribution of IMDB ratings by film genre (genres > 5 movies)", y=1.02)
g_box.set_axis_labels("Film Rating", "")
g_box.set_titles("{col_name}")
plt.show()

# PLOT 2: HISTOGRAMS
# Histograms show the shape of your data - like stacking coins by value
print("\n📊 Creating histograms...")
print("Histograms show:")
print("- How many movies fall into each rating range")
print("- The 'shape' of the data (normal, skewed, etc.)")
print("- Common vs. rare rating values")

g_hist = sns.FacetGrid(movies, 
                       col="genre", 
                       hue="genre",
                       col_wrap=4, 
                       sharex=True, 
                       sharey=False)        # Different genres might have different counts
g_hist.map(sns.histplot, 'rating')
g_hist.fig.suptitle("Distribution of IMDB ratings by film genre (genres > 5 movies)", y=1.02)
g_hist.set_axis_labels("Film Rating", "Count")
g_hist.set_titles("{col_name}")
plt.show()

# PLOT 3: ECDF (EMPIRICAL CUMULATIVE DISTRIBUTION FUNCTION)
# This shows what percentage of movies have a rating at or below each value
print("\n📈 Creating ECDF plots...")
print("ECDF plots show:")
print("- What % of movies have ratings ≤ any given value")
print("- Steeper slopes mean more movies at that rating")
print("- Useful for comparing distributions between genres")

g_ecdf = sns.FacetGrid(movies, 
                       col="genre", 
                       hue="genre", 
                       col_wrap=4, 
                       sharex=False,        # Different genres might have different ranges
                       sharey=False)
g_ecdf.map(sns.ecdfplot, 'rating')
g_ecdf.fig.suptitle("Distribution of IMDB ratings by film genre (genres > 5 movies)", y=1.02)
g_ecdf.set_axis_labels("Film Rating", "Cumulative Proportion")
g_ecdf.set_titles("{col_name}")  # Fixed: was g_hist instead of g_ecdf
plt.show()

# PLOT 4: DENSITY PLOTS (KERNEL DENSITY ESTIMATION)
# Like a smooth version of a histogram - shows the probability of different values
print("\n🌊 Creating density plots...")
print("Density plots show:")
print("- A smooth curve showing where data is most/least common")
print("- Higher peaks = more common rating values")
print("- Like a 'smoothed out' histogram")

g_kde = sns.FacetGrid(movies, 
                      col="genre",
                      hue="genre",
                      col_wrap=4, 
                      sharex=True, 
                      sharey=False)
g_kde.map(sns.kdeplot, 'rating', fill=True)  # fill=True colors under the curve
g_kde.fig.suptitle("Distribution of IMDB ratings by film genre (genres > 5 movies)", y=1.02)
g_kde.set_axis_labels("Film Rating", "Density")
g_kde.set_titles("{col_name}")
plt.show()

# =============================================================================
# 6. CALCULATING CONFIDENCE INTERVALS (Getting scientific about our estimates)
# =============================================================================

print("\n" + "="*50)
print("🔬 STATISTICAL ANALYSIS - CONFIDENCE INTERVALS")
print("="*50)

print("What's a confidence interval?")
print("- It's a range around our average that probably contains the 'true' average")
print("- 95% confidence means: if we repeated this study 100 times,")
print("  95 of those times the true average would be in our range")
print("- Wider intervals = less certain, Narrower intervals = more certain")

# This is a complex chain of operations - let's break it down step by step:
genre_formula_ci = (
    movies                                    # Start with our filtered movies dataset
    .groupby('genre')['rating']              # Group by genre, focus on rating column
    .agg(['mean', 'std', 'count']) # Calculate statistics for each genre
    .rename(columns={                        # Rename columns to be clearer
        'mean': 'mean_rating', 
        'std': 'sd_rating'
    })
    .assign(                                 # Add new calculated columns
        # t_critical: the number we multiply by to get 95% confidence
        # It comes from the t-distribution (used when we don't know the population standard deviation)
        t_critical=lambda x: stats.t.ppf(0.975, x['count'] - 1),
        
        # Standard error: how much our sample mean might vary from the true mean
        # Smaller sample = higher error, More variable data = higher error
        se_rating=lambda x: x['sd_rating'] / np.sqrt(x['count']),
        
        # Margin of error: how far off our estimate might be
        margin_of_error=lambda x: x['t_critical'] * x['se_rating'],
        
        # The confidence interval bounds
        rating_low=lambda x: x['mean_rating'] - x['margin_of_error'],   # Lower bound
        rating_high=lambda x: x['mean_rating'] + x['margin_of_error']   # Upper bound
    )
    .sort_values(by='mean_rating', ascending=False)  # Sort by average rating (highest first)
    .round(2)                                        # Round to 2 decimal places
    .reset_index()                                   # Make genre a regular column again
)

print("\n📊 95% Confidence Intervals for Average Ratings by Genre:")
print(genre_formula_ci)

# =============================================================================
# 7. ADVANCED VISUALIZATIONS (Bringing statistics to life)
# =============================================================================

print("\n" + "="*50)
print("🎨 ADVANCED VISUALIZATIONS")
print("="*50)

# PLOT 5: JITTER PLOT WITH CONFIDENCE INTERVALS
# Shows individual movies AND the statistical summary
print("\n🎯 Creating jitter plot with confidence intervals...")
print("This plot shows:")
print("- Each dot is an individual movie")
print("- Orange diamonds show the average rating for each genre")
print("- Error bars show the 95% confidence interval")
print("- 'Jitter' spreads dots out so you can see them all")

plt.figure(figsize=(12, 10))  # Make a large plot

# Get genres ordered by their average rating (best to worst)
ordered_genres = genre_formula_ci.sort_values('mean_rating', ascending=False)['genre']

# Plot individual movies as dots
sns.stripplot(
    data=movies,
    x='rating',         # Rating on x-axis
    y='genre',          # Genre on y-axis
    hue='genre',        # Color by genre
    order=ordered_genres,  # Use our sorted order
    jitter=0.35,        # Spread dots out vertically so we can see them all
    alpha=0.7,          # Make dots semi-transparent
    size=3              # Small dots so plot isn't cluttered
)

# Overlay the mean and confidence interval
sns.pointplot(
    data=movies,
    x='rating',
    y='genre',
    order=ordered_genres,
    estimator=np.mean,     # Calculate the mean
    errorbar=('ci', 95),   # Add 95% confidence interval error bars
    join=False,            # Don't connect points with lines
    color='darkorange',    # Orange color to stand out
    markers='d',           # Diamond-shaped markers
    scale=1.1,             # Make markers a bit bigger
    capsize=0.2            # Size of the caps on error bars
)

plt.title("IMDB Ratings by Genre: Individual Films and Mean Confidence Intervals", fontsize=16)
plt.xlabel("IMDB Rating", fontsize=12)
plt.ylabel("Genre", fontsize=12)
plt.xticks(np.arange(0, 11, 1))  # Show ratings from 0 to 10
plt.tight_layout()  # Automatically adjust spacing
plt.show()

# PLOT 6: CLEAN ERROR BAR PLOT
# A cleaner version focusing just on the confidence intervals
print("\n📊 Creating clean error bar plot...")
print("This plot focuses on:")
print("- The average rating for each genre (black dots)")
print("- The confidence interval (gray error bars)")
print("- Exact numbers labeled for easy reading")

# Prepare data in the same order as our sorted results
ordered_genres_desc = genre_formula_ci['genre']
ci_data_for_plot = genre_formula_ci.set_index('genre').loc[ordered_genres_desc].reset_index()

plt.figure(figsize=(12, 10))

# Create error bar plot
plt.errorbar(
    y=ci_data_for_plot['genre'],              # Genres on y-axis
    x=ci_data_for_plot['mean_rating'],        # Mean ratings on x-axis
    xerr=ci_data_for_plot['margin_of_error'], # Error bars showing confidence interval
    fmt='o',                                  # Use circles for the points
    color='black',                            # Black points
    ecolor='gray',                            # Gray error bars
    elinewidth=3,                             # Thick error bars
    capsize=5,                                # Size of caps on error bars
    label='Mean Rating with 95% CI'
)

# Add text labels with the exact numbers
for index, row in ci_data_for_plot.iterrows():
    # Lower bound (left, in blue)
    plt.text(row['rating_low'] - 0.05, index, f"{row['rating_low']:.2f}", 
             color='blue', ha='right', va='center', fontsize=9)
    
    # Mean (center, in black and bold)
    plt.text(row['mean_rating'], index + 0.15, f"{row['mean_rating']:.2f}", 
             color='black', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Upper bound (right, in red)
    plt.text(row['rating_high'] + 0.05, index, f"{row['rating_high']:.2f}", 
             color='red', ha='left', va='center', fontsize=9)

plt.title("95% Confidence Intervals for Mean IMDB Ratings by Genre", fontsize=16)
plt.xlabel("Mean IMDB Rating", fontsize=12)
plt.ylabel("Genre", fontsize=12)
plt.gca().invert_yaxis()  # Put highest rated genres at the top
plt.grid(axis='x', linestyle='--', alpha=0.7)  # Add vertical grid lines
plt.tight_layout()
plt.show()

# =============================================================================
# 8. INTERPRETATION GUIDE
# =============================================================================

print("\n" + "="*50)
print("🤔 HOW TO INTERPRET YOUR RESULTS")
print("="*50)

print("Key Questions to Ask:")
print("\n1. WHICH GENRES RATE HIGHEST/LOWEST?")
print("   Look at the mean ratings in your confidence interval table.")

print("\n2. HOW CERTAIN ARE WE ABOUT THESE DIFFERENCES?")
print("   - Wider confidence intervals = less certain")
print("   - If two genres' confidence intervals overlap a lot, they might not be truly different")

print("\n3. WHAT DOES THE SHAPE OF EACH DISTRIBUTION TELL US?")
print("   - Normal (bell-shaped): Most movies are average, few very good/bad")
print("   - Skewed left: Most movies are good, few are bad")
print("   - Skewed right: Most movies are bad, few are good")
print("   - Flat: Ratings are spread evenly")

print("\n4. ARE THERE OUTLIERS?")
print("   Look for dots outside the whiskers in boxplots - these are unusual movies")

print("\n5. SAMPLE SIZE MATTERS!")
print("   Genres with more movies give us more reliable estimates")
print("   Check the 'count' column in your results")

print("\n🎉 You've completed a full statistical analysis!")
print("You've learned data loading, cleaning, visualization, and statistical inference!")