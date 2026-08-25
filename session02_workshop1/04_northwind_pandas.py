# Pandas Basic EDA (Exploratory Data Analysis) - Beginner's Guide

# The Northwind database is a classic sample database originally created by Microsoft 
# for learning and demonstrating relational database concepts. 
# It models a fictional company "Northwind Traders" that imports and exports specialty foods, 
# containing tables such as Customers, Orders, Products, Employees, Suppliers, and more.

# =============================================================================
# IMPORTING LIBRARIES
# =============================================================================

# Import pandas - the main library for data manipulation and analysis
import pandas as pd

# Import matplotlib for basic plotting and visualization
import matplotlib.pyplot as plt

# Import seaborn - makes matplotlib plots prettier and easier to create
import seaborn as sns

# Import skimpy for quick summary statistics (like a fancy version of df.describe())
from skimpy import skim

# =============================================================================
# LOADING DATA FROM EXCEL FILES
# =============================================================================

# The `read_excel` function of the `pandas` package lets you choose the sheet name 
# you want to read from the Excel workbook. 

# NAMING CONVENTION TIP: When creating a new dataframe, it's good practice to start 
# its variable name with "df_" so that you always know the variable contains a DataFrame, 
# as opposed to just a string or number.

# Load the Orders sheet into a DataFrame
df_orders = pd.read_excel("data/northwind_database.xlsx", sheet_name="Orders")

# Load the Customers sheet into a DataFrame  
df_customers = pd.read_excel("data/northwind_database.xlsx", sheet_name="Customers")

# =============================================================================
# BASIC DATA EXPLORATION - COUNTING ROWS AND COLUMNS
# =============================================================================

# .shape returns a tuple (rows, columns) - tells us the dimensions of our data
df_orders.shape      # Returns something like (830, 14) meaning 830 rows, 14 columns
df_customers.shape   # Returns something like (91, 11) meaning 91 rows, 11 columns

# Extract just the number of rows (index 0 of the shape tuple)
print(f"Number of orders: {df_orders.shape[0]}")       # Gets the first element (rows)
print(f"Number of customers: {df_customers.shape[0]}")  # Gets the first element (rows)

# IMPORTANT DISTINCTION: .shape[0] gives total rows, but for customers we might want 
# to count UNIQUE customers in case there are duplicates
print("Customer columns:", df_customers.columns)  # Show all column names

# .nunique() counts the number of unique values in a column (removes duplicates)
print(f"Number of unique customers: {df_customers['CustomerID'].nunique()}")

# =============================================================================
# USING SKIM FOR QUICK DATA OVERVIEW
# =============================================================================

# skim() gives a comprehensive overview of your data:
# - Data types of each column
# - Missing values count  
# - Summary statistics (mean, median, etc.)
# - Unique values count
skim(df_orders)

# =============================================================================
# ACCESSING SPECIFIC DATA POINTS WITH .loc[]
# =============================================================================

# .loc[] is used to access data by row and column labels/names
# Syntax: df.loc[row_index, "column_name"]

# Find the contact name of the customer at row index 25
print(df_customers.loc[25, "ContactName"])  # Should print "Carine Schmitt"

# Load the Order Details sheet to access more granular order information
df_order_details = pd.read_excel("data/northwind_database.xlsx", sheet_name="Order Details")

# Find the ProductID at row index 1425 in the Order Details
print(df_order_details.loc[1425, "ProductID"])  # Should print 54

# Working with dates: .dt.day_name() converts a date to the day of the week
# First get the shipped date, then convert it to day name
shipped_date = df_orders.loc[29, "ShippedDate"]
print(shipped_date.day_name())  # Should print "Tuesday"

# =============================================================================
# RENAMING COLUMNS
# =============================================================================

# Method 1: Create a completely independent copy with renamed columns
# .copy() ensures changes to this new DataFrame won't affect the original
# Use this when you want to be absolutely sure DataFrames are separate
df_orders_renamed = df_orders.rename(columns={
    'Freight': 'FreightCost',    # Old name: New name
    'ShipName': 'VesselName'     # Old name: New name
}).copy()

# Method 2: Create a new DataFrame with renamed columns (without full copy)
# This is faster and uses less memory, but the new DataFrame might share 
# some data with the original behind the scenes
df_orders_renamed_again = df_orders.rename(columns={
    "Freight": "FreightCost", 
    "ShipName": "VesselName"
})

# =============================================================================
# CREATING NEW COLUMNS AND CALCULATIONS
# =============================================================================

# Create a new column by multiplying two existing columns
# This calculates the total cost for each line item (price × quantity)
df_order_details['item_cost'] = df_order_details['UnitPrice'] * df_order_details['Quantity']

# Use skim to see our new column and its statistics
skim(df_order_details)

# =============================================================================
# SORTING DATA
# =============================================================================

# Sort by Freight column in descending order (highest to lowest)
# [['OrderID', 'Freight']] selects only these two columns to display
df_orders.sort_values("Freight", ascending=False)[['OrderID', 'Freight']]

# Default sorting is ascending (lowest to highest)
df_orders.sort_values("Freight")[['OrderID', 'Freight']]

# reset_index(drop=True) renumbers the rows from 0, 1, 2, ... after sorting
# Without drop=True, the old index would become a new column
# With drop=True, the old index is discarded
df_orders.sort_values("Freight").reset_index(drop=True)[['OrderID', 'Freight']]

# =============================================================================
# BASIC PLOTTING WITH MATPLOTLIB
# =============================================================================

# Create a scatter plot: each point represents one order detail
# x-axis: Quantity, y-axis: item_cost
df_order_details.plot(x='Quantity', y='item_cost', kind='scatter')
plt.show()  # Display the plot

# Create a histogram showing the distribution of item costs
df_order_details.plot(y='item_cost', kind='hist')
plt.show()  # Display the plot

# =============================================================================
# ADVANCED PLOTTING WITH SEABORN
# =============================================================================

# Create a more customized histogram using seaborn
sns.histplot(data=df_order_details,     # The DataFrame to use
             x='item_cost',             # Column for x-axis
             bins=20,                   # Number of bins (bars) in histogram
             binrange=(0, 4000))        # Limit x-axis from 0 to 4000

# Add labels and title for better readability
plt.title("Histogram of Item Cost")
plt.xlabel("Item Cost")
plt.ylabel("Frequency")  # How many items fall in each cost range
plt.show()

# =============================================================================
# COUNTING VALUES AND FREQUENCY ANALYSIS
# =============================================================================

# .value_counts() counts how many times each value appears
# Shows which employee handled the most orders
employee_order_counts = df_orders["EmployeeID"].value_counts()
print(employee_order_counts)

# Plot the counts as a bar chart
# Each bar shows how many orders each employee processed
df_orders["EmployeeID"].value_counts().plot(kind="bar")
plt.show()

# normalize=True converts counts to proportions (percentages)
# Shows what percentage of total orders each employee handled
df_orders["EmployeeID"].value_counts(normalize=True).plot(kind="bar")
plt.show()

# =============================================================================
# ANALYZING CONTINUOUS VARIABLES (FREIGHT COSTS)
# =============================================================================

# For continuous variables like Freight, value_counts() isn't very useful
# because each value might be unique
freight_counts = df_orders["Freight"].value_counts()
print(freight_counts)

# This bar chart will be messy because freight values are mostly unique
df_orders["Freight"].value_counts().plot(kind="bar")
plt.show()

# .describe() gives summary statistics for numerical columns:
# count, mean, std (standard deviation), min, 25%, 50% (median), 75%, max
freight_summary = df_orders["Freight"].describe()
print(freight_summary)

# Box plot shows the distribution of freight costs
# - The box shows the middle 50% of data (25th to 75th percentile)
# - The line in the middle is the median
# - The whiskers show the range
# - Dots beyond whiskers are potential outliers
df_orders["Freight"].plot(kind="box")
plt.show()

# Histogram shows the shape of the distribution
# Most freight costs are low, with a few high-cost shipments
df_orders["Freight"].plot(kind="hist")
plt.show()



# =============================================================================
# SELECTING COLUMNS AND FILTERING DATA
# =============================================================================

# Select multiple columns - even duplicates (repeats the same column)
# This demonstrates how to select specific columns you're interested in
df_orders[  ["OrderID","OrderID","OrderID"]   ]

# Filtering data with .isin() - selects rows where a column matches values in a list
# This finds all orders shipped to either France or Belgium
df_orders[df_orders["ShipCountry"].isin(["France", "Belgium"])]

# Combining filtering AND column selection
# First filter for France/Belgium orders, then select only the ShipCountry column
df_orders[df_orders["ShipCountry"].isin(["France", "Belgium"])]["ShipCountry"]

# Count occurrences of each value in the filtered data
# Shows how many orders were shipped to each country (France vs Belgium)
df_orders[df_orders["ShipCountry"].isin(["France", "Belgium"])]["ShipCountry"].value_counts()

# Convert counts to percentages with normalize=True
# Shows what percentage of the filtered orders went to each country
df_orders[df_orders["ShipCountry"].isin(["France", "Belgium"])]["ShipCountry"].value_counts(normalize=True)

# Filter for orders within a specific date range
(df_orders["OrderDate"] > "1996-08-03")   &   (df_orders["OrderDate"] < "1997-01-30")
df_orders[df_orders["OrderDate"].between("1996-08-03", "1997-01-30")]

# =============================================================================
# KEY CONCEPTS RECAP FOR BEGINNERS:
# =============================================================================

# 1. DataFrames (.shape, .loc[], .columns)
# 2. Loading data (pd.read_excel())
# 3. Creating new columns (df['new_col'] = calculation)
# 4. Sorting data (.sort_values())
# 5. Counting values (.value_counts(), .nunique())
# 6. Summary statistics (.describe(), skim())
# 7. Basic plotting (.plot(), plt.show())
# 8. Difference between categorical (EmployeeID) and continuous (Freight) variables
