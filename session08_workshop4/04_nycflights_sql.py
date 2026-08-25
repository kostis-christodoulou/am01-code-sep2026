# Large Datasets with DuckDB, and Pandas

# duckdb: A fast, in-process analytical database. "In-process" means it runs inside our Python script,
# requiring no separate server installation. It's great for data analysis.
import duckdb

import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

# time, os, requests, io: Standard Python libraries for timing our code, interacting with the operating system,
# making HTTP requests (like downloading files), and handling in-memory data streams.
import time
import os
import requests
from io import StringIO

# nycflights13: A Python package that provides the same datasets as the famous R package.
# It contains data about all flights that departed from NYC in 2013.
import nycflights13 as nf


# --- Introduction to Databases ---

# When data is too large to fit into a computer's RAM (memory), we can't use pandas to load it all at once.
# Instead, we store it in a **database**, which is a system designed to manage and retrieve vast amounts of data efficiently.
# Databases live on a disk (either locally or on a remote server).
# We interact with a database by sending it a **query**—a request for specific information.
# The database processes this query and returns only the data we asked for, which is usually small enough to fit in memory.
# Inside a database, data is organized into **tables**, which are just like pandas DataFrames, with rows and columns.


# --- Databases and Python ---

# The language used to communicate with most databases is **SQL** (**S**tructured **Q**uery **L**anguage).
# While powerful, SQL can feel less intuitive than Python code.
# The great news is that we can use Python libraries like DuckDB to send SQL queries to a database.
# The results can then be loaded directly into pandas DataFrames for further analysis and plotting.

# --- `SQL` commands vs `pandas` verbs ---

# Many pandas operations have a direct equivalent in SQL.
# Understanding this mapping helps you translate your data manipulation logic between the two systems.

# +------------------------+------------------------------------+
# |    `SQL` command...    | ... translate to `pandas` method   |
# +========================+====================================+
# | SELECT col1, col2      | df[['col1', 'col2']]               |
# +------------------------+------------------------------------+
# | FROM table             | The DataFrame `df` itself          |
# +------------------------+------------------------------------+
# | WHERE condition        | df.query('condition') or df[df['col'] > 5] |
# +------------------------+------------------------------------+
# | GROUP BY col1, col2    | df.groupby(['col1', 'col2'])       |
# +------------------------+------------------------------------+
# | ORDER BY col           | df.sort_values('col')              |
# +------------------------+------------------------------------+
# | LIMIT 10               | df.head(10)                        |
# +------------------------+------------------------------------+
# | JOIN other_table       | pd.merge(df, other_df)             |
# +------------------------+------------------------------------+


# --- Establish a connection with the database ---

# To work with a database, we first need to connect to it.
# For this example, we'll use DuckDB to create an **in-memory** database.
# This means the database is temporary and exists only in our computer's RAM while the script is running.
# It's perfect for learning because it requires no setup.
print("--- Connecting to in-memory DuckDB database ---")
connection = duckdb.connect(database=':memory:')
print("Connection successful:", connection)


# --- Populate our database with `flights` data ---

# Now that we have an empty database, let's add some tables to it.
# We'll use the data from the `nycflights13` package.
print("\n--- Loading nycflights13 data into pandas DataFrames ---")

# These lines load the datasets into pandas DataFrames first.
flights_df = nf.flights
weather_df = nf.weather
airlines_df = nf.airlines
airports_df = nf.airports
planes_df = nf.planes

# To make a pandas DataFrame queryable by DuckDB, we "register" it as a virtual table.
# This doesn't copy the data. It just tells DuckDB where to find it.
connection.register('flights_df_view', flights_df)

# Now, we run an SQL command to create a permanent table named 'flights' inside our DuckDB database.
# This command selects all data from our virtual table and materializes it as a real database table.
connection.execute('CREATE TABLE flights AS SELECT * FROM flights_df_view;')

# **Indexes** are special lookup tables that the database can use to speed up queries significantly.
# It's like the index at the back of a book; it helps the database find rows with specific values much faster.
# We're creating indexes on columns that we expect to filter or join on frequently.
print("Creating indexes on the flights table for faster queries...")
connection.execute('CREATE INDEX idx_flights_date ON flights (year, month, day);')
connection.execute('CREATE INDEX idx_flights_carrier ON flights (carrier);')
connection.execute('CREATE INDEX idx_flights_tailnum ON flights (tailnum);')
connection.execute('CREATE INDEX idx_flights_dest ON flights (dest);')

print("Successfully populated 'flights' table in DuckDB.")

# --- Pandas Equivalent (for indexing) ---
# In pandas, the `.set_index()` method can speed up row lookups based on the index label.
# However, it is less powerful than SQL indexes, which can span multiple columns and are
# used by a sophisticated query optimizer to speed up a wide range of operations (filters, joins, etc.).
#
# flights_df_indexed = flights_df.set_index('tailnum')
# specific_flight = flights_df_indexed.loc['N321AS'] # This is now faster


#  --- Database objects vs. DataFrames ---

# We can ask the database to show us all the tables it contains.
print("\n--- Listing tables in the database ---")
tables = connection.execute("SHOW TABLES;").fetchdf() # .fetchdf() returns the result as a pandas DataFrame
print(tables)

# --- Pandas Equivalent ---
# In a pandas-only workflow, you don't have a database of tables. You just have
# different DataFrame variables in your script's memory, like `flights_df`, `weather_df`, etc.
# You might keep track of them in a list:
# dataframes_in_memory = ['flights_df', 'weather_df', 'airlines_df', 'airports_df', 'planes_df']
# print(dataframes_in_memory)


# We can create a reference to a table. This object, `flights_db`, doesn't hold the data itself.
# It's a "pointer" or a "relation object" that we can use to build queries.
flights_db = connection.table("flights")
print(f"\nType of flights_db object: {type(flights_db)}")

# --- Pandas Equivalent ---
# The pandas equivalent is simply the DataFrame variable itself.
# flights_pd = flights_df


# --- Generating queries ---

# DuckDB has a "lazy" API. This means that when you write a query, it doesn't run immediately.
# Instead, it builds a query plan. The database only does the work when you explicitly ask for the results.
# This allows the database to optimize the entire chain of operations before executing anything.

# Find flights that had a departure delay of more than 120 minutes.
print("\n--- Query: Flights with departure delay > 120 minutes ---")
# This line builds the query but does NOT execute it yet.
delayed_flights_query = flights_db.filter('dep_delay > 120')
# Notice the output just shows the query plan, not the data.
print(delayed_flights_query)

# --- Pandas Equivalent (Eager Execution) ---
# Pandas is "eager," meaning it executes operations immediately.
# This would filter the DataFrame and store the result in a new DataFrame right away.
#
# delayed_flights_df_pandas = flights_df.query('dep_delay > 120')
# print(delayed_flights_df_pandas.head())


# Here's how to write the same query using a raw SQL string.
delayed_flights_df_sql = connection.execute("""
    SELECT * FROM flights WHERE dep_delay > 120
""").df() # The .df() at the end executes the query and returns a pandas DataFrame.


# Let's build a more complex query: calculate the mean departure delay for each origin-destination pair.
print("\n--- Query: Mean departure delay by origin and destination ---")
# This is a raw SQL query.
mean_delay_db_query = connection.execute("""
    SELECT
        origin,
        dest,
        mean(dep_delay) AS mean_dep_delay -- Calculate the mean and name the new column
    FROM flights
    GROUP BY origin, dest -- Group rows so the mean is calculated for each group
    HAVING mean_dep_delay IS NOT NULL -- Filter out groups where the result is null
    ORDER BY mean_dep_delay DESC -- Sort the results
""")

# Again, this returns a relation object, not the data.
print("Type of the query object:", type(mean_delay_db_query))

# To get the results, we call .df() on the query object.
mean_delay_df = mean_delay_db_query.df()
print("Top 5 results of the query:")
print(mean_delay_df.head())


# --- Pandas Equivalent ---
#
# mean_delay_df_pandas = (
#     flights_df
#     .groupby(['origin', 'dest'])  # Corresponds to GROUP BY
#     .agg(mean_dep_delay=('dep_delay', 'mean'))  # Corresponds to SELECT ... mean()
#     .dropna()  # Corresponds to HAVING ... IS NOT NULL
#     .sort_values('mean_dep_delay', ascending=False)  # Corresponds to ORDER BY ... DESC
#     .reset_index()  # Flatten the grouped index
# )
# print(mean_delay_df_pandas.head())


# --- Laziness as a virtue ---

# Let's chain several operations. We want to find the average delay for each plane (identified by 'tailnum'),
# but only for planes that have more than 100 flights.
print("\n--- Building a more complex lazy query for tailnum delays ---")

# This builds the entire query plan without moving any data.
tailnum_delay_db_query = flights_db.aggregate(
    [
        'mean_dep_delay := mean(dep_delay)', # Calculate mean departure delay
        'mean_arr_delay := mean(arr_delay)', # Calculate mean arrival delay
        'n := count(*)'                      # Count the number of flights
    ],
    'tailnum' # The grouping column
).filter('n > 100').order('mean_arr_delay DESC')

# The output shows the planned steps, not the data.
print(tailnum_delay_db_query)


# Here is the equivalent query written in raw SQL.
tailnum_delay_sql = """
    SELECT
      tailnum,
      mean(dep_delay) AS mean_dep_delay,
      mean(arr_delay) AS mean_arr_delay,
      count(*) AS n
    FROM flights
    GROUP BY tailnum
    HAVING n > 100
    ORDER BY mean_arr_delay DESC;
"""

# --- Pandas Equivalent ---
#
# tailnum_delay_df_pandas = (
#     flights_df
#     .groupby('tailnum')
#     .agg(
#         mean_dep_delay=('dep_delay', 'mean'),
#         mean_arr_delay=('arr_delay', 'mean'),
#         n=('flight', 'count')  # Count flights for each tailnum
#     )
#     .query('n > 100')  # This is equivalent to SQL's HAVING clause here
#     .sort_values('mean_arr_delay', ascending=False)
#     .reset_index()
# )
# print(tailnum_delay_df_pandas.head())


# --- Collect the data into your local Python environment ---

# To execute the query and bring the results into a pandas DataFrame, we use `.df()`.
# This is the moment the database does its work and we pull the data into memory.
print("\n--- Collecting tailnum delay data into a pandas DataFrame ---")
tailnum_delay_df = connection.execute(tailnum_delay_sql).df()
print(tailnum_delay_df.head())


# --- Use the local DataFrame to plot ---

# Now that our aggregated data is in a pandas DataFrame (`tailnum_delay_df`),
# we can use any of Python's plotting libraries, like Seaborn, just as we normally would.
print("\n--- Plotting arrival vs. departure delay ---")
plt.figure(figsize=(10, 8))
sns.set_theme(style="whitegrid")

plot = sns.scatterplot(
    data=tailnum_delay_df,
    x='mean_dep_delay',
    y='mean_arr_delay',
    size='n',      # Make the point size proportional to the number of flights
    alpha=0.3,     # Make points semi-transparent
    sizes=(20, 500)
)

# Add a reference line where y = x. Points below this line represent planes that
# made up time during the flight (arrival delay was less than departure delay).
plot.axline([0, 0], [1, 1], color='tomato', linestyle='--', linewidth=2, label='y=x (no change in delay)')

plt.title("Most planes manage to make up time, even if they depart late")
plt.xlabel("Mean departure delay (minutes)")
plt.ylabel("Mean arrival delay (minutes)")
plt.legend(title='Number of flights')
plt.grid(True)
# plt.show() # Uncomment to display the plot


# --- Joins ---

# Joins are used to combine tables based on a common column.
# Here, we will join the `flights` table with the `planes` table using the `tailnum` column,
# which uniquely identifies each aircraft. This allows us to combine flight details with aircraft manufacturing details.
print("\n--- Performing a LEFT JOIN with the 'planes' table ---")

# First, let's create the 'planes' table in our database from the pandas DataFrame.
connection.execute('CREATE TABLE planes AS SELECT * FROM planes_df;')
connection.execute('CREATE INDEX idx_planes_tailnum ON planes (tailnum);') # Add index for faster joins

print("Tables in the database now:")
print(connection.execute("SHOW TABLES;").fetchdf())

# This SQL query performs a LEFT JOIN. It takes all rows from `flights` (the left table)
# and finds matching rows in `planes` (the right table) where the `tailnum` is the same.
join_sql = """
    SELECT
        f.year, f.month, f.day, f.dep_time, f.arr_time, f.carrier, f.flight, f.tailnum,
        p.year AS year_built, p.type, p.model
    FROM flights AS f
    LEFT JOIN planes AS p ON f.tailnum = p.tailnum
    LIMIT 5
"""

print("\nResult of the join (first 5 rows):")
joined_db_df = connection.execute(join_sql).df()
print(joined_db_df)

# --- Pandas Equivalent ---
#
# # The `pd.merge` function is the pandas equivalent of a SQL JOIN.
# joined_df_pandas = pd.merge(
#     left=flights_df,
#     right=planes_df,
#     on='tailnum',        # The common column to join on
#     how='left'           # 'left' corresponds to LEFT JOIN
# )
#
# # Select and rename columns to match the SQL output
# joined_df_pandas = joined_df_pandas[[
#     'year_x', 'month', 'day', 'dep_time', 'arr_time', 'carrier', 'flight', 'tailnum',
#     'year_y', 'type', 'model'
# ]].rename(columns={'year_x': 'year', 'year_y': 'year_built'})
#
# print("\nPandas equivalent of the join (first 5 rows):")
# print(joined_df_pandas.head(5))


# --- Relational API vs. Raw SQL: Which to Choose? ---
# The previous sections are left as is, as they provide a great explanation.

# --- Close the database connection ---
# It's good practice to close the connection when you're done with it to free up resources.
connection.close()
print("\n--- Database connection closed ---")