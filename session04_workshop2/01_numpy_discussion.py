# Differences in Pret-a-manger prices between London and Boston <https://www.economist.com/finance-and-economics/2019/09/07/why-americans-pay-more-for-lunch-than-britons-do>.

# A list of strings
name_list = [
    'Lobster roll',
    'Chicken caesar',
    'Bang bang chicken',
    'Ham and cheese',
    'Tuna and cucumber',
    'Egg'
]

boston_price_list = [9.99, 7.99, 7.49, 7.00, 6.29, 4.99]

# Here are the prices in London, converted to dollars at $1.25 / £1.
london_price_list = [7.5, 5, 4.4, 5, 3.75, 2.25]

# Lists provide some arithmetic operators, but they might not do what you want.
# For example, the `+` operator works with lists:
boston_price_list + london_price_list

# NumPy arrays
import numpy as np

boston_price_array = np.array(boston_price_list)
london_price_array = np.array(london_price_list)

type(london_price_array_price_array)
# <class 'numpy.ndarray'>

# type of elements in the array- they must all be the same!
london_price_array.dtype

name_array = np.array(name_list)
name_array

# pairwise differences in prices
differences = boston_price_array - london_price_array
differences 

percent_differences = relative_differences * 100
percent_differences

np.min(percent_differences), np.max(percent_differences)


# Vectorization batch operations on data without for loops.
# Arithmetic, relational and boolean operations between equally-sized arrays are applied element-wise.

# Example: price comparison
boston_price_array > london_price_array

# Statistical summaries
differences.mean(), differences.std(ddof=1) # use ddof=1 for sample standard deviation

# relative_differences
differences = boston_price_array - london_price_array
relative_differences = differences / london_price_array
relative_differences

# Boolean arrays

x = np.array([[True, True], [False, False]])
y = np.array([[False, True], [True, False]])

# Find logical relation *and* between x and y
np.logical_and(x, y)
x & y

# Find logical relation *or* between x and y
np.logical_or(x, y)
x | y

# Axis: Column- or Row-wise computation
a = np.arange(1,10).reshape(3,3)
print(a)
# >>> array([[1, 2, 3],
# [4, 5, 6],
# [7, 8, 9]])

a_sum_cols = np.sum(a, axis=0) # axis=0 is default for column-wise operations
print(a_sum_cols)
# >>> [12 15 18]

a_sum_rows = np.sum(a, axis=1) # axis=1 is for row-wise operations
print(a_sum_rows)
# >>> [ 6 15 24]