# Representing Dates and Times in Data Science
# Time stamps, which represent dates and times.
# ISO 8601 is your friend https://en.wikipedia.org/wiki/ISO_8601

import pandas as pd  
from datetime import datetime, date  
import numpy as np

not_really_a_date = 'September 1, 2025'
type(not_really_a_date)

not_really_a_time = '8:15:00'
type(not_really_a_time)

pd.Timestamp('8:15:00')
pd.Timestamp(not_really_a_time)

date_of_birth = pd.Timestamp('May 8, 1968')
date_of_birth2 = pd.Timestamp('8 May, 1968')

# Alternative way to create a Timestamp using `pd.to_datetime()`
date_of_birth3 = pd.to_datetime('8/5/1968', dayfirst=True)

# `pd.to_datetime()` accepts flexible date parsing arguments like `dayfirst`, `yearfirst`, and custom `format`, 
# while `pd.Timestamp` only accepts basic string inputs without parsing flags.

date_of_birth
date_of_birth2
date_of_birth3

type(date_of_birth)

# Create a date object
date_obj = pd.to_datetime("2025-12-25")
type(date_obj)

# If the string specifies a date but no time, Pandas fills in midnight as the default time.
# If you assign the `Timestamp` to a variable, you can use the variable name to get the year, month, and day, like this:
date_of_birth.year, date_of_birth.month, date_of_birth.day

date_of_birth.day_name()
date_of_birth.month_name()

today = date.today()  # reads system's current date
today

# now() returns current date and time
now = pd.Timestamp.now()
now

age = now - date_of_birth
age


# get components
age.days // 365.24  # integer division to get age in years

np.floor(age.days / 365.24) # rounding down, with np.floor()
np.ceil(age.days / 365.24)  # rounding up


# has my birthday passed this year?
# get components (year, month, day) for this year's birthday
bday_this_year = pd.Timestamp(now.year, 
                              date_of_birth.month, 
                              date_of_birth.day)
bday_this_year

now > bday_this_year

# Exercise 1: Compute the age difference in days between two people with different birthdays.

# Exercise 2: Compute how many days from today to Christmas.

# Exercise 3: Any two people with different birthdays have a "Double Day" when one is twice as old as the other.
# Suppose you are given two `Timestamp` values, `d1` and `d2`, that represent birthdays for two people.  
# Use `Timestamp` arithmetic to compute their double day.
# With the following dates, the result should be 2031-09-23.

# 1. Define the two birthday Timestamps
d1 = pd.Timestamp('1968-05-08')
d2 = pd.Timestamp('2000-01-15')
