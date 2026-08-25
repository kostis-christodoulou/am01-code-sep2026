# Loading libraries

import pandas as pd  # pandas for manipulating datasets
import matplotlib.pyplot as plt  # for plotting
import seaborn as sns  # statistically informative plots
import numpy as np  # mathematical plots
from scipy import stats  # stats formulas
import warnings  # handling warnings

# masking warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('default')
sns.set_palette("husl")

# Set font for plots
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'sans-serif'

# Load the credit data

credit = pd.read_csv('data/credit.csv')
# Rename 'own' column to 'own_house' 
credit = credit.rename(columns={'own': 'own_house'})



# Display summary statistics
print("Data Summary:")
print(credit.info())
print("\nFirst few rows:")
print(credit.head())

# Summary statistics of balance
print("\nSummary statistics of balance:")
print(credit['balance'].describe())

# Summary statistics of balance vs. married and balance vs. student
print("\nBalance vs. married:")
print(credit.groupby('married')['balance'].describe().round(2))

print("\nBalance vs. student:")
print(credit.groupby('student')['balance'].describe().round(2))

print("\nBalance vs. own_house:")
print(credit.groupby('own_house')['balance'].describe().round(2))

# ---------------------------
# Balance vs. married

print("\nBalance vs. married (detailed):")
print(credit.groupby('married')['balance'].describe())

# Box plot
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
sns.boxplot(data=credit, x='balance', y='married')
plt.title('Balance vs. Married')

# Density plot
plt.subplot(1, 2, 2)
for married_status in credit['married'].unique():
    subset = credit[credit['married'] == married_status]
    plt.hist(subset['balance'], alpha=0.3, density=True, label=married_status, bins=20)
plt.xlabel('Balance')
plt.ylabel('Density')
plt.title('Balance Distribution by Married Status')
plt.legend()
plt.tight_layout()
plt.show()

# Pair plot (equivalent to ggpairs)
plt.figure(figsize=(8, 6))
sns.pairplot(credit[['married', 'balance']], hue='married', diag_kind='hist')
plt.suptitle('Balance vs. Married - Pair Plot', y=1.02)
plt.show()

# T-test
married_yes = credit[credit['married'] == 'Yes']['balance']
married_no = credit[credit['married'] == 'No']['balance']
t_stat, p_value = stats.ttest_ind(married_yes, 
                                  married_no,
                                  equal_var=False)
print(f"\nT-test for balance vs. married:")
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.4f}")

# ---------------------------
# Balance vs. student

# Box plot
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
sns.boxplot(data=credit, x='balance', y='student')
plt.title('Balance vs. Student')

# Density plot
plt.subplot(1, 2, 2)
for student_status in credit['student'].unique():
    subset = credit[credit['student'] == student_status]
    plt.hist(subset['balance'], alpha=0.3, density=True, label=student_status, bins=20)
plt.xlabel('Balance')
plt.ylabel('Density')
plt.title('Balance Distribution by Student Status')
plt.legend()
plt.tight_layout()
plt.show()

# Pair plot
plt.figure(figsize=(8, 6))
sns.pairplot(credit[['student', 'balance']], hue='student', diag_kind='hist')
plt.suptitle('Balance vs. Student - Pair Plot', y=1.02)
plt.show()

# T-test
student_yes = credit[credit['student'] == 'Yes']['balance']
student_no = credit[credit['student'] == 'No']['balance']
t_stat, p_value = stats.ttest_ind(student_yes, 
                                  student_no, 
                                  equal_var=False)
print(f"\nT-test for balance vs. student:")
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.4f}")

# Additional analysis: Balance vs. own_house
print("\nBalance vs. own_house:")
print(credit.groupby('own_house')['balance'].describe())

# Box plot for own_house
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
sns.boxplot(data=credit, x='balance', y='own_house')
plt.title('Balance vs. Own House')

# Density plot for own_house
plt.subplot(1, 2, 2)
for house_status in credit['own_house'].unique():
    subset = credit[credit['own_house'] == house_status]
    plt.hist(subset['balance'], alpha=0.3, density=True, label=house_status, bins=20)
plt.xlabel('Balance')
plt.ylabel('Density')
plt.title('Balance Distribution by House Ownership')
plt.legend()
plt.tight_layout()
plt.show()

# T-test for own_house
house_yes = credit[credit['own_house'] == 'Yes']['balance']
house_no = credit[credit['own_house'] == 'No']['balance']
t_stat, p_value = stats.ttest_ind(house_yes, 
                                  house_no, 
                                  equal_var=False)
print(f"\nT-test for balance vs. own_house:")
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.4f}") 