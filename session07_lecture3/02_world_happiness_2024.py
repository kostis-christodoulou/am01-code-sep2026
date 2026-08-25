import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import requests
import warnings
import janitor
warnings.filterwarnings('ignore')

plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'sans-serif'


# Use pandas to read it as dataframe
world_happiness = pd.read_csv('data/world_happiness.csv')

# Clean column names using pyjanitor
world_happiness = world_happiness.clean_names()

# Inspect dataframe
print("Dataframe info:")
print(world_happiness.info())
print("\nFirst few rows:")
print(world_happiness.head())

# Get correlation matrix- drop rank and country (a string)
corr_matrix = world_happiness.drop(['rank', 'country'], axis=1).corr()  # Select only numeric columns 
print(corr_matrix)

# Produce scatterplot-correlation matrix for 2024

# Create correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, 
            annot=True, 
            cmap='coolwarm', 
            center=0,
            square=True, 
            linewidths=0.5)
plt.title('Correlation Matrix of World Happiness 2024', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()



# Summary statistics for happiness score
print("\nSummary statistics for happiness:")
print(world_happiness['happiness'].describe())

# Fit models
model1 = smf.ols('happiness ~ 1', data=world_happiness).fit()
print("\nModel 1 (Intercept only):")
print(model1.summary())

model2 = smf.ols('happiness ~ freedom_life_choices', data=world_happiness).fit()
print("\nModel 2 (Freedom to make life choices):")
print(model2.summary())

model3 = smf.ols('happiness ~ gdp_percapita + freedom_life_choices', data=world_happiness).fit()
print("\nModel 3 (GDP + Freedom):")
print(model3.summary())

model4 = smf.ols('happiness ~ gdp_percapita + social_support + freedom_life_choices', data=world_happiness).fit()
print("\nModel 4 (GDP + Social Support + Freedom):")
print(model4.summary())

model5 = smf.ols('happiness ~ gdp_percapita + healthy_life_expectancy + social_support + freedom_life_choices', data=world_happiness).fit()
print("\nModel 5 (GDP + Life Expectancy + Social Support + Freedom):")
print(model5.summary())

# =============================================================================
# Comparison of different models
# =============================================================================

from statsmodels.iolib.summary2 import summary_col

summary = summary_col([model1, model2, model3, model4, model5], 
                      stars=True, 
                      float_format='%0.3f',
                      model_names=['Model 1', 'Model 2', 'Model 3', 'Model 4', 'Model 5'], 
                      info_dict={'Residual SE': lambda x: f"{x.scale ** 0.5:.3f}"})
print(summary) 