# Import the necessary Python libraries.

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import janitor
import warnings
warnings.filterwarnings('ignore')

# statsmodels: The go-to library in Python for rigorous statistical modeling.
# We import the formula API, which lets us use R-style formulas. 
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor

# --- Load and Prepare the Data ---


uber_surge = pd.read_csv('data/uber_surge.csv').clean_names()  # janitor's clean_names makes column names snake_case
print(uber_surge.head())
uber_surge.info()

print("\n--- Summary Statistics ---")
# .T transposes the output to make it easier to read.
print(uber_surge.describe().T)



# The Python equivalent of GGally::ggpairs is seaborn.pairplot. 
# It creates a grid of scatterplots for each pair of variables and histograms/density plots on the diagonal.
print("\n--- Generating Pairplot (Numerical Columns) ---")
sns.pairplot(uber_surge,
             hue='region', 
             plot_kws={'alpha': 0.5})
plt.suptitle('Pairplot of Pick Up Time vs. Number of Drivers', y=1.02)
plt.show() 



# For a single column, we can just select it and use .describe().
print("\n--- Detailed stats for 'pick_up_time' ---")
print(uber_surge['pick_up_time'].describe())



# ---------------------- 0. Intercept-only model ----------------------
# An intercept-only model serves as a baseline for comparison with more complex models.
# It helps us understand how much better our predictors perform compared to just using the mean.

# This is an "intercept-only" model. It predicts the pick-up time using just the overall mean.
# We use smf.ols (Ordinary Least Squares) to define the model with an R-style formula.
print("\n--- Model 0: Intercept-Only Model ---")
model0 = smf.ols('pick_up_time ~ 1', data=uber_surge).fit()
# The .summary() method provides a comprehensive output similar to R's summary() or msummary(). [34]
print(model0.summary())


# ---------------------- 1. Relationship time vs number of drivers ? ----------------------

# R equivalent:
# ggplot(uber_surge, aes(x = number_of_drivers, y = pick_up_time_min)) +
#   geom_point() + geom_smooth(method = "lm", se = FALSE)

# In Seaborn, regplot() creates a scatter plot and fits a linear regression model. [1, 3]
print("\n--- Plotting Pick Up Time vs. Number of Drivers ---")
plt.figure(figsize=(8, 6))
sns.regplot(x='number_of_drivers', 
            y='pick_up_time', 
            data=uber_surge, 
            ci=None, 
            line_kws={'color': 'blue'})
plt.title('Pick Up Time vs. Number of Drivers')
plt.grid(True)
plt.show() # Uncomment to display plot


# Regression of pick up time vs. number of drivers
print("\n--- Model 1: Pick Up Time ~ Number of Drivers ---")
model1 = smf.ols('pick_up_time ~ number_of_drivers', data=uber_surge).fit()
print(model1.summary())
print(f"Residual SE: {model1.scale ** 0.5:.3f}")  # Residual standard error is the square root of the scale (MSE of residuals)




# ---------------------- 2. Use model 1 to predict pick up time  ----------------------


# To get fitted values, residuals, and prediction intervals, we use the .get_prediction() method. [2, 8]
# We then call .summary_frame() to get these values in a new DataFrame.
print("\n--- Augmenting Model 1 with Predictions ---")
predictions_df = model1.get_prediction(uber_surge).summary_frame(alpha=0.05)

# We can join this back to the original data to have everything in one place.
uber_surge_augmented = uber_surge.join(predictions_df)
print(uber_surge_augmented.head())


# ---------------------- 3. pick up time on drivers + surge price ? ----------------------


# Regression model with two predictors
print("\n--- Model 2: Pick Up Time ~ Number of Drivers + Surge Price ---")
model2 = smf.ols('pick_up_time ~ number_of_drivers + surge_price', data=uber_surge).fit()
print(model2.summary())
print(f"Residual SE: {model2.scale ** 0.5:.3f}")  # Residual standard error is the square root of the scale (MSE of residuals)


# Calculate Variance Inflation Factor (VIF) to check for multicollinearity. 
# A VIF > 5 or 10 is often considered a sign of high multicollinearity.

from statsmodels.stats.outliers_influence import variance_inflation_factor
# VIF for Model 3
X_m3 = uber_surge.drop(columns=['pick_up_time', 'region']).assign(Intercept=1)
vif_data = pd.DataFrame()
vif_data["feature"] = X_m3.columns.drop('Intercept')
vif_data["VIF"] = [variance_inflation_factor(X_m3.values, i) for i in range(X_m3.shape[1]-1)]
print("\n--- VIF for Model 3 ---")
print(vif_data)


# High colinearity... number of drivers and surge price have a correlation of -0.98
# Let's confirm this by looking at the correlation and the pairplot again.
print("\n--- Correlation Matrix ---")
print(uber_surge.drop(columns=['region']).corr().round(2))

# The pairplot from earlier also visually confirms this strong negative correlation.


# ---------------------- 4. pick up time on surge price ? ----------------------

# Because of the high VIF, we build a model with only surge_price.
print("\n--- Model 3: Pick Up Time ~ Surge Price ---")
model3 = smf.ols('pick_up_time ~ surge_price', data=uber_surge).fit()
print(model3.summary())
print(f"Residual SE: {model3.scale ** 0.5:.3f}")  # Residual standard error is the square root of the scale (MSE of residuals)



# ---------------------- 5. pick up time on drivers + region ? ----------------------

# Seaborn's lmplot is perfect for this. It automatically creates separate regression lines for each category defined by 'hue'. [11, 18, 22]
print("\n--- Plotting Pick Up Time vs. Drivers, by Region ---")
sns.lmplot(x='number_of_drivers', 
           y='pick_up_time', 
           hue='region', 
           data=uber_surge,
           scatter_kws={'alpha': 0.7}, 
           ci=None)
plt.title('Pick Up Time vs. Drivers, by Region')
plt.show() # Uncomment to display plot


# Statsmodels automatically handles the categorical 'region' variable by creating dummy variables.
print("\n--- Model 4: Pick Up Time ~ Number of Drivers + Region ---")
model4 = smf.ols('pick_up_time ~ number_of_drivers + region', data=uber_surge).fit()
print(model4.summary().tables[1])  # Print only the coefficients table for brevity
print("R-squared:", model4.rsquared.round(4))
print("Adjusted R-squared:", model4.rsquared_adj.round(4))
print(f"Residual SE: {model4.scale ** 0.5:.3f}")  # Residual standard error is the square root of the scale (MSE of residuals).


# =============================================================================
# Comparison of different models
# =============================================================================

from statsmodels.iolib.summary2 import summary_col

summary = summary_col([model1, model2, model3, model4], 
                      stars=True, 
                      float_format='%0.3f',
                      model_names=['Model 1', 'Model 2', 'Model 3', 'Model 4'], 
                      info_dict={'Residual SE': lambda x: f"{x.scale ** 0.5:.3f}"})
print(summary) 