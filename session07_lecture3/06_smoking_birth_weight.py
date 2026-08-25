import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set the aesthetic style of the plots
sns.set_palette("husl")
# Set the style and create the plot canvas.
sns.set_style("ticks")
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'sans-serif'

# Import the data set
smoking_birth_weight = pd.read_csv("data/smoking_birth_weight.csv")

# Summary statistics
print("Data summary:")
print(smoking_birth_weight.info())
print("\nMissing values:")
print(smoking_birth_weight.isnull().sum())
print("\nDescriptive statistics:")
print(smoking_birth_weight.describe())


# Count smokers vs non-smokers
habit_counts = smoking_birth_weight['habit'].value_counts()
habit_proportions = smoking_birth_weight['habit'].value_counts(normalize=True)
print("\nSmoking habit counts:")
print(habit_counts)
print("\nProportions:")
print(habit_proportions)

# Descriptive statistics on weight by smoking habit
print("\nWeight statistics by smoking habit:")
print(smoking_birth_weight.groupby('habit')['weight_kg'].describe().round(3))


# T-test with unequal variance (Welch's t-test)

smoker_weight = smoking_birth_weight[smoking_birth_weight['habit'] == 'smoker']['weight_kg']
nonsmoker_weight = smoking_birth_weight[smoking_birth_weight['habit'] == 'nonsmoker']['weight_kg']
t_stat, p_value = stats.ttest_ind(smoker_weight, nonsmoker_weight, equal_var=False)
print(f"\nT-test results: t-stat={t_stat:.4f}, p-value={p_value:.4f}")



# Model building
print("\nOverall weight statistics:")
print(smoking_birth_weight['weight_kg'].describe())

# Model 1: Intercept only
model1 = smf.ols('weight_kg ~ 1', data=smoking_birth_weight).fit()
print("\nModel 1 (Intercept only):")
print(model1.summary2().tables[1])
print("R-squared:", model1.rsquared.round(4))
print("Adjusted R-squared:", model1.rsquared_adj.round(4))
print(f"Residual SE: {model1.scale ** 0.5:.3f}")  # Residual standard error is the square root of the scale (MSE of residuals).

# Model 2: Habit (smoking vs non-smoking)
model2 = smf.ols('weight_kg ~ habit', data=smoking_birth_weight).fit()
print("\nModel 2 (Habit):")
print(model2.summary2().tables[1])
print("R-squared:", model2.rsquared.round(4))
print("Adjusted R-squared:", model2.rsquared_adj.round(4))
print(f"Residual SE: {model2.scale ** 0.5:.3f}")  # Residual standard error is the square root of the scale (MSE of residuals).


# Model 3: Weeks + Habit
model3 = smf.ols('weight_kg ~ weeks + habit', data=smoking_birth_weight).fit()
print("\nModel 3 (Weeks + Habit):")
print(model3.summary2().tables[1])
print("R-squared:", model3.rsquared.round(4))
print("Adjusted R-squared:", model3.rsquared_adj.round(4))
print(f"Residual SE: {model3.scale ** 0.5:.3f}")  # Residual standard error is the square root of the scale (MSE of residuals).

# Model 4: Weeks + Habit + gender + whitemom + gained_kg
model4 = smf.ols('weight_kg ~ weeks + habit + gender + whitemom + gained_kg', data=smoking_birth_weight).fit()
print("\nModel 4 (Weeks + Habit + Gender + WhiteMom + Gained KG):")
print(model4.summary2().tables[1])
print("R-squared:", model4.rsquared.round(4))
print("Adjusted R-squared:", model4.rsquared_adj.round(4))
print(f"Residual SE: {model4.scale ** 0.5:.3f}")  # Residual standard error is the square root of the scale (MSE of residuals).

# Model 5: All variables

model5 = smf.ols('weight_kg ~ weeks + habit + whitemom + gender + gained_kg + father_age + mother_age + mature + preterm + visits + lowbirthweight', data=smoking_birth_weight).fit()
print("\nModel 5 (All variables):")
print(model5.summary2().tables[1])
print("R-squared:", model5.rsquared.round(4))
print("Adjusted R-squared:", model5.rsquared_adj.round(4))
print(f"Residual SE: {model5.scale ** 0.5:.3f}")  # Residual standard error is the square root of the scale (MSE of residuals).


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

def check_model_diagnostics(model):

    """
    Creates a 2x2 grid of regression diagnostic plots for a fitted statsmodels model.
    This version is robust and works across different statsmodels versions.
    """

    # ----- 1. GATHER DATA FROM THE FITTED MODEL -----
    fitted_values = model.fittedvalues
    residuals = model.resid
    studentized_residuals = model.get_influence().resid_studentized_internal
    
    # ----- 2. SETUP THE PLOTTING AREA -----
    fig, axes = plt.subplots(2, 2, figsize=(8, 5))
    fig.suptitle('Regression Diagnostic Plots', fontsize=18, y=1.02)
    sns.set_style('whitegrid')

    # ----- PLOT 1: Residuals vs. Fitted -----
    sns.residplot(x=fitted_values, y=residuals, lowess=True, 
                  scatter_kws={'alpha': 0.8}, 
                  line_kws={'color': 'blue', 'lw': 2}, ax=axes[0, 0])
    axes[0, 0].set_title('Residuals vs. Fitted Values', fontsize=14)
    axes[0, 0].set_xlabel('Fitted Values', fontsize=12)
    axes[0, 0].set_ylabel('Residuals', fontsize=12)

   # ----- PLOT 2: Normal Q-Q Plot  -----
    # Create a probability plot object
    probplot = sm.ProbPlot(residuals, fit=True)
    
    # Get the theoretical and sample quantiles
    theoretical_quantiles = probplot.theoretical_quantiles
    sample_quantiles = probplot.sample_quantiles
    
    # Plot the scatter points
    sns.scatterplot(x=theoretical_quantiles, y=sample_quantiles, ax=axes[0, 1], alpha=0.8)
    
    # Plot the reference line
    line_coords = np.linspace(min(theoretical_quantiles), max(theoretical_quantiles), 100)
    axes[0, 1].plot(line_coords, line_coords, color='blue', linestyle='--', lw=2)
    
    axes[0, 1].set_title('Normal Q-Q Plot', fontsize=14)
    axes[0, 1].set_xlabel('Theoretical Quantiles', fontsize=12)
    axes[0, 1].set_ylabel('Sample Quantiles', fontsize=12)

    # ----- PLOT 3: Scale-Location Plot -----
    sqrt_studentized_residuals = np.sqrt(np.abs(studentized_residuals))
    sns.scatterplot(x=fitted_values, y=sqrt_studentized_residuals, ax=axes[1, 0], alpha=0.8)
    sns.regplot(x=fitted_values, y=sqrt_studentized_residuals, scatter=False, lowess=True,
                line_kws={'color': 'blue', 'lw': 2}, ax=axes[1, 0])
    axes[1, 0].set_title('Scale-Location Plot', fontsize=14)
    axes[1, 0].set_xlabel('Fitted Values', fontsize=12)
    axes[1, 0].set_ylabel('√|Standardized Residuals|', fontsize=12)

    # ----- PLOT 4: Residuals vs. Leverage -----
    sm.graphics.influence_plot(model, ax=axes[1, 1], criterion="cooks", size=20)
    for text in axes[1, 1].texts:
        text.set_fontsize(8)
    axes[1, 1].set_title('Residuals vs. Leverage', fontsize=14)
    axes[1, 1].set_xlabel('Leverage', fontsize=12)
    axes[1, 1].set_ylabel('Standardized Residuals', fontsize=12)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()

# --- Example Usage ---
# (Assuming you have a fitted model called 'model4')
# check_model_diagnostics(model4)
check_model_diagnostics(model4)