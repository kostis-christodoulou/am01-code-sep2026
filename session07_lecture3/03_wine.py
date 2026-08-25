import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.compat import lzip
import statsmodels.stats.api as sms
from skimpy import skim

# Load the dataset
wine = pd.read_csv('data/wine.csv')


# Replicating skimr::skim(wine) with pandas
print("Data Summary:")
print(wine.info())
print("\nDescriptive Statistics:")
print(wine.describe())

skim(wine)
corr_matrix = wine.corr()  # Select only numeric columns 
print(corr_matrix)

# Produce scatterplot-correlation matrix for 2024

# Create correlation heatmap
plt.figure(figsize=(8, 5))
sns.heatmap(corr_matrix, 
            annot=True, 
            cmap='coolwarm', 
            center=0,
            square=True, 
            linewidths=0.5)
plt.title('Correlation Matrix of Wine Prices', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()

# Scatterplot matrix similar to ggpairs
# Excluding the 'year' column for the pairplot
plt.figure(figsize=(8, 5))
sns.pairplot(wine, diag_kind='kde', plot_kws={'alpha':0.5}, height=1.5, aspect=1)
plt.suptitle('Scatterplot Matrix of Wine Data', y=1.02)
plt.tight_layout()
plt.show()


# --- Regression Models ---

# Model 0: Intercept only
model0 = smf.ols('price ~ 1', data=wine).fit()
print("\n--- Model 0 Summary ---")
print(model0.summary())

# Model 1: price ~ AGST
model1 = smf.ols('price ~ AGST', data=wine).fit()
print("\n--- Model 1 Summary ---")
print(model1.summary())

# Model 2: price ~ AGST + harvest_rain
model2 = smf.ols('price ~ AGST + harvest_rain', data=wine).fit()
print("\n--- Model 2 Summary ---")
print(model2.summary())

# Model 3: Full model (price ~ . )
# We manually list all predictors as statsmodels doesn't have a direct '.' equivalent
predictors_m3 = ' + '.join(wine.columns.drop(['price']))
model3 = smf.ols(f'price ~ {predictors_m3}', data=wine).fit()
print("\n--- Model 3 Summary ---")
print(model3.summary2())

# VIF for Model 3
X_m3 = wine.drop(columns=['price']).assign(Intercept=1)
vif_data = pd.DataFrame()
vif_data["feature"] = X_m3.columns.drop('Intercept')
vif_data["VIF"] = [variance_inflation_factor(X_m3.values, i) for i in range(X_m3.shape[1]-1)]
print("\n--- VIF for Model 3 ---")
print(vif_data)


# Diagnostic plots for Model 3 (similar to check_model)
def plot_model_diagnostics(model):
    residuals = model.resid
    fitted_vals = model.fittedvalues

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Residuals vs Fitted
    sns.residplot(x=fitted_vals, y=residuals, lowess=True,
                  scatter_kws={'alpha': 0.5},
                  line_kws={'color': 'red', 'lw': 2}, ax=axes[0, 0])
    axes[0, 0].set_title('Residuals vs Fitted')
    axes[0, 0].set_xlabel('Fitted values')
    axes[0, 0].set_ylabel('Residuals')

    # Normal Q-Q
    sm.qqplot(residuals, line='s', ax=axes[0, 1])
    axes[0, 1].set_title('Normal Q-Q')

    # Scale-Location
    sqrt_abs_resid = (abs(residuals))**0.5
    axes[1, 0].scatter(fitted_vals, sqrt_abs_resid, alpha=0.5)
    sns.regplot(x=fitted_vals, y=sqrt_abs_resid, scatter=False, lowess=True,
                line_kws={'color': 'red', 'lw': 2}, ax=axes[1, 0])
    axes[1, 0].set_title('Scale-Location')
    axes[1, 0].set_xlabel('Fitted values')
    axes[1, 0].set_ylabel('sqrt{Standardized Residuals}')

    # Residuals vs Leverage
    fig = sm.graphics.influence_plot(model, ax=axes[1, 1], criterion="cooks")
    axes[1, 1].set_title('Residuals vs Leverage')

    plt.tight_layout()
    plt.show()

print("\n--- Diagnostic Plots for Model 3 ---")
plot_model_diagnostics(model3)


# Model 4: price ~ . - age
predictors_m4 = ' + '.join(wine.columns.drop(['price',  'age']))
model4 = smf.ols(f'price ~ {predictors_m4}', data=wine).fit()
print("\n--- Model 4 Summary ---")
print(model4.summary2())

print("\n--- Diagnostic Plots for Model 4 ---")
plot_model_diagnostics(model4)


# Model 5: price ~ . - France_Population
predictors_m5 = ' + '.join(wine.columns.drop(['price', 'France_Population']))
model5 = smf.ols(f'price ~ {predictors_m5}', data=wine).fit()
print("\n--- Model 5 Summary ---")
print(model5.summary())

print("\n--- Diagnostic Plots for Model 5 ---")
plot_model_diagnostics(model5)

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
