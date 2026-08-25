# Regression: Explore relationships among sets of variables.
# As an example, we will use data from the General Social Survey (GSS) to explore the relationship between education, sex, age, and income.

# The GSS dataset contains hundreds of columns and the data can be found at https://gss.norc.org/get-the-data.
# The GSS is a sociological survey used to collect data on demographic characteristics and attitudes of residents of the United States.
# The dataset is maintained by the National Opinion Research Center (NORC) at the University of Chicago.
# We'll work with an extract that contains just the columns we need

import pandas as pd          # THE data analysis library - like Excel but much more powerful
import seaborn as sns        # Makes beautiful statistical plots easily
import matplotlib.pyplot as plt  # The basic plotting library (seaborn uses this underneath)
import numpy as np           # Math operations and working with arrays of numbers
from scipy import stats     # Advanced statistical functions
from skimpy import skim     # Gives us nice data summaries (like R's skim function)
import statsmodels.formula.api as smf  # For regression modeling
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Set up how we want our plots to look (like choosing a theme)
sns.set_style("ticks")      # This gives a clean look with borders

# Load our data from a CSV file
gss = pd.read_csv('data/gss_extract_2022.csv')
gss.head()

# We'll start with a simple regression, estimating the parameters of real income as a function of years of education.
# First we'll select the subset of the data where both variables are valid.
regression_df = gss[['realinc', 'educ']].dropna()


model1 = smf.ols('realinc ~ educ', data=gss).fit()
print(model1.summary2()) # summary2 avoids scientific notation
print(f"Residual SE: {model1.scale ** 0.5:.3f}")  # Residual standard error is the square root of the scale (MSE of residuals)

# Compute the regression of `realinc` as a function of `age`
model2 = smf.ols('realinc ~ age', data=gss).fit()
print(model2.summary2()) 
print(f"Residual SE: {model2.scale ** 0.5:.3f}")  # Residual standard error is the square root of the scale (MSE of residuals)


# In model3 `realinc` is the variable we are trying to explain or predict, which is called the **dependent variable** because it depends on the the other 
# variables -- or at least we expect it to.
# The other variables, `educ` and `age`, are called **independent variables** or sometimes "predictors".
# The `+` sign indicates that we expect the contributions of the independent variables to be additive.
# The result contains an intercept and two slopes, which estimate the average contribution of each predictor with the other predictor held constant.
# * The estimated slope for `educ` is about 3665 -- so if we compare two people with the same age, and one has an additional year of education, 
#  we expect their income to be higher by $3665.
# * The estimated slope for `age` is about 55 -- so if we compare two people with the same education, and one is a year older, we expect their income to be higher by $55.

# In this model, the contribution of age is quite small, but as we'll see in the next section that might be misleading.

model3 = smf.ols('realinc ~ educ + age', data=gss).fit()
print(model3.summary2())
print(f"Residual SE: {model3.scale ** 0.5:.3f}")  # Residual standard error is the square root of the scale (MSE of residuals)


grouped = gss.groupby('age')
type(grouped)
mean_income_by_age = grouped['realinc'].mean().reset_index()

gss.groupby('age')['realinc'].mean()
gss.groupby('age')['realinc'].describe()


# Seaborn scatter plot
sns.scatterplot(data=mean_income_by_age,
                x='age', 
                y='realinc', 
                alpha=0.5)
plt.xlabel('Age (years)')
plt.ylabel('Income (1986 $)')
plt.title('Average income, grouped by age')
plt.show()

# Group by both 'age' and 'sex' in pandas
grouped = gss.groupby(['age', 'sex'])
mean_income = grouped['realinc'].mean().reset_index()

# Seaborn scatter plot

# Define a palette where the first color is blue, the second is orange
custom_palette = ["#1f77b4", "#ff7f0e"]

sns.scatterplot(data=mean_income,
                x='age', 
                y='realinc', 
                hue='sex',
                palette=custom_palette,
                alpha=0.5)
plt.xlabel('Age (years)')
plt.ylabel('Income (1986 $)')
plt.title('Average income, grouped by age and sex')
plt.show()

# Average income increases from age 20 to age 50, then starts to fall.
# And that explains why the estimated slope is so small, because the relationship is non-linear.
# To describe a non-linear relationship, we'll create a new variable called `age2` that equals `age` squared -- so it is called a **quadratic term**.
gss['age2'] = gss['age'] ** 2

# Now we can estimate a model that includes the quadratic term.
model4 = smf.ols('realinc ~ educ + age + age2', data=gss).fit()
print(model4.summary2())
print(f"Residual SE: {model4.scale ** 0.5:.3f}")  # Residual standard error is the square root of the scale (MSE of residuals)


# **Exercise:** The graph in the previous exercise suggests that the relationship between income and education is non-linear.  So let's try fitting a non-linear model.

# - Add a column named `educ2` to the `gss` DataFrame -- it should contain the values from `educ` squared.  
gss['educ2'] = gss['educ'] ** 2

# - Make a scatter plot of `realinc` vs. `educ` (like we did for `age`), and see if the relationship looks non-linear.
sns.scatterplot(data=gss, x='educ', y='realinc', alpha=0.5)
plt.xlabel('Education (years)')
plt.ylabel('Income (1986 $)')
plt.title('Income vs. Education')
plt.show()



# - Add a column named `educ2` to the `gss` DataFrame -- it should contain the values from `educ` squared.  
gss['educ2'] = gss['educ'] ** 2
# - Run a regression that uses `educ`, `educ2`, `age`, and `age2` to predict `realinc`.
model6 = smf.ols('realinc ~ educ + educ2 + age + age2', data=gss).fit()
print(model6.summary2())
print(f"Residual SE: {model6.scale ** 0.5:.3f}")  # Residual standard error is the square root of the scale (MSE of residuals)




# What if we add 'sex' as a predictor?
model5 = smf.ols('realinc ~ educ + age + age2 + C(sex)', data=gss).fit() # C() tells statsmodels to treat 'sex' as a categorical variable
print(model5.summary2())
print(f"Residual SE: {model5.scale ** 0.5:.3f}")  # Residual standard error is the square root of the scale (MSE of residuals)



# One of the GSS questions asks "Would you favor or oppose a law which would require a person to obtain a police permit before he or she could buy a gun?"
# The responses are in a column called `gunlaw` -- here are the values.  
gss['gunlaw'].value_counts() # counts
gss['gunlaw'].value_counts(normalize=True) # proportions

# `1` means yes and `2` means no, so most respondents are in favor.
# Before we can use this variable in a logistic regression, we have to recode it so `1` means "yes" and `0` means "no".
# We can do that by replacing `2` with `0`.
gss['gunlaw'] = gss['gunlaw'].replace({2: 0})

# If we want to predict a category, we can use **logistic regression**.
logit_model = smf.logit('gunlaw ~ educ + educ2 + age + age2 + C(sex)', data=gss).fit()
print(logit_model.summary2())

# The parameters are in the form of **log odds** -- I won't explain them in detail here, except to say that positive values make the outcome more likely 
# and negative values make the outcome less likely.
# For example, the parameter associated with `sex=2` is `0.74`, which indicates that women are more likely to support this form of gun control.

# To see how much more likely, we can generate predictions, as we did with linear regression.
# As an example, we'll generate predictions for different ages and sexes, with education held constant.
# First we need a `DataFrame` with a range of values for `age` and a fixed value of `educ`.

predicted_probs = logit_model.predict()  # predicted probabilities of gunlaw=1 for each row in gss

mask = predicted_probs.notna() & true_labels.notna()


predicted_classes = (predicted_probs >= 0.5).astype(int)  # classify as 1 if predicted_prob >= 0.5, else 0
gss['predicted_gunlaw'] = predicted_classes  # add to gss DataFrame


true_labels = gss['gunlaw'].dropna().astype(int) # actual classes (0 or 1) where gunlaw is not NA
predicted_labels = predicted_classes[gss['gunlaw'].notna()].astype(int) # predicted classes based on thresholding at 0.5


clean_probs = predicted_probs[mask]
clean_true = true_labels[mask]

print(len(true_labels), len(predicted_labels)) # should be the same length


# Now we can compare the predicted values to the actual values.
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve


cm = confusion_matrix(clean_true, clean_probs >= 0.5)  # confusion matrix at threshold 0.5
tn, fp, fn, tp = cm.ravel()

auc_score = roc_auc_score(clean_true, clean_probs)

# Optionally, get ROC curve points:
fpr, tpr, thresholds = roc_curve(clean_true, clean_probs)

