# ==============================================================================
# 1. SETUP: IMPORT LIBRARIES
# ==============================================================================
# Import necessary libraries for data handling, analysis, and plotting.

import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from gapminder import gapminder

# Filter data for the year 2007
gapminder_2007 = gapminder[gapminder['year'] == 2007]

# Create a boxplot of life expectancy by continent

# Sort continents alphabetically for boxplots
sorted_continents = sorted(gapminder_2007['continent'].unique())

plt.figure(figsize=(10, 6))
sns.boxplot(y='continent', x='lifeExp', data=gapminder_2007, order=sorted_continents, boxprops=dict(facecolor='none'))
plt.title('Life Expectancy by Continent in 2007')
plt.ylabel('')
plt.xlabel('Life Expectancy')
plt.show()


# Regression: lifeExp by continent for 2007
model = smf.ols('lifeExp ~ C(continent)', data=gapminder_2007).fit()
print(model.summary().tables[1])

# =============================================================================
# Lifexp by time to find increase in life expectancy over time
# =============================================================================

# Function to plot scatterplot and run regression for any selected country
def run_country_regression(country_name):

    # Scatter plot of life expectancy vs year for the selected country
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x='year', 
                    y='lifeExp', 
                    data=gapminder[gapminder['country'] == country_name], 
                    color='black')
    sns.regplot(x='year', 
                y='lifeExp', 
                data=gapminder[gapminder['country'] == country_name], 
                scatter=False, 
                color='blue', 
                line_kws={'label':"Best Fit"})
    plt.title(f'Life Expectancy in {country_name} 1952-2007')
    plt.xlabel('')
    plt.ylabel('Life Expectancy')
    plt.show()
    country_data = gapminder[gapminder['country'] == country_name].copy()
    country_data['year_offset'] = country_data['year'] - 1952
    model = smf.ols('lifeExp ~ year_offset', data=country_data).fit()
    print(f"Regression results for {country_name}:")
    print(model.summary().tables[1])

# Example usage:
run_country_regression('Greece')
run_country_regression('Turkey')
run_country_regression('China')
run_country_regression('Rwanda')