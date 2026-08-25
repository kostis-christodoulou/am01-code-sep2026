import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

my_alpha = 0.68

# Create figure with subplots
fig, axes = plt.subplots(2, 4, figsize=(16, 10))
fig.suptitle('Many Distributions', fontsize=16)

# Standard Normal distribution
x_normal = np.linspace(-4, 4, 1000)
y_normal = stats.norm.pdf(x_normal, 0, 1)
axes[0, 0].fill_between(x_normal, y_normal, alpha=my_alpha, color='steelblue')
axes[0, 0].set_title('Standard Normal distribution')
axes[0, 0].set_ylabel('dnorm(0,1)')
axes[0, 0].grid(True, alpha=0.3)

# Uniform distribution
x_uniform = np.linspace(0, 1, 1000)
y_uniform = stats.uniform.pdf(x_uniform, 0, 1)
axes[0, 1].fill_between(x_uniform, y_uniform, alpha=my_alpha, color='steelblue')
axes[0, 1].set_title('Uniform distribution')
axes[0, 1].set_ylabel('dunif(0,1)')
axes[0, 1].grid(True, alpha=0.3)

# Binomial distribution (p=0.5, n=50)
x_binom1 = np.arange(0, 51)
y_binom1 = stats.binom.pmf(x_binom1, 50, 0.5)
axes[0, 2].bar(x_binom1, y_binom1, alpha=my_alpha, color='steelblue')
axes[0, 2].set_title('Binomial, p = 0.5, size = 50')
axes[0, 2].set_xlabel('')
axes[0, 2].set_ylabel('')
axes[0, 2].grid(True, alpha=0.3)

# Binomial distribution (p=0.1, n=50)
x_binom2 = np.arange(0, 51)
y_binom2 = stats.binom.pmf(x_binom2, 50, 0.1)
axes[0, 3].bar(x_binom2, y_binom2, alpha=my_alpha, color='steelblue')
axes[0, 3].set_title('Binomial, p = 0.1, size = 50')
axes[0, 3].set_xlabel('')
axes[0, 3].set_ylabel('')
axes[0, 3].grid(True, alpha=0.3)

# Gamma distribution (shape=1.5, scale=3)
x_gamma2 = np.linspace(0, 30, 1000)
y_gamma2 = stats.gamma.pdf(x_gamma2, 1.5, scale=3)
axes[1, 0].fill_between(x_gamma2, y_gamma2, alpha=my_alpha, color='steelblue')
axes[1, 0].set_title('Gamma, shape=1.5, scale=3')
axes[1, 0].set_xlabel('')
axes[1, 0].set_ylabel('')
axes[1, 0].grid(True, alpha=0.3)

# Gamma distribution (shape=4, scale=2)
x_gamma1 = np.linspace(0, 30, 1000)
y_gamma1 = stats.gamma.pdf(x_gamma1, 4, scale=2)
axes[1, 1].fill_between(x_gamma1, y_gamma1, alpha=my_alpha, color='steelblue')
axes[1, 1].set_title('Gamma, shape=4, scale=2')
axes[1, 1].set_xlabel('')
axes[1, 1].set_ylabel('')
axes[1, 1].grid(True, alpha=0.3)

# Exponential distribution
x_exp = np.linspace(0, 4, 1000)
y_exp = stats.expon.pdf(x_exp, 0, 1)
axes[1, 2].fill_between(x_exp, y_exp, alpha=my_alpha, color='steelblue')
axes[1, 2].set_title('Exponential distribution')
axes[1, 2].set_ylabel('dexp(rate = 1)')
axes[1, 2].grid(True, alpha=0.3)

# Log-Normal distribution
x_lognorm = np.linspace(0, 4, 1000)
y_lognorm = stats.lognorm.pdf(x_lognorm, 1, 0, 1)
axes[1, 3].fill_between(x_lognorm, y_lognorm, alpha=my_alpha, color='steelblue')
axes[1, 3].set_title('Log-Normal distribution')
axes[1, 3].set_ylabel('dlnorm(0,1)')
axes[1, 3].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Note: The skewness of the gamma distribution only depends on its shape parameter, k, 
# and it is equal to 2/sqrt(k) 
print("Skewness of Gamma distribution with shape=1.5:", stats.gamma.stats(1.5, moments='s'))