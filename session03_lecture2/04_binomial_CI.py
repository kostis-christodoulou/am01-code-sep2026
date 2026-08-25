# =============================================================================
# 1. IMPORT LIBRARIES
# =============================================================================
# - statsmodels: A powerful library for many statistical models and tests, including confidence intervals for proportions.
# - scipy.stats: A core scientific library that includes functions for statistical tests, like the binomial test.

from statsmodels.stats.proportion import proportion_confint
from scipy.stats import binomtest


# =============================================================================
# 2. SETUP PARAMETERS
# =============================================================================
# By defining all our inputs in one place, the script becomes much easier
# to read and modify for a different problem.

# --- Observed Data ---
num_successes = 91      # The number of successful outcomes we observed (e.g., 91 heads).
num_trials = 198        # The total number of trials (e.g., 198 coin flips).

# --- Analysis Settings ---
# We want a 95% confidence interval. The significance level (alpha) is 1 minus the confidence level.
confidence_level = 0.95
alpha = 1 - confidence_level

# For our hypothesis test, we need a "null hypothesis". Let's test if the
# true proportion could be 50% (like a fair coin).
hypothesized_proportion = 0.5


# =============================================================================
# 3. ANALYSIS 1: CALCULATE THE CONFIDENCE INTERVAL
# =============================================================================
# A confidence interval gives us a range of plausible values for the true,
# underlying proportion of successes in the population, based on our sample data.

print("--- Analysis 1: Confidence Interval ---")

# Use statsmodels to calculate the interval.
# - count: The number of successes observed.
# - nobs: The total number of observations (trials).
# - alpha: The significance level. alpha=0.05 corresponds to a 95% CI.
# - method='normal': Uses the standard normal approximation, which is common.
ci_low, ci_upp = proportion_confint(count=num_successes,
                                    nobs=num_trials,
                                    alpha=alpha,
                                    method='normal')

# round the results to 3 d.p. after they have been calculated.
ci_low_rounded = round(ci_low, 3)
ci_upp_rounded = round(ci_upp, 3)

# Print the result in a full sentence to provide context and aid understanding.
print(f"Observed Proportion: {num_successes / num_trials:.3f}")
print(f"The {confidence_level:.0%} confidence interval for the true proportion is ({ci_low_rounded}, {ci_upp_rounded}).")
print("Interpretation: We are 95% confident that the true underlying proportion of successes is between "
      f"{ci_low_rounded*100:.1f}% and {ci_upp_rounded*100:.1f}%.")


# =============================================================================
# 4. ANALYSIS 2: PERFORM A HYPOTHESIS TEST
# =============================================================================
# A hypothesis test gives us a probability (the p-value) to help us decide
# if our observed data is statistically different from a hypothesized value.

print("\n--- Analysis 2: Binomial Hypothesis Test ---")

# We are testing the following hypotheses:
# - Null Hypothesis (H₀): The true proportion of successes is equal to our hypothesized value (p = 0.5).
# - Alternative Hypothesis (Hₐ): The true proportion is NOT equal to 0.5.

# `binomtest` calculates the exact probability of seeing a result as extreme
# as ours (or more extreme), assuming the null hypothesis is true.
test_result = binomtest(k=num_successes,
                        n=num_trials,
                        p=hypothesized_proportion)

# The p-value is the key output of the test.
p_value = test_result.pvalue

print(f"Null Hypothesis: The true proportion is {hypothesized_proportion}")
print(f"Calculated p-value: {p_value:.4f}")

# An if/else block makes the conclusion clear for the user.
# We compare the p-value to our significance level (alpha).
if p_value < alpha:
    print(f"Conclusion: Since the p-value ({p_value:.4f}) is less than our significance level ({alpha}), "
          "we reject the null hypothesis.")
    print("This means our observed result is statistically significant and is unlikely to have occurred if the true proportion was 0.5.")
else:
    print(f"Conclusion: Since the p-value ({p_value:.4f}) is greater than our significance level ({alpha}), "
          "we fail to reject the null hypothesis.")
    print("This means our observed result is not statistically significant; it's consistent with what we might see if the true proportion was 0.5.")


# =============================================================================
# 5. FINAL SUMMARY
# =============================================================================
# We can connect the results from both analyses.
print("\n--- Summary ---")
if hypothesized_proportion >= ci_low and hypothesized_proportion <= ci_upp:
    print(f"The hypothesized proportion ({hypothesized_proportion}) falls WITHIN our 95% confidence interval ({ci_low_rounded}, {ci_upp_rounded}).")
    print("This aligns with our hypothesis test conclusion: we cannot reject the idea that the true proportion might be 0.5.")
else:
    print(f"The hypothesized proportion ({hypothesized_proportion}) falls OUTSIDE our 95% confidence interval ({ci_low_rounded}, {ci_upp_rounded}).")
    print("This aligns with our hypothesis test conclusion: we can reject the idea that the true proportion is 0.5.")