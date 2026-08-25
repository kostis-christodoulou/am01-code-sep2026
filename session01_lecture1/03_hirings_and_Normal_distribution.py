# Import all the libraries we need for statistical analysis and visualization
import matplotlib.pyplot as plt  # Used for creating plots and customizing them
import numpy as np               # Used for numerical operations and creating arrays
from scipy import stats         # Used for statistical functions like normal distribution calculations
import seaborn as sns           # Used for beautiful statistical visualizations

# Set style for better looking plots - makes charts more professional
plt.style.use('seaborn-v0_8')  # Apply seaborn styling to all plots
sns.set_palette("husl")        # Set a colorful palette for consistent colors

# PROBLEM SETUP
# Test scores are normally distributed with a mean of 525 and standard deviation of 55. 
# Automatic accepts have exam score >= 600 and automatic rejects score <= 425
# We want to calculate the percentage of applicants who are automatically accepted and rejected

# Define the parameters of our normal distribution
mean_score = 525  # Average test score across all applicants
sd_score = 55     # Standard deviation - measures how spread out the scores are

# PROBABILITY CALCULATIONS SECTION
# Calculate probabilities using the cumulative distribution function (CDF)

# For those automatically accepted (score >= 600)
# We use 1 - CDF because CDF gives us P(X <= 600), but we want P(X >= 600)
prob_auto_accept = 1 - stats.norm.cdf(600, mean_score, sd_score)
print(f"Percentage automatically accepted (score >= 600): {prob_auto_accept:.4f} ({prob_auto_accept*100:.2f}%)")

# For those automatically rejected (score <= 425)
# CDF directly gives us P(X <= 425), which is exactly what we want
prob_auto_reject = stats.norm.cdf(425, mean_score, sd_score)
print(f"Percentage automatically rejected (score <= 425): {prob_auto_reject:.4f} ({prob_auto_reject*100:.2f}%)")

# VISUALIZATION 1: Original Thresholds
# Create a comprehensive visualization showing the distribution and cutoff points

# Create a figure with specified size (12 inches wide, 8 inches tall)
plt.figure(figsize=(12, 8))

# Generate x-values for our plot: from 4 standard deviations below mean to 4 above
# This covers virtually all possible scores (99.99% of the distribution)
x_range = np.linspace(mean_score - 4*sd_score, mean_score + 4*sd_score, 1000)

# Calculate the probability density for each x-value
# PDF (Probability Density Function) gives us the height of the curve at each point
y_range = stats.norm.pdf(x_range, mean_score, sd_score)

# Create the main distribution curve using seaborn
sns.lineplot(x=x_range, y=y_range, linewidth=2, label='Normal distribution', color='blue')

# Fill areas under the curve to show different regions
# Green area: automatically accepted applicants (score >= 600)
plt.fill_between(x_range, y_range, where=(x_range >= 600), alpha=0.6, color='green', 
                 label=f'Auto accept (≥600): {prob_auto_accept*100:.1f}%')

# Red area: automatically rejected applicants (score <= 425)
plt.fill_between(x_range, y_range, where=(x_range <= 425), alpha=0.6, color='red', 
                 label=f'Auto reject (≤425): {prob_auto_reject*100:.1f}%')

# Add vertical lines to mark important thresholds
plt.axvline(600, color='green', linestyle='--', linewidth=2, label='Auto accept threshold')  # Accept threshold
plt.axvline(425, color='red', linestyle='--', linewidth=2, label='Auto reject threshold')    # Reject threshold
plt.axvline(mean_score, color='black', linestyle='-', linewidth=1, label=f'Mean: {mean_score}')  # Population mean

# Customize the plot appearance
plt.title('Test Score Distribution with Automatic Accept/Reject Thresholds', fontsize=16)
plt.xlabel('Test Score', fontsize=12)
plt.ylabel('Density', fontsize=12)  # Density = probability per unit on x-axis
plt.legend(fontsize=10)  # Show legend explaining colors and lines
plt.grid(True, alpha=0.3)  # Add light grid for easier reading
plt.show()

# INVERSE PROBABILITY CALCULATIONS SECTION
# Now we solve the reverse problem: if we want specific percentages, what should the thresholds be?
# If we wanted to automatically accept 15% and automatically reject 10%, 
# we need to find the scores that correspond to these percentiles

print("\n" + "="*60)
print("INVERSE CALCULATION: Finding thresholds for desired percentages")
print("="*60)

# Find the Z-score (standardized score) that corresponds to the 85th percentile
# 85th percentile means 85% of scores are below this point, so 15% are above (auto accept)
z_85 = stats.norm.ppf(0.85)  # ppf = percent point function (inverse of CDF)
print(f"Z-score for 85th percentile: {z_85:.4f}")

# Find the Z-score that corresponds to the 10th percentile
# 10th percentile means 10% of scores are below this point (auto reject)
z_10 = stats.norm.ppf(0.10)
print(f"Z-score for 10th percentile: {z_10:.4f}")

# Convert Z-scores back to actual test scores using our distribution parameters
# Formula: X = mean + Z * standard_deviation
x_85 = stats.norm.ppf(0.85, mean_score, sd_score)  # Test score for 85th percentile
x_10 = stats.norm.ppf(0.10, mean_score, sd_score)  # Test score for 10th percentile

print(f"Test score for 85th percentile (auto accept 15%): {x_85:.1f}")
print(f"Test score for 10th percentile (auto reject 10%): {x_10:.1f}")

# VISUALIZATION 2: New Thresholds
# Show what the distribution looks like with our new percentage-based thresholds

plt.figure(figsize=(12, 8))

# Create the main distribution curve using seaborn (same as before)
sns.lineplot(x=x_range, y=y_range, linewidth=2, label='Normal distribution', color='blue')

# Fill areas under the curve with new thresholds
# Green area: top 15% of applicants (automatically accepted)
plt.fill_between(x_range, y_range, where=(x_range >= x_85), alpha=0.6, color='green', 
                 label=f'Auto accept (≥{x_85:.1f}): 15%')

# Red area: bottom 10% of applicants (automatically rejected)
plt.fill_between(x_range, y_range, where=(x_range <= x_10), alpha=0.6, color='red', 
                 label=f'Auto reject (≤{x_10:.1f}): 10%')

# Add vertical lines for the new thresholds
plt.axvline(x_85, color='green', linestyle='--', linewidth=2, label=f'Auto accept threshold: {x_85:.1f}')
plt.axvline(x_10, color='red', linestyle='--', linewidth=2, label=f'Auto reject threshold: {x_10:.1f}')
plt.axvline(mean_score, color='black', linestyle='-', linewidth=1, label=f'Mean: {mean_score}')

# Customize the plot appearance
plt.title('Test Score Distribution with 15% Auto Accept / 10% Auto Reject Thresholds', fontsize=16)
plt.xlabel('Test Score', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.show()

# COMPARISON VISUALIZATION
# Create a side-by-side comparison using seaborn subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Left plot: Original thresholds
sns.lineplot(x=x_range, y=y_range, linewidth=2, color='blue', ax=ax1)
ax1.fill_between(x_range, y_range, where=(x_range >= 600), alpha=0.6, color='green', 
                label=f'Accept: {prob_auto_accept*100:.1f}%')
ax1.fill_between(x_range, y_range, where=(x_range <= 425), alpha=0.6, color='red', 
                label=f'Reject: {prob_auto_reject*100:.1f}%')
ax1.axvline(600, color='green', linestyle='--', linewidth=2)
ax1.axvline(425, color='red', linestyle='--', linewidth=2)
ax1.axvline(mean_score, color='black', linestyle='-', linewidth=1)
ax1.set_title('Original Thresholds\n(Fixed Scores: 425, 600)', fontsize=14)
ax1.set_xlabel('Test Score')
ax1.set_ylabel('Density')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Right plot: New percentage-based thresholds
sns.lineplot(x=x_range, y=y_range, linewidth=2, color='blue', ax=ax2)
ax2.fill_between(x_range, y_range, where=(x_range >= x_85), alpha=0.6, color='green', 
                label='Accept: 15.0%')
ax2.fill_between(x_range, y_range, where=(x_range <= x_10), alpha=0.6, color='red', 
                label='Reject: 10.0%')
ax2.axvline(x_85, color='green', linestyle='--', linewidth=2)
ax2.axvline(x_10, color='red', linestyle='--', linewidth=2)
ax2.axvline(mean_score, color='black', linestyle='-', linewidth=1)
ax2.set_title('New Thresholds\n(Fixed Percentages: 10%, 15%)', fontsize=14)
ax2.set_xlabel('Test Score')
ax2.set_ylabel('Density')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()  # Ensure plots don't overlap
plt.show()

# SUMMARY RESULTS SECTION
# Create a comprehensive summary table comparing both approaches

print("\n" + "="*70)
print("COMPREHENSIVE SUMMARY OF THRESHOLD APPROACHES")
print("="*70)

print(f"\n📊 APPROACH 1: Fixed Score Thresholds")
print(f"   Auto accept threshold: 600 points")
print(f"   Auto reject threshold: 425 points")
print(f"   Results:")
print(f"     • Auto accept: {prob_auto_accept*100:.1f}% of applicants")
print(f"     • Auto reject: {prob_auto_reject*100:.1f}% of applicants")
print(f"     • Manual review: {(1-prob_auto_accept-prob_auto_reject)*100:.1f}% of applicants")

print(f"\n📊 APPROACH 2: Fixed Percentage Thresholds")
print(f"   Auto accept threshold: {x_85:.1f} points (85th percentile)")
print(f"   Auto reject threshold: {x_10:.1f} points (10th percentile)")
print(f"   Results:")
print(f"     • Auto accept: 15.0% of applicants")
print(f"     • Auto reject: 10.0% of applicants")
print(f"     • Manual review: 75.0% of applicants")

print(f"\n💡 KEY INSIGHTS:")
print(f"   • Fixed scores give variable percentages depending on applicant pool")
print(f"   • Fixed percentages give variable scores but consistent selection rates")
print(f"   • Mean score: {mean_score}, Standard deviation: {sd_score}")
print(f"   • Both approaches can be valid depending on institutional goals")

# ADDITIONAL ANALYSIS: What percentage falls in manual review for each approach?
manual_review_original = (1 - prob_auto_accept - prob_auto_reject) * 100
manual_review_new = 75.0  # By design: 100% - 15% - 10%

print(f"\n📋 WORKLOAD COMPARISON:")
print(f"   Original approach: {manual_review_original:.1f}% need manual review")
print(f"   New approach: {manual_review_new:.1f}% need manual review")

if manual_review_original > manual_review_new:
    difference = manual_review_original - manual_review_new
    print(f"   💼 New approach reduces manual review workload by {difference:.1f} percentage points")
else:
    difference = manual_review_new - manual_review_original
    print(f"   💼 New approach increases manual review workload by {difference:.1f} percentage points")