# Import necessary libraries
import numpy as np  # For numerical operations, especially with arrays
import pandas as pd  # For data manipulation and analysis (using DataFrames)
from dataclasses import dataclass  # A decorator to easily create classes for storing data
import matplotlib.pyplot as plt  # For creating static, animated, and interactive visualizations
import seaborn as sns  # A high-level interface for drawing attractive and informative statistical graphics


# A simple class to store the results of an inference test.
# The '@dataclass' decorator automatically generates special methods like __init__ and __repr__.
@dataclass
class InferResult:
    """Stores the result of an inference calculation."""
    obs: float  # The observed statistic from the original data (e.g., the actual difference in means)
    null: np.ndarray  # An array of statistics generated under the null hypothesis (the null distribution)
    meta: dict  # A dictionary to store metadata about the test (e.g., stat type, null type)

    def get_p_value(self, direction="two-sided"):
        """
        Calculates the p-value from the null distribution.
        The p-value is the probability of observing a statistic as extreme as or more extreme than
        the one from our original sample, assuming the null hypothesis is true.
        """
        null = self.null  # The array of simulated statistics
        obs = self.obs    # The single observed statistic from the real data
        center = null.mean()  # The center of the null distribution

        # For a two-sided test, we look for values as far from the center as our observed statistic, in either direction.
        if direction == "two-sided":
            # Calculate the proportion of null values that are as extreme as the observed value.
            return (np.abs(null - center) >= abs(obs - center)).mean()
        # For a right-tailed test, we look for values greater than or equal to our observed statistic.
        if direction == "right":
            return (null >= obs).mean()
        # For a left-tailed test, we look for values less than or equal to our observed statistic.
        if direction == "left":
            return (null <= obs).mean()
        # If the direction is not one of the valid options, raise an error.
        raise ValueError("direction must be 'two-sided', 'left', or 'right'")


# The main class that orchestrates the entire inference process.
# It uses a "chaining" pattern where you call one method after another (e.g., Infer(df).specify(...).hypothesize(...)).
class Infer:
    def __init__(self, df: pd.DataFrame):
        """Initializes the inference process with a pandas DataFrame."""
        self.df = df
        # Initialize internal variables to store the user's specifications.
        # These are prefixed with an underscore (_) to indicate they are for internal use.
        self._response = None       # The dependent variable (y)
        self._group = None          # The categorical explanatory variable
        self._predictor = None      # The numeric explanatory variable (x)
        self._success = None        # The label for a "success" outcome in proportions
        self._null = None           # The type of null hypothesis ('independence' or 'point')
        self._mu = None             # The hypothesized mean/median for a point null
        self._p = None              # The hypothesized proportion for a point null
        self._sigma = None          # A known standard deviation (for z-tests)
        self._reps = 1000           # The number of repetitions for the simulation
        self._type = None           # The simulation method ('permute', 'bootstrap', 'simulate')
        self._seed = 42             # A seed for the random number generator to ensure reproducibility

    def specify(self, formula=None, response=None, group=None, predictor=None, success=None):
        """
        Specifies the variables to be used in the analysis.
        This can be done using a formula string like "y ~ x" or by passing column names directly.
        """
        if formula is not None:
            # If a formula is provided (e.g., "height ~ gender"), parse it.
            lhs, rhs = [s.strip() for s in formula.split("~", 1)]
            self._response = lhs  # The left-hand side is the response variable
            # Determine if the right-hand side is a numeric predictor or a categorical group.
            if pd.api.types.is_numeric_dtype(self.df[rhs]):
                self._predictor = rhs
            else:
                self._group = rhs
        else:
            # If arguments are passed directly, assign them.
            self._response = response
            self._group = group
            self._predictor = predictor
        
        self._success = success  # Store the success label if provided (for proportions)
        return self  # Return the object itself to allow method chaining

    def hypothesize(self, null="independence", mu=None, p=None, sigma=None):
        """Sets up the null hypothesis for the test."""
        if null not in ("independence", "point"):
            raise ValueError("null must be 'independence' or 'point'")
        self._null = null      # 'independence' (no relationship) or 'point' (a specific value)
        self._mu = mu          # The hypothesized mean for a point null test
        self._p = p            # The hypothesized proportion for a point null test
        self._sigma = sigma    # The known population standard deviation
        return self

    def generate(self, reps=1000, type="permute", seed=42):
        """
        Generates the null distribution through simulation.
        - 'permute': Shuffles labels to break any existing relationship (for independence).
        - 'bootstrap': Samples with replacement from the original data (for point estimates).
        - 'simulate': Draws from a theoretical model (e.g., binomial for proportions).
        """
        self._reps = int(reps)
        self._type = type
        self._seed = seed
        return self

    def calculate(self, stat, order=None):
        """
        Calculates the observed statistic and generates the null distribution.
        This is the core method where the statistical logic resides.
        """
        if self._response is None:
            raise ValueError("You must specify variables using specify() first.")
        
        # Set up a random number generator with a seed for reproducibility.
        rng = np.random.default_rng(self._seed)
        # Clean up the statistic name.
        stat_name = stat.lower().strip() if isinstance(stat, str) else None

        # --- ONE-SAMPLE TESTS ---
        # This block handles tests on a single variable (e.g., is the average height 65 inches?).
        if self._group is None and self._predictor is None:
            if self._null != "point":
                raise ValueError("One-sample tests require hypothesize(null='point').")
            
            # Get the data for the response variable and remove any missing values.
            x = self.df[self._response].dropna().to_numpy()
            n = x.size

            # Check which statistic the user wants to calculate.
            if stat_name in ("mean", "median", "sum", "sd", "prop", "count", "t", "z"):
                # --- Calculate the Observed Statistic from the original data ---
                if stat_name == "mean":
                    obs = x.mean()
                elif stat_name == "median":
                    obs = np.median(x)
                elif stat_name == "sd":
                    obs = x.std(ddof=1)  # ddof=1 for sample standard deviation
                elif stat_name == "prop":
                    obs = x.mean() # For 0/1 data, the mean is the proportion of 1s
                elif stat_name == "t":
                    if self._mu is None: raise ValueError("t-statistic requires a null mean 'mu'.")
                    s = x.std(ddof=1)
                    obs = (x.mean() - self._mu) / (s / np.sqrt(n))
                elif stat_name == "z":
                    if self._mu is None and self._p is None: raise ValueError("z-statistic requires 'mu' or 'p'.")
                    # z-statistic for a mean with a known sigma
                    if self._sigma is not None:
                        obs = (x.mean() - self._mu) / (self._sigma / np.sqrt(n))
                    # z-statistic for a proportion
                    else:
                        p0 = self._p
                        phat = x.mean()
                        obs = (phat - p0) / np.sqrt(p0 * (1 - p0) / n)

                # --- Generate the Null Distribution ---
                # For proportions, we can simulate directly from a binomial distribution.
                if stat_name in ("prop",) and self._type == "simulate":
                    if self._p is None: raise ValueError("Simulating a proportion requires a null proportion 'p'.")
                    # Generate 'reps' number of proportions from 'n' trials with probability 'p'.
                    null = rng.binomial(n=n, p=self._p, size=self._reps) / n
                else:
                    # For most other one-sample stats, we use bootstrapping.
                    if self._type != "bootstrap":
                        raise ValueError("Use generate(type='bootstrap') for one-sample tests.")
                    # Create bootstrap samples by resampling with replacement.
                    boots = rng.choice(x, size=(self._reps, n), replace=True)
                    
                    # Calculate the statistic for each bootstrap sample.
                    if stat_name == "mean":
                        bstats = boots.mean(axis=1)
                        # Center the bootstrap distribution at the null hypothesis mean.
                        null = bstats - bstats.mean() + float(self._mu)
                    elif stat_name == "median":
                        bstats = np.median(boots, axis=1)
                        null = bstats - bstats.mean() + float(self._mu)
                    elif stat_name == "sd":
                        bstats = boots.std(axis=1, ddof=1)
                        # Center at the observed sd since there's no point null for sd.
                        null = bstats - bstats.mean() + float(obs)
                    elif stat_name == "t":
                        bmeans = boots.mean(axis=1)
                        bsds = boots.std(axis=1, ddof=1)
                        # Calculate t-statistic for each bootstrap sample.
                        null = (bmeans - float(self._mu)) / (bsds / np.sqrt(n))

                # Return the result object containing the observed stat and null distribution.
                return InferResult(obs=obs, null=null, meta={"stat": stat_name, "null": "point", "type": self._type, "reps": self._reps})

            raise ValueError("Unsupported one-sample statistic.")

        # --- TWO-VARIABLE TESTS ---
        # This block handles tests for relationships between two variables.
        # Create a clean DataFrame with no missing values for the variables of interest.
        d = self.df[[c for c in [self._response, self._group, self._predictor] if c is not None]].dropna().copy()

        # --- Numeric vs. Numeric (Slope, Correlation) ---
        if self._predictor is not None and self._group is None:
            if self._null != "independence" or self._type != "permute":
                raise ValueError("For slope/correlation, use hypothesize('independence') and generate('permute').")
            
            y = d[self._response].to_numpy()
            x = d[self._predictor].to_numpy()

            # Helper function to calculate slope
            def slope(yv, xv):
                xbar, ybar = xv.mean(), yv.mean()
                num = ((xv - xbar) * (yv - ybar)).sum()
                den = ((xv - xbar) ** 2).sum()
                return num / den

            if stat_name == "slope":
                obs = slope(y, x)
                null = np.empty(self._reps)
                # For the null distribution, repeatedly shuffle the y-values and recalculate the slope.
                for i in range(self._reps):
                    y_perm = rng.permutation(y)
                    null[i] = slope(y_perm, x)
                return InferResult(obs=obs, null=null, meta={"stat": "slope", "null": "independence", "type": "permute"})

            if stat_name == "correlation":
                obs = np.corrcoef(x, y)[0, 1]
                null = np.empty(self._reps)
                # Similarly, shuffle y-values and recalculate the correlation.
                for i in range(self._reps):
                    y_perm = rng.permutation(y)
                    null[i] = np.corrcoef(x, y_perm)[0, 1]
                return InferResult(obs=obs, null=null, meta={"stat": "correlation", "null": "independence", "type": "permute"})
            
            raise ValueError("Supported stats for numeric-numeric are 'slope' and 'correlation'.")

        # --- Categorical Grouping Variable ---
        if self._group is None:
            raise ValueError("A grouping variable or a numeric predictor is required.")

        gcol = self._group
        ycol = self._response
        levels = pd.Index(d[gcol].unique())

        # --- Chi-Square Test (Categorical vs. Categorical) ---
        if stat_name in ("chisq", "chi-square"):
            # Create a contingency table (crosstab) of observed counts.
            tbl = pd.crosstab(d[ycol], d[gcol])
            obs_counts = tbl.to_numpy()
            # Calculate expected counts under the null hypothesis of independence.
            row_sums = obs_counts.sum(axis=1, keepdims=True)
            col_sums = obs_counts.sum(axis=0, keepdims=True)
            total = obs_counts.sum()
            exp = row_sums @ col_sums / total
            # The chi-square statistic is the sum of (obs - exp)^2 / exp.
            obs = ((obs_counts - exp) ** 2 / exp).sum()
            
            # Generate null distribution by permuting the group labels.
            null = np.empty(self._reps)
            g_vals = d[gcol].to_numpy()
            for i in range(self._reps):
                # Shuffle the group labels and recalculate the chi-square stat.
                perm_g_vals = rng.permutation(g_vals)
                # ... (logic to calculate chi-square for the permuted data)
            return InferResult(obs=obs, null=null, meta={"stat": "chisq", "null": "independence", "type": "permute"})

        # --- Two-Sample Tests (for groups with exactly two levels) ---
        if len(levels) != 2:
            raise ValueError("Two-sample statistics require exactly two group levels.")
        
        # Determine the order of the groups for subtraction (e.g., group1 - group0).
        if order is None:
            order = [levels[0], levels[1]]
        g1, g0 = order

        # --- Numeric Response Variable (e.g., height vs. gender) ---
        if pd.api.types.is_numeric_dtype(d[ycol]):
            # Split the data into two groups.
            x1 = d.loc[d[gcol] == g1, ycol].to_numpy()
            x0 = d.loc[d[gcol] == g0, ycol].to_numpy()

            # Calculate the observed statistic.
            if stat_name == "diff in means":
                obs = x1.mean() - x0.mean()
            elif stat_name == "diff in medians":
                obs = np.median(x1) - np.median(x0)
            elif stat_name in ("diff in sd", "diff in std", "diff in sds"):
                obs = x1.std(ddof=1) - x0.std(ddof=1)
            elif stat_name == "t":
                m1, m0 = x1.mean(), x0.mean()
                s1, s0 = x1.std(ddof=1), x0.std(ddof=1)
                n1, n0 = x1.size, x0.size
                obs = (m1 - m0) / np.sqrt((s1**2 / n1) + (s0**2 / n0))
            else:
                raise ValueError("Unsupported numeric two-sample statistic.")
            
            # Generate the null distribution by permuting the group labels.
            null = np.empty(self._reps)
            all_y = d[ycol].to_numpy()
            for i in range(self._reps):
                # Shuffle all response values.
                perm_y = rng.permutation(all_y)
                # Assign the first n1 values to group 1, the rest to group 0.
                a = perm_y[:len(x1)]
                b = perm_y[len(x1):]
                
                # Recalculate the statistic for the permuted groups.
                if stat_name == "diff in means":
                    null[i] = a.mean() - b.mean()
                elif stat_name == "diff in medians":
                    null[i] = np.median(a) - np.median(b)
                elif stat_name in ("diff in sd", "diff in std", "diff in sds"):
                    null[i] = a.std(ddof=1) - b.std(ddof=1)
                elif stat_name == "t":
                    m1, m0 = a.mean(), b.mean()
                    s1, s0 = a.std(ddof=1), b.std(ddof=1)
                    n1, n0 = a.size, b.size
                    null[i] = (m1 - m0) / np.sqrt((s1**2 / n1) + (s0**2 / n0))

            return InferResult(obs=obs, null=null, meta={"stat": stat_name, "order": order, "null": "independence", "type": "permute"})
        
        # --- Categorical Response (Proportions) ---
        if self._success is None:
            raise ValueError("For proportion-based stats, specify the 'success' label.")
        
        # Calculate counts for a 2x2 contingency table.
        a = ((d[gcol] == g1) & (d[ycol] == self._success)).sum() # group1, success
        b = ((d[gcol] == g1) & (d[ycol] != self._success)).sum() # group1, failure
        c = ((d[gcol] == g0) & (d[ycol] == self._success)).sum() # group0, success
        e = ((d[gcol] == g0) & (d[ycol] != self._success)).sum() # group0, failure
        
        n1 = a + b
        n0 = c + e
        p1 = a / n1 if n1 > 0 else 0
        p0 = c / n0 if n0 > 0 else 0
        
        # Calculate observed statistic.
        if stat_name == "diff in props":
            obs = p1 - p0
        elif stat_name == "odds ratio":
            # Add 0.5 to prevent division by zero (Haldane-Anscombe correction).
            obs = (a + 0.5) * (e + 0.5) / ((b + 0.5) * (c + 0.5))
        else:
            raise ValueError("Unsupported categorical two-sample statistic.")
            
        # Generate null distribution by permutation.
        null = np.empty(self._reps)
        # ... (permutation logic similar to the numeric case) ...
        
        return InferResult(obs=obs, null=null, meta={"stat": stat_name, "order": order, "null": "independence", "type": "permute"})


# --- Visualization Helper Functions ---

def _shade(ax, null, obs, direction):
    """A helper function to shade the p-value area on the histogram."""
    # ... (code to draw the shaded region) ...

def _format_title(meta, obs, p):
    """A helper function to create a clean title for the plot."""
    # ... (code to format the title string) ...

# We need to redefine the InferResult class here so its methods are available to the rest of the script.
class InferResult:
    def __init__(self, obs, null, meta):
        self.obs = obs
        self.null = np.asarray(null)
        self.meta = meta

    def get_p_value(self, direction="two-sided"):
        # ... (p-value calculation logic as defined before) ...
        center = self.null.mean()
        if direction == "two-sided":
            return (np.abs(self.null - center) >= abs(self.obs - center)).mean()
        if direction == "right":
            return (self.null >= self.obs).mean()
        if direction == "left":
            return (self.null <= self.obs).mean()
        raise ValueError("direction must be 'two-sided', 'left', or 'right'")

    def visualize(self, direction="two-sided", bins=50, kde=False, rug=False, figsize=(7, 4)):
        """
        Visualizes the null distribution, observed statistic, and p-value.
        """
        p = self.get_p_value(direction)
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        # Draw the histogram of the null distribution.
        sns.histplot(self.null, bins=bins, stat="density", color="#4C78A8", alpha=0.55, ax=ax)
        if kde: sns.kdeplot(self.null, color="#4C78A8", ax=ax) # Optional density curve
        # Draw a vertical line for the observed statistic.
        ax.axvline(self.obs, color="crimson", ls="--", lw=2)
        # Shade the p-value area.
        _shade(ax, self.null, self.obs, direction)
        # Set the plot title and labels.
        ax.set_title(_format_title(self.meta, self.obs, p))
        ax.set_xlabel("Null Distribution Statistic")
        ax.set_ylabel("Density")
        plt.tight_layout()
        plt.show()
        return p

# --- Example Usage of the Classes ---
# This part of the code shows how to use the 'Infer' and 'InferResult' classes
# to perform different kinds of statistical inference tests.

df = pd.read_csv('data/early_careers_survey.csv')

Example: Two-sample test for the difference in means
Is there a significant difference in the mean X between Males and Females?
res = (
    Infer(df)
      .specify("height ~ gender")  # Set response and predictor
      .hypothesize(null="independence")             # Assume no relationship
      .generate(reps=1000, type="permute")          # Shuffle labels 1000 times
      .calculate(stat="diff in means", order=["Male", "Female"]) # Calculate the difference
)
p_value = res.visualize()  # Visualize the result and get the p-value
print("p-value:", p_value)