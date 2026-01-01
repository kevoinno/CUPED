import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    from scipy.stats import ttest_ind
    return np, pd, ttest_ind


@app.cell
def _():
    # Simulate n data points

    # Simulate the (X, Y) with correlation rho
    # Make a treatment indicator array T
    # Add treatment effect to Y_i if T_i = 1
    return


@app.cell
def _(np, pd):
    def simulate_correlated_data(
        n: int, tau: float, mean: list[float], sd: list[float], rho: float
    ) -> pd.DataFrame:
        """
        Generate synthetic correlated data for A/B testing simulations.

        Parameters
        ----------
        n : int
            Number of samples to generate (must be even for balanced groups)
        tau : float
            Treatment effect to add to outcome for treated units
        mean : list[float]
            2-element list [mean_x, mean_y] for covariate and outcome
        sd : list[float]
            2-element list [sd_x, sd_y] for standard deviations
        rho : float
            Correlation coefficient between x and y (-1 to 1)

        Returns
        -------
        pd.DataFrame
            DataFrame with 'y' (outcome), 't' (treatment), 'x' (covariate)

        Raises
        ------
        ValueError
            For invalid parameter values with user-friendly messages
        """
        # Input validation with user-friendly messages
        if n % 2 != 0:
            raise ValueError("Sample size must be even for balanced treatment groups")
        if len(mean) != 2 or len(sd) != 2:
            raise ValueError(
                "Mean and standard deviation must each contain exactly 2 values"
            )
        if not all(s > 0 for s in sd):
            raise ValueError("Standard deviations must be positive values")
        if not (-1 <= rho <= 1):
            raise ValueError("Correlation coefficient must be between -1 and 1")

        # Extract parameters
        sd_x, sd_y = sd[0], sd[1]

        # Calculate covariance matrix
        cov_x_y = rho * sd_x * sd_y
        cov_matrix = [[sd_x**2, cov_x_y], [cov_x_y, sd_y**2]]

        # Generate correlated samples
        x, y = np.random.multivariate_normal(mean, cov_matrix, n).T

        # Create 50/50 treatment assignment
        t = np.repeat([0, 1], n // 2)
        np.random.shuffle(t)

        # Apply treatment effect
        y = np.where(t == 1, y + tau, y)

        # Return DataFrame
        return pd.DataFrame({"y": y, "t": t, "x": x})
    return (simulate_correlated_data,)


@app.cell
def _(np):
    np.random.seed(123)
    return


@app.cell
def _():
    n = 10000  # sample size
    tau = 5  # treatment effect
    mean = [0, 0]  # mean vector
    sd = [100, 100]  # standard deviations
    rho = 0.6  # correlation
    return mean, n, rho, sd, tau


@app.cell
def _(mean, n, rho, sd, simulate_correlated_data, tau):
    data = simulate_correlated_data(n, tau, mean, sd, rho)
    return (data,)


@app.cell
def _(np, pd, ttest_ind):
    def run_ttest(
        treatment: str, outcome: str, data: pd.DataFrame, print_results: bool = True
    ) -> dict[str, float]:
        """
        Perform t-test analysis comparing treated vs control groups.

        Parameters
        ----------
        treatment : str
            Column name in data representing the treatment indicator (0/1)
        outcome : str
            Column name in data representing the outcome variable
        data : pd.DataFrame
            DataFrame containing treatment, outcome, and covariate columns
        print_results : bool, default=True
            Whether to print formatted results to console

        Returns
        -------
        dict[str, float]
            Dictionary containing test results with keys:
            - 'effect_size': Average treatment effect
            - 'pvalue': Two-sided p-value
            - 't': t-statistic
            - 'std_error': Standard error of the effect estimate

        Examples
        --------
        >>> results = run_ttest("treatment", "revenue", df)
        >>> print(results['effect_size'])
        2.34
        """
        treated = data[data[treatment] == 1]
        control = data[data[treatment] == 0]

        ttest_res = ttest_ind(treated[outcome], control[outcome], equal_var=True)
        ate = np.mean(treated[outcome]) - np.mean(control[outcome])
        std_error = ate / ttest_res.statistic

        if print_results:
            print(f"Effect size: {ate:.2f}")
            print(f"pvalue : {ttest_res.pvalue:.2f}")
            print(f"t-statistic : {ttest_res.statistic:.2f}")
            print(f"std error : {std_error:.2f}")

        res = {
            "effect_size": ate,
            "pvalue": ttest_res.pvalue,
            "t": ttest_res.statistic,
            "std_error": std_error,
        }

        return res
    return (run_ttest,)


@app.cell
def _(data, run_ttest):
    # Naive estimate
    _, __, ___, ____ = run_ttest('t', 'y', data)
    return


@app.cell
def _(data, np, run_ttest):
    theta = np.cov(data.x, data.y, ddof=1)[0, 1] / np.var(data.x, ddof=1)
    mean_x = np.mean(data.x)
    data['y_cv'] = data.y - theta * (data.x - mean_x)
    _, __, ____1, _____1 = run_ttest('t', 'y_cv', data)
    return


if __name__ == "__main__":
    app.run()
