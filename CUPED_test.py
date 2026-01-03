import marimo

__generated_with = "0.18.4"
app = marimo.App(layout_file="layouts/CUPED_test.grid.json")


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    from scipy.stats import ttest_ind
    import marimo as mo
    import altair as alt
    return alt, mo, np, pd, plt, ttest_ind


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
def _(np, pd, run_ttest, simulate_correlated_data):
    def replicate_ab_test(
        r: int,
        n: int,
        tau: float,
        mean: list[float],
        sd: list[float],
        rho: float,
        progress_bar,
    ) -> pd.DataFrame:
        """
        Perform Monte Carlo simulation to compare naive vs CUPED ATE estimators.

        Runs multiple A/B test simulations to generate sampling distributions
        of average treatment effect (ATE) estimates, comparing naive difference-in-means
        against CUPED-adjusted estimates.

        Parameters
        ----------
        r : int
            Number of simulation replications
        n : int
            Sample size per experiment (must be even)
        tau : float
            True treatment effect (added to treated group)
        mean : list[float]
            [mean_x, mean_y] for covariate and outcome distributions
        sd : list[float]
            [sd_x, sd_y] for covariate and outcome standard deviations
        rho : float
            Correlation between covariate x and outcome y (-1 to 1)

        Returns
        -------
        pd.DataFrame
            DataFrame with sampling distributions containing:
            - 'naive_ate': Naive difference-in-means estimates across r replications
            - 'cuped_ate': CUPED-adjusted estimates across r replications

        Notes
        -----
        Each replication simulates an A/B test with correlated baseline data,
        applies both naive and CUPED estimation, and collects the results.
        The returned DataFrame can be used to analyze estimator precision
        and variance reduction.

        Examples
        --------
        >>> results = replicate_ab_test(r=1000, n=10000, tau=5.0,
        ...                            mean=[0, 0], sd=[100, 100], rho=0.6)
        >>> print(f"Naive std: {results['naive_ate'].std():.3f}")
        >>> print(f"CUPED std: {results['cuped_ate'].std():.3f}")
        """
        # Initialize result storage
        naive_estimates = []
        cuped_estimates = []

        # Run r simulation replications
        for i in range(r):
            # Generate synthetic A/B test data
            data = simulate_correlated_data(n, tau, mean, sd, rho)

            # Compute naive ATE (simple difference-in-means)
            naive_ate = run_ttest("t", "y", data, print_results=False)["effect_size"]
            naive_estimates.append(naive_ate)

            # Compute CUPED-adjusted ATE: y_cuped = y - θ(x - x̄)
            # where θ = cov(x,y) / var(x) is the regression coefficient
            theta = np.cov(data.x, data.y, ddof=1)[0, 1] / np.var(data.x, ddof=1)
            mean_x = np.mean(data.x)
            data["y_cv"] = data.y - theta * (data.x - mean_x)

            cuped_ate = run_ttest("t", "y_cv", data, print_results=False)["effect_size"]
            cuped_estimates.append(cuped_ate)

            progress_bar.update()

        # Package results as DataFrame for analysis
        res = pd.DataFrame({"naive_ate": naive_estimates, "cuped_ate": cuped_estimates})

        return res
    return (replicate_ab_test,)


@app.cell
def _(pd, plt):
    def generate_sampling_distribution(data: pd.DataFrame) -> dict[str, float]:
        """
        Generate comparative histograms of naive vs CUPED ATE sampling distributions.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame with 'naive_ate' and 'cuped_ate' columns from replicate_ab_test

        Returns
        -------
        dict[str, float]
            Dictionary with standard errors: {'naive_std_error': float, 'cuped_std_error': float}
        """
        # Calculate standard errors (SD of sampling distributions)
        naive_std_error = data["naive_ate"].std()
        cuped_std_error = data["cuped_ate"].std()

        # Calculate variance reduction percentage (matching experimentation literature)
        variance_reduction = (
            (naive_std_error**2 - cuped_std_error**2) / naive_std_error**2
        ) * 100

        # Create side-by-side histograms
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Naive histogram
        ax1.hist(
            data["naive_ate"], bins=50, alpha=0.7, color="lightcoral", edgecolor="black"
        )
        ax1.axvline(
            data["naive_ate"].mean(),
            color="red",
            linestyle="-",
            linewidth=2,
            label=f"Mean: {data['naive_ate'].mean():.2f}",
        )
        ax1.axvline(
            5.0, color="black", linestyle="--", linewidth=2, label="True Effect (τ = 5)"
        )
        ax1.set_title(
            f"Distribution of Naive ATE Estimates\n Standard Error = {naive_std_error:.3f}",
            fontsize=12,
            pad=20,
        )
        ax1.set_xlabel("ATE Estimate", fontsize=11)
        ax1.set_ylabel("Frequency", fontsize=11)
        ax1.legend()

        # CUPED histogram
        ax2.hist(
            data["cuped_ate"], bins=50, alpha=0.7, color="lightgreen", edgecolor="black"
        )
        ax2.axvline(
            data["cuped_ate"].mean(),
            color="green",
            linestyle="-",
            linewidth=2,
            label=f"Mean: {data['cuped_ate'].mean():.2f}",
        )
        ax2.axvline(
            5.0, color="black", linestyle="--", linewidth=2, label="True Effect (τ = 5)"
        )
        ax2.set_title(
            f"Distribution of CUPED ATE Estimates\n Standard Error = {cuped_std_error:.3f}",
            fontsize=12,
            pad=20,
        )
        ax2.set_xlabel("ATE Estimate", fontsize=11)
        ax2.set_ylabel("Frequency", fontsize=11)
        ax2.legend()

        # Overall title
        plt.suptitle(
            "Comparison of Sampling Distributions: Naive vs CUPED",
            fontsize=14,
            y=0.98,
        )
        plt.tight_layout()

        return fig
        # return {"naive_std_error": naive_std_error, "cuped_std_error": cuped_std_error}
    return


@app.cell
def _(alt, pd):
    def generate_sampling_distribution_altair(data: pd.DataFrame):
        """
        Create interactive comparative histograms using Altair.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame with 'naive_ate' and 'cuped_ate' columns

        Returns
        -------
        alt.Chart
            Interactive Altair chart with faceted histograms
        """
        # Melt data from wide to long format for Altair
        melted_data = data.melt(var_name="method", value_name="ate")

        # Create base histogram (no faceting yet)
        base_chart = (
            alt.Chart(melted_data)
            .mark_bar(opacity=0.7)
            .encode(
                x=alt.X("ate:Q", bin=alt.Bin(maxbins=50), title="ATE"),
                y=alt.Y("count():Q", title="Frequency"),
                color=alt.Color(
                    "method:N",
                    scale=alt.Scale(
                        domain=["naive_ate", "cuped_ate"],
                        range=["lightcoral", "lightgreen"],
                    ),
                    title="Method",
                    legend=alt.Legend(
                        labelExpr="datum.label == 'naive_ate' ? 'Naive ATE' : 'CUPED ATE'"
                    ),
                ),
            )
        )

        # Create reference line for true effect (τ = 5.0)
        # Must use same data as base chart for faceting to work
        reference_line = (
            alt.Chart(melted_data)
            .mark_rule(color="black", strokeDash=[5, 5], size=2)
            .encode(x=alt.datum(5.0))
        )

        # Layer first, then apply faceting to the result
        chart = (
            alt.layer(base_chart, reference_line)
            .facet(column=alt.Column("method:N", title=None, header=None))
            .properties(title=f"Sampling Distributions /n Naive (SE = {data['naive_ate'].std():.3f}) vs CUPED (SE = {data['cuped_ate'].std():.3f})")
            .configure_title(fontSize=16, anchor="middle")
            .configure_axis(
                grid=False, domain=True
            )  # Remove grid lines, keep axis lines
        )

        return chart
    return (generate_sampling_distribution_altair,)


@app.cell
def _(mo):
    # Parameter sliders with grid layout

    # Data Parameters section
    data_title = mo.md("### Data Parameters")

    # Header row - equal width columns
    header = mo.hstack(
        [mo.md("**X (covariate)**"), mo.md("**Y (outcome)**")], justify="space-around"
    )

    # Mean row - equal width columns
    mean_x_slider = mo.ui.slider(-50, 50, value=0, step=5, show_value=True)
    mean_y_slider = mo.ui.slider(-50, 50, value=0, step=5, show_value=True)
    mean_row = mo.hstack(
        [
            mo.vstack([mo.md("Mean"), mean_x_slider]),
            mo.vstack([mo.md("Mean"), mean_y_slider]),
        ],
        justify="space-around",
    )

    # SD row - equal width columns
    sd_x_slider = mo.ui.slider(10, 200, value=100, step=10, show_value=True)
    sd_y_slider = mo.ui.slider(10, 200, value=100, step=10, show_value=True)
    sd_row = mo.hstack(
        [mo.vstack([mo.md("SD"), sd_x_slider]), mo.vstack([mo.md("SD"), sd_y_slider])],
        justify="space-around",
    )

    # Single-column parameters
    rho_slider = mo.ui.slider(0.0, 0.9, value=0.6, step=0.05, show_value=True)
    correlation_section = mo.vstack([mo.md("Correlation (X,Y)"), rho_slider])

    tau_slider = mo.ui.slider(-15.0, 15.0, value=5.0, step=0.5, show_value=True)
    treatment_section = mo.vstack([mo.md("Treatment Effect (τ)"), tau_slider])

    # Simulation Parameters section
    sim_title = mo.md("### Simulation Parameters")

    # Simulation row - equal width columns
    r_slider = mo.ui.slider(100, 2000, value=500, step=100, show_value=True)
    n_slider = mo.ui.slider(500, 20000, value=2000, step=500, show_value=True)
    sim_row = mo.hstack(
        [
            mo.vstack([mo.md("Replications (r)"), r_slider]),
            mo.vstack([mo.md("Sample Size (n)"), n_slider]),
        ],
        justify="space-around",
    )

    # Combine all sections with spacing (no horizontal lines)
    left_panel = mo.vstack(
        [
            data_title,
            header,
            mean_row,
            sd_row,
            correlation_section,
            treatment_section,
            sim_title,
            sim_row,
        ],
        gap=1,
    )

    left_panel
    return (
        left_panel,
        mean_x_slider,
        mean_y_slider,
        n_slider,
        r_slider,
        rho_slider,
        sd_x_slider,
        sd_y_slider,
        tau_slider,
    )


@app.cell
def _(
    mean_x_slider,
    mean_y_slider,
    n_slider,
    r_slider,
    rho_slider,
    sd_x_slider,
    sd_y_slider,
    tau_slider,
):
    # set parameters from sliders
    mean = [mean_x_slider.value, mean_y_slider.value]
    sd = [sd_x_slider.value, sd_y_slider.value]
    n = n_slider.value
    tau = tau_slider.value
    rho = rho_slider.value
    r = r_slider.value
    return mean, n, r, rho, sd, tau


@app.cell
def _(left_panel, mo, results_display):
    # Create complete CUPED tab with parameters and results
    cuped_tab = mo.vstack(
        [
            mo.md("# CUPED Simulator"),
            left_panel,
            mo.md("---"),  # Separator
            results_display,
        ]
    )
    return (cuped_tab,)


@app.cell
def _(cuped_tab, mo):
    # Create tabbed interface
    tabs = mo.ui.tabs(
        {
            "CUPED": cuped_tab,
            "Extras": mo.md("Extras tab coming soon..."),  # Placeholder
        }
    )

    tabs
    return


@app.cell
def _(
    generate_sampling_distribution_altair,
    mean,
    mo,
    n,
    np,
    r,
    replicate_ab_test,
    rho,
    run_ttest,
    sd,
    simulate_correlated_data,
    tau,
):
    # Generate single experiment data for individual analysis
    data = simulate_correlated_data(n, tau, mean, sd, rho)

    print("=== Single Experiment Results ===\n")
    print("=== Naive Results ===")
    # Run naive estimate
    run_ttest("t", "y", data)

    # Run CUPED estimate
    theta = np.cov(data.x, data.y, ddof=1)[0, 1] / np.var(data.x, ddof=1)
    mean_x = np.mean(data.x)
    data["y_cv"] = data.y - theta * (data.x - mean_x)
    print("\n=== CUPED Results ===")
    run_ttest("t", "y_cv", data)

    # Generate replication data for sampling distribution analysis
    with mo.status.progress_bar(total=r) as progress_bar:
        out_df = replicate_ab_test(r, n, tau, mean, sd, rho, progress_bar)

    print(f"CUPED std: {out_df['cuped_ate'].std():.3f}")

    # Generate comprehensive sampling distribution visualization
    chart = generate_sampling_distribution_altair(out_df)
    altair_display = mo.ui.altair_chart(chart)

    # Calculate variance reduction for display
    naive_std = out_df["naive_ate"].std()
    cuped_std = out_df["cuped_ate"].std()
    variance_reduction = ((naive_std**2 - cuped_std**2) / naive_std**2) * 100

    # Create readable variance reduction text
    variance_text = mo.md(
        f"CUPED reduced the variance by {variance_reduction:.1f}%."
    ).callout(kind="info")

    # Display results in a clean layout
    results_display = mo.vstack([mo.md("### Results"), altair_display, variance_text])
    results_display
    return (results_display,)


@app.cell
def _():
    # Display the main tabbed interface
    return


if __name__ == "__main__":
    app.run()
