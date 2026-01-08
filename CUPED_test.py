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
    import time
    import typing
    return alt, mo, np, pd, plt, time, ttest_ind


@app.cell
def _(np):
    def vectorized_simulate_correlated_data(r, n, tau, mean, sd, rho):
        """
        Generate synthetic A/B test data for multiple replications simultaneously.

        Creates correlated covariate and outcome data for Monte Carlo simulations,
        generating all replications at once for efficient vectorized processing.

        Parameters
        ----------
        r : int
            Number of simulation replications
        n : int
            Sample size per replication (must be even)
        tau : float
            True treatment effect added to treated units
        mean : List[float]
            [mean_x, mean_y] for covariate and outcome distributions
        sd : List[float]
            [sd_x, sd_y] for covariate and outcome standard deviations
        rho : float
            Correlation coefficient between x and y (-1 to 1)

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            x : ndarray, shape (r, n)
                Covariate values for all replications
            y : ndarray, shape (r, n)
                Outcome values for all replications
            t : ndarray, shape (r, n)
                Treatment assignments (0/1) for all replications

        Raises
        ------
        ValueError
            If n is not even, or if parameter lengths/shapes are invalid

        Notes
        -----
        Treatment assignments are randomly shuffled within each replication to ensure
        balance while maintaining independence across replications.
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

        # Generate correlated samples for x, y
        data = np.random.multivariate_normal(mean, cov_matrix, size=(r, n))
        x = data[:, :, 0]
        y = data[:, :, 1]

        # Create a 1D 50/50 treatment assignment array
        t_1d = np.repeat([0, 1], n // 2)
        unshuffled_t = np.tile(t_1d, (r, 1))

        t = np.apply_along_axis(np.random.permutation, 1, unshuffled_t)

        # Add treatment effect
        y = np.where(t == 1, y + tau, y)

        return x, y, t
    return (vectorized_simulate_correlated_data,)


@app.cell
def _(np):
    def vectorized_ate(y, t):
        """
        Calculate average treatment effects for multiple replications simultaneously.

        Computes ATE = mean(y|t=1) - mean(y|t=0) for each replication using
        vectorized operations on treatment-masked arrays.

        Parameters
        ----------
        y : ndarray, shape (r, n)
            Outcome values for r replications of n samples each
        t : ndarray, shape (r, n)
            Treatment assignments (0/1) for r replications of n samples each

        Returns
        -------
        ndarray, shape (r,)
            Average treatment effect estimates for each replication

        Notes
        -----
        Uses boolean masking to efficiently compute group means without explicit
        data splitting, enabling vectorized computation across all replications.
        """
        treated_mask = t == 1
        control_mask = t == 0

        # mean(group) = sum(group) / num in group
        treated_means = np.sum(y * treated_mask, axis=1) / np.sum(treated_mask, axis=1)
        control_means = np.sum(y * control_mask, axis=1) / np.sum(control_mask, axis=1)

        return treated_means - control_means
    return (vectorized_ate,)


@app.cell
def _(np):
    def vectorized_cuped(x, y):
        """
        Apply CUPED adjustment to outcome data for multiple replications.

        Performs covariate adjustment y_cv = y - θ(x - x̄) where θ is the
        regression coefficient estimated from each replication's data.

        Parameters
        ----------
        x : ndarray, shape (r, n)
            Covariate values for r replications of n samples each
        y : ndarray, shape (r, n)
            Outcome values for r replications of n samples each

        Returns
        -------
        ndarray, shape (r, n)
            CUPED-adjusted outcome values y_cv for all replications

        Notes
        -----
        Calculates θ = cov(x,y) / var(x) using sample statistics (ddof=1) for each
        replication independently. Adjustment is applied using vectorized broadcasting
        operations for computational efficiency.
        """
        x_means = np.mean(x, axis=1, keepdims=True)
        y_means = np.mean(y, axis=1, keepdims=True)
        n = x.shape[1]

        cov_x_y = np.sum((x - x_means) * (y - y_means), axis=1) / (n - 1)
        var_x = np.var(x, axis=1, ddof=1)
        theta = cov_x_y / var_x

        # y_cv = y - theta * (x - x_means)
        y_cv = y - theta[:, np.newaxis] * (x - x_means)

        return y_cv
    return (vectorized_cuped,)


@app.cell
def _(
    pd,
    vectorized_ate,
    vectorized_cuped,
    vectorized_simulate_correlated_data,
):
    def vectorized_replicate_ab_test(r, n, tau, mean, sd, rho) -> pd.DataFrame:
        """
        Run Monte Carlo simulation comparing naive vs CUPED ATE estimators.

        Generates synthetic A/B test data for multiple replications and computes
        both naive difference-in-means and CUPED-adjusted treatment effect estimates
        using fully vectorized operations for computational efficiency.

        Parameters
        ----------
        r : int
            Number of simulation replications
        n : int
            Sample size per replication (must be even)
        tau : float
            True treatment effect (added to treated group)
        mean : List[float]
            [mean_x, mean_y] for covariate and outcome distributions
        sd : List[float]
            [sd_x, sd_y] for covariate and outcome standard deviations
        rho : float
            Correlation coefficient between x and y (-1 to 1)

        Returns
        -------
        pd.DataFrame
            DataFrame with columns:
            - 'naive_ate': Naive difference-in-means estimates (r values)
            - 'cuped_ate': CUPED-adjusted estimates (r values)

        Notes
        -----
        Fully vectorized implementation eliminates Python loops over replications,
        providing significant performance improvements over iterative approaches.
        Results can be used to analyze estimator precision and variance reduction.
        """
        # Simulate all data in an (r x n) array
        x, y, t = vectorized_simulate_correlated_data(r, n, tau, mean, sd, rho)

        # Compute naive ates for r replications
        naive_ates = vectorized_ate(y, t)

        # Compute covariate adjusted y
        y_cv = vectorized_cuped(x, y)

        # Compute CUPED ates for r replications
        cuped_ates = vectorized_ate(y_cv, t)

        res = {"naive_ate": naive_ates, "cuped_ate": cuped_ates}

        return pd.DataFrame(res)
    return (vectorized_replicate_ab_test,)


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
            .properties(
                title=f"Sampling Distributions of ATE Naive (SE = {data['naive_ate'].std():.3f}) vs CUPED (SE = {data['cuped_ate'].std():.3f})"
            )
            .configure_title(fontSize=16, anchor="middle")
            .configure_axis(
                grid=False, domain=True
            )  # Remove grid lines, keep axis lines
        )

        return chart
    return (generate_sampling_distribution_altair,)


@app.cell
def _(mo, replicate_ab_test, time, vectorized_replicate_ab_test):
    def speed_benchmark():
        """
        Tests the speed between the unvectorized and vectorized implementations of CUPED
        """
        loop_start = time.time()
        with mo.status.progress_bar(total=2000) as progress_bar:
            replicate_ab_test(
                r=2000,
                n=20000,
                tau=5,
                mean=[0, 0],
                sd=[100, 100],
                rho=0.6,
                progress_bar=progress_bar,
            )
        loop_end = time.time()
        loop_runtime = loop_end - loop_start
        print(f"Time for iterative replication {loop_runtime:.2f}")

        vectorized_start = time.time()
        vectorized_replicate_ab_test(
            r=2000, n=20000, tau=5, mean=[0, 0], sd=[100, 100], rho=0.6
        )
        vectorized_end = time.time()
        vectorized_runtime = vectorized_end - vectorized_start

        print(f"Time for vectorized replication {vectorized_runtime:.2f}")

        print(
            f"% Change in simulation time = {(vectorized_runtime - loop_runtime) / loop_runtime * 100:.2f}"
        )
    return (speed_benchmark,)


@app.cell
def _(mo):
    # Define sliders
    mean_x_slider = mo.ui.slider(-50, 50, value=0, step=5, show_value=True)
    mean_y_slider = mo.ui.slider(-50, 50, value=0, step=5, show_value=True)
    sd_x_slider = mo.ui.slider(10, 200, value=100, step=10, show_value=True)
    sd_y_slider = mo.ui.slider(10, 200, value=100, step=10, show_value=True)
    rho_slider = mo.ui.slider(0.0, 0.9, value=0.6, step=0.05, show_value=True)
    tau_slider = mo.ui.slider(-15.0, 15.0, value=5.0, step=0.5, show_value=True)
    r_slider = mo.ui.slider(100, 2000, value=500, step=100, show_value=True)
    n_slider = mo.ui.slider(500, 20000, value=2000, step=500, show_value=True)

    # Markdown template to replicate old hstack layout using HTML flex
    template = """
    ### Data Parameters

    <div style="display: flex; justify-content: space-around;">
      <div>Mean X: {mean_x}</div>
      <div>Mean Y: {mean_y}</div>
    </div>

    <div style="display: flex; justify-content: space-around;">
      <div>SD X: {sd_x}</div>
      <div>SD Y: {sd_y}</div>
    </div>

    Correlation (X,Y): {rho}

    Treatment Effect (τ): {tau}

    ### Simulation Parameters

    <div style="display: flex; justify-content: space-around;">
      <div>Replications (r): {r}</div>
      <div>Sample Size (n): {n}</div>
    </div>
    """

    # Batch sliders into the template, then wrap in form
    form = (
        mo.md(template)
        .batch(
            mean_x=mean_x_slider,
            mean_y=mean_y_slider,
            sd_x=sd_x_slider,
            sd_y=sd_y_slider,
            rho=rho_slider,
            tau=tau_slider,
            r=r_slider,
            n=n_slider,
        )
        .form(submit_button_label="Run Simulation", bordered=False)
    )
    return (form,)


@app.cell
def _(form):
    # Get form values (dict from batch); use defaults if not submitted yet
    params = form.value or {
        "mean_x": 0,
        "mean_y": 0,
        "sd_x": 100,
        "sd_y": 100,
        "rho": 0.6,
        "tau": 5.0,
        "r": 500,
        "n": 2000,
    }
    mean = [params["mean_x"], params["mean_y"]]
    sd = [params["sd_x"], params["sd_y"]]
    n = params["n"]
    tau = params["tau"]
    rho = params["rho"]
    r = params["r"]
    return mean, n, r, rho, sd, tau


@app.cell
def _(np):
    def compute_single_experiment_results(
        n, tau, mean, sd, rho, run_ttest, simulate_correlated_data, alt, pd, mo
    ):
        np.random.seed(67)
        single_data = simulate_correlated_data(n, tau, mean, sd, rho)
        naive_results = run_ttest("t", "y", single_data, print_results=False)
        theta = np.cov(single_data.x, single_data.y, ddof=1)[0, 1] / np.var(
            single_data.x, ddof=1
        )
        single_data["y_cv"] = single_data.y - theta * (
            single_data.x - single_data.x.mean()
        )
        cuped_results = run_ttest("t", "y_cv", single_data, print_results=False)

        # Table
        table_data = [
            {
                "Method": "Naive",
                "Effect Size": round(naive_results["effect_size"], 3),
                "Std Error": round(naive_results["std_error"], 3),
                "P-Value": round(naive_results["pvalue"], 3),
            },
            {
                "Method": "CUPED",
                "Effect Size": round(cuped_results["effect_size"], 3),
                "Std Error": round(cuped_results["std_error"], 3),
                "P-Value": round(cuped_results["pvalue"], 3),
            },
        ]
        table = mo.ui.table(table_data)

        # Graph
        graph_data = pd.DataFrame(
            {
                "method": ["Naive", "CUPED"],
                "effect": [naive_results["effect_size"], cuped_results["effect_size"]],
                "se": [naive_results["std_error"], cuped_results["std_error"]],
            }
        )
        graph_data["ymin"] = graph_data["effect"] - 1.96 * graph_data["se"]
        graph_data["ymax"] = graph_data["effect"] + 1.96 * graph_data["se"]

        points = (
            alt.Chart(graph_data)
            .mark_circle(size=100)
            .encode(
                x=alt.X("method:N", title="Method"),
                y=alt.Y("effect:Q", title="Effect Size"),
                color=alt.Color(
                    "method:N",
                    scale=alt.Scale(
                        domain=["Naive", "CUPED"], range=["lightcoral", "lightgreen"]
                    ),
                ),
            )
        )
        error_bars = (
            alt.Chart(graph_data)
            .mark_errorbar()
            .encode(
                x="method:N",
                y="ymin:Q",
                y2="ymax:Q",
                color=alt.Color(
                    "method:N",
                    scale=alt.Scale(
                        domain=["Naive", "CUPED"], range=["lightcoral", "lightgreen"]
                    ),
                ),
            )
        )
        rule = (
            alt.Chart()
            .mark_rule(color="black", strokeDash=[5, 5])
            .encode(y=alt.datum(tau))
        )
        graph = (points + error_bars + rule).properties(
            title="Single Experiment Effect Estimates with 95% CI",
            width=400,
            height=300,
        )
        graph_display = mo.ui.altair_chart(graph)

        # Info box
        reduction = (
            (naive_results["std_error"] ** 2 - cuped_results["std_error"] ** 2)
            / naive_results["std_error"] ** 2
            * 100
        )
        info = mo.md(
            f"In this single experiment, CUPED reduced the variance of the effect estimate by {reduction:.1f}%."
        ).callout(kind="info")

        return table, graph_display, info
    return (compute_single_experiment_results,)


@app.cell
def _(
    alt,
    compute_single_experiment_results,
    generate_sampling_distribution_altair,
    mean,
    mo,
    n,
    pd,
    r,
    rho,
    run_ttest,
    sd,
    simulate_correlated_data,
    tau,
    vectorized_replicate_ab_test,
):
    # Generate single experiment data for individual analysis
    data = simulate_correlated_data(n, tau, mean, sd, rho)

    # Generate replication data for sampling distribution analysis
    out_df = vectorized_replicate_ab_test(r, n, tau, mean, sd, rho)

    # print(f"CUPED std: {out_df['cuped_ate'].std():.3f}")

    # Generate comprehensive sampling distribution visualization
    chart = generate_sampling_distribution_altair(out_df)
    altair_display = mo.ui.altair_chart(chart)

    # Calculate variance reduction for display
    naive_std = out_df["naive_ate"].std()
    cuped_std = out_df["cuped_ate"].std()
    variance_reduction = ((naive_std**2 - cuped_std**2) / naive_std**2) * 100

    # Create readable variance reduction text
    variance_text = mo.md(
        f"Over the course of {r} experiments, CUPED reduced the sampling variance by {variance_reduction:.1f}%."
    ).callout(kind="info")

    # Single experiment results
    table, graph_display, info = compute_single_experiment_results(
        n, tau, mean, sd, rho, run_ttest, simulate_correlated_data, alt, pd, mo
    )

    # Tabs
    single_exp_tab = mo.vstack([table, graph_display, info])
    replicated_tab = mo.vstack([altair_display, variance_text])
    results_display = mo.ui.tabs(
        {
            "Single Experiment Results": single_exp_tab,
            "Replicated Results": replicated_tab,
        }
    )
    return (results_display,)


@app.cell
def _(form, mo, results_display):
    # Create complete CUPED tab with parameters and results
    cuped_tab = mo.vstack(
        [mo.md("# CUPED Simulator \n"), mo.hstack([form, results_display])]
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
def _(mo, speed_benchmark):
    # Prevents benchmarking from running
    mo.stop(True)
    speed_benchmark()
    return


@app.cell
def _(mo):
    mo.md("""
    ## Note

    The code below is no longer used because the vectorized implementations were far more efficient
    """)
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
        mean : List[float]
            2-element list [mean_x, mean_y] for covariate and outcome
        sd : List[float]
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
        mean : List[float]
            [mean_x, mean_y] for covariate and outcome distributions
        sd : List[float]
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
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
