import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full")


@app.cell
def _():
    import time
    import marimo as mo
    import numpy as np
    import pandas as pd
    import altair as alt
    import statsmodels.formula.api as smf
    from scipy.stats import ttest_ind

    return alt, mo, np, pd, smf, time, ttest_ind


@app.cell
def _(np, pd):
    # ==========================================================================
    # SIMULATION FUNCTIONS
    # ==========================================================================

    def simulate_correlated_data(
        n: int, tau: float, mean: list[float], sd: list[float], rho: float
    ) -> pd.DataFrame:
        """
        Generate synthetic correlated data for a single A/B test.

        Parameters
        ----------
        n : int
            Number of samples (must be even for balanced groups)
        tau : float
            Treatment effect to add to treated units
        mean : list[float]
            [mean_x, mean_y] for covariate and outcome
        sd : list[float]
            [sd_x, sd_y] for standard deviations
        rho : float
            Correlation coefficient between x and y (-1 to 1)

        Returns
        -------
        pd.DataFrame
            DataFrame with 'y' (outcome), 't' (treatment), 'x' (covariate)
        """
        if n % 2 != 0:
            raise ValueError("Sample size must be even for balanced treatment groups")
        if not (-1 <= rho <= 1):
            raise ValueError("Correlation coefficient must be between -1 and 1")

        sd_x, sd_y = sd[0], sd[1]
        cov_x_y = rho * sd_x * sd_y
        cov_matrix = [[sd_x**2, cov_x_y], [cov_x_y, sd_y**2]]

        x, y = np.random.multivariate_normal(mean, cov_matrix, n).T
        t = np.repeat([0, 1], n // 2)
        np.random.shuffle(t)
        y = np.where(t == 1, y + tau, y)

        return pd.DataFrame({"y": y, "t": t, "x": x})

    def vectorized_simulate_correlated_data(r, n, tau, mean, sd, rho):
        """
        Generate synthetic A/B test data for multiple replications simultaneously.

        Parameters
        ----------
        r : int
            Number of simulation replications
        n : int
            Sample size per replication (must be even)
        tau : float
            True treatment effect
        mean : list[float]
            [mean_x, mean_y] for distributions
        sd : list[float]
            [sd_x, sd_y] for standard deviations
        rho : float
            Correlation coefficient between x and y

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            x, y, t arrays each with shape (r, n)
        """
        if n % 2 != 0:
            raise ValueError("Sample size must be even for balanced treatment groups")

        sd_x, sd_y = sd[0], sd[1]
        cov_x_y = rho * sd_x * sd_y
        cov_matrix = [[sd_x**2, cov_x_y], [cov_x_y, sd_y**2]]

        data = np.random.multivariate_normal(mean, cov_matrix, size=(r, n))
        x = data[:, :, 0]
        y = data[:, :, 1]

        t_1d = np.repeat([0, 1], n // 2)
        unshuffled_t = np.tile(t_1d, (r, 1))
        t = np.apply_along_axis(np.random.permutation, 1, unshuffled_t)

        y = np.where(t == 1, y + tau, y)

        return x, y, t

    def vectorized_ate(y, t):
        """
        Calculate average treatment effects for multiple replications.

        Parameters
        ----------
        y : np.ndarray
            Outcome values, shape (r, n)
        t : np.ndarray
            Treatment assignments, shape (r, n)

        Returns
        -------
        np.ndarray
            ATE estimates, shape (r,)
        """
        treated_mask = t == 1
        control_mask = t == 0

        treated_means = np.sum(y * treated_mask, axis=1) / np.sum(treated_mask, axis=1)
        control_means = np.sum(y * control_mask, axis=1) / np.sum(control_mask, axis=1)

        return treated_means - control_means

    def vectorized_cuped(x, y):
        """
        Apply CUPED adjustment: y_cv = y - theta * (x - x_mean).

        Parameters
        ----------
        x : np.ndarray
            Covariate values, shape (r, n)
        y : np.ndarray
            Outcome values, shape (r, n)

        Returns
        -------
        np.ndarray
            CUPED-adjusted outcomes, shape (r, n)
        """
        x_means = np.mean(x, axis=1, keepdims=True)
        y_means = np.mean(y, axis=1, keepdims=True)
        n = x.shape[1]

        cov_x_y = np.sum((x - x_means) * (y - y_means), axis=1) / (n - 1)
        var_x = np.var(x, axis=1, ddof=1)
        theta = cov_x_y / var_x

        y_cv = y - theta[:, np.newaxis] * (x - x_means)
        return y_cv

    def vectorized_replicate_ab_test(r, n, tau, mean, sd, rho) -> pd.DataFrame:
        """
        Run Monte Carlo simulation comparing naive vs CUPED estimators.

        Returns
        -------
        pd.DataFrame
            DataFrame with 'naive_ate' and 'cuped_ate' columns
        """
        x, y, t = vectorized_simulate_correlated_data(r, n, tau, mean, sd, rho)
        naive_ates = vectorized_ate(y, t)
        y_cv = vectorized_cuped(x, y)
        cuped_ates = vectorized_ate(y_cv, t)

        return pd.DataFrame({"naive_ate": naive_ates, "cuped_ate": cuped_ates})

    return simulate_correlated_data, vectorized_replicate_ab_test


@app.cell
def _(mo):
    # ==========================================================================
    # CONTROLS PANEL
    # ==========================================================================

    # Create individual sliders
    mean_x_slider = mo.ui.slider(-50, 50, value=0, step=5, show_value=True)
    mean_y_slider = mo.ui.slider(-50, 50, value=0, step=5, show_value=True)
    sd_x_slider = mo.ui.slider(10, 200, value=100, step=10, show_value=True)
    sd_y_slider = mo.ui.slider(10, 200, value=100, step=10, show_value=True)
    rho_slider = mo.ui.slider(0.0, 0.9, value=0.6, step=0.05, show_value=True)
    tau_slider = mo.ui.slider(-15.0, 15.0, value=5.0, step=0.5, show_value=True)
    n_slider = mo.ui.slider(500, 20000, value=2000, step=500, show_value=True)
    r_slider = mo.ui.slider(100, 2000, value=500, step=100, show_value=True)

    # Slider column only; the collapsible wrapper lives in the main layout cell
    controls_panel = mo.vstack([
        mo.md("**Covariate (X)**"),
        mo.hstack([
            mo.vstack([mo.md("Mean"), mean_x_slider]),
            mo.vstack([mo.md("Std Dev"), sd_x_slider]),
        ], gap=2),
        mo.md("**Outcome (Y)**"),
        mo.hstack([
            mo.vstack([mo.md("Mean"), mean_y_slider]),
            mo.vstack([mo.md("Std Dev"), sd_y_slider]),
        ], gap=2),
        mo.vstack([mo.md("**Correlation (X, Y)**"), rho_slider]),
        mo.vstack([mo.md("**Treatment Effect (τ)**"), tau_slider]),
        mo.md("---"),
        mo.md("### Simulation Settings"),
        mo.vstack([mo.md("**Sample Size (n)**"), n_slider]),
        mo.vstack([mo.md("**Replications (r)**"), r_slider]),
    ])
    return (
        controls_panel,
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
    # Extract slider values for downstream cells
    mean_x = mean_x_slider.value
    mean_y = mean_y_slider.value
    sd_x = sd_x_slider.value
    sd_y = sd_y_slider.value
    rho = rho_slider.value
    tau = tau_slider.value
    n = n_slider.value
    r = r_slider.value

    mean = [mean_x, mean_y]
    sd = [sd_x, sd_y]
    return mean, n, r, rho, sd, tau


@app.cell
def _(
    mean,
    n,
    np,
    r,
    rho,
    sd,
    simulate_correlated_data,
    tau,
    ttest_ind,
    vectorized_replicate_ab_test,
):
    # ==========================================================================
    # DATA GENERATION
    # ==========================================================================

    # Set seed for reproducibility in single experiment
    np.random.seed(42)

    # Generate single experiment data (used by Tab 1 and Tab 3)
    single_exp_data = simulate_correlated_data(n, tau, mean, sd, rho)

    # Compute naive results for single experiment
    treated = single_exp_data[single_exp_data["t"] == 1]
    control = single_exp_data[single_exp_data["t"] == 0]
    naive_ttest = ttest_ind(treated["y"], control["y"], equal_var=True)
    naive_effect = np.mean(treated["y"]) - np.mean(control["y"])
    naive_se = naive_effect / naive_ttest.statistic
    naive_ci_low = naive_effect - 1.96 * naive_se
    naive_ci_high = naive_effect + 1.96 * naive_se

    # Compute CUPED results for single experiment
    theta = np.cov(single_exp_data["x"], single_exp_data["y"], ddof=1)[0, 1] / np.var(
        single_exp_data["x"], ddof=1
    )
    single_exp_data["y_cv"] = single_exp_data["y"] - theta * (
        single_exp_data["x"] - single_exp_data["x"].mean()
    )

    treated_cv = single_exp_data[single_exp_data["t"] == 1]
    control_cv = single_exp_data[single_exp_data["t"] == 0]
    cuped_ttest = ttest_ind(treated_cv["y_cv"], control_cv["y_cv"], equal_var=True)
    cuped_effect = np.mean(treated_cv["y_cv"]) - np.mean(control_cv["y_cv"])
    cuped_se = cuped_effect / cuped_ttest.statistic
    cuped_ci_low = cuped_effect - 1.96 * cuped_se
    cuped_ci_high = cuped_effect + 1.96 * cuped_se

    # Run replicated simulation for Tab 2
    replicated_results = vectorized_replicate_ab_test(r, n, tau, mean, sd, rho)

    # Calculate variance reduction
    naive_sampling_se = replicated_results["naive_ate"].std()
    cuped_sampling_se = replicated_results["cuped_ate"].std()
    variance_reduction = (
        (naive_sampling_se**2 - cuped_sampling_se**2) / naive_sampling_se**2
    ) * 100
    return (
        cuped_ci_high,
        cuped_ci_low,
        cuped_effect,
        cuped_sampling_se,
        cuped_se,
        cuped_ttest,
        naive_ci_high,
        naive_ci_low,
        naive_effect,
        naive_sampling_se,
        naive_se,
        naive_ttest,
        replicated_results,
        single_exp_data,
        variance_reduction,
    )


@app.cell
def _(
    alt,
    cuped_ci_high,
    cuped_ci_low,
    cuped_effect,
    cuped_se,
    cuped_ttest,
    mo,
    naive_ci_high,
    naive_ci_low,
    naive_effect,
    naive_se,
    naive_ttest,
    pd,
    tau,
):
    # ==========================================================================
    # TAB 1: SINGLE EXPERIMENT
    # ==========================================================================

    # Extract p-values
    naive_pvalue = naive_ttest.pvalue
    cuped_pvalue = cuped_ttest.pvalue

    # Scorecard data
    scorecard_html = f"""
    <div style="display: flex; gap: 2rem; margin-bottom: 1.5rem;">
         <div style="flex: 1; padding: 1rem; background: #f0fdf4; border-radius: 8px; border-left: 4px solid #10b981;">
            <div style="color: #64748b; font-size: 0.875rem; margin-bottom: 0.25rem;">CUPED Estimate</div>
            <div style="font-size: 1.5rem; font-weight: 600; color: #1e293b;">{cuped_effect:.3f}</div>
            <div style="color: #64748b; font-size: 0.75rem;">95% CI: [{cuped_ci_low:.3f}, {cuped_ci_high:.3f}]</div>
            <div style="color: #64748b; font-size: 0.75rem;">Std. error: {cuped_se:.3f}</div>
            <div style="color: #64748b; font-size: 0.75rem;">p-value: {cuped_pvalue:.4f}</div>
        </div>
        <div style="flex: 1; padding: 1rem; background: #f1f5f9; border-radius: 8px; border-left: 4px solid #6366f1;">
            <div style="color: #64748b; font-size: 0.875rem; margin-bottom: 0.25rem;">Naive Estimate</div>
            <div style="font-size: 1.5rem; font-weight: 600; color: #1e293b;">{naive_effect:.3f}</div>
            <div style="color: #64748b; font-size: 0.75rem;">95% CI: [{naive_ci_low:.3f}, {naive_ci_high:.3f}]</div>
            <div style="color: #64748b; font-size: 0.75rem;">Std. error: {naive_se:.3f}</div>
            <div style="color: #64748b; font-size: 0.75rem;">p-value: {naive_pvalue:.4f}</div>
        </div>
        <div style="flex: 1; padding: 1rem; background: #fafafa; border-radius: 8px; border-left: 4px solid #1e293b;">
            <div style="color: #64748b; font-size: 0.875rem; margin-bottom: 0.25rem;">True Effect (τ)</div>
            <div style="font-size: 1.5rem; font-weight: 600; color: #1e293b;">{tau:.1f}</div>
        </div>
    </div>
    """
    tab1_scorecard = mo.Html(scorecard_html)

    # Interval plot data
    interval_data = pd.DataFrame({
        "method": ["Naive", "CUPED"],
        "effect": [naive_effect, cuped_effect],
        "ci_low": [naive_ci_low, cuped_ci_low],
        "ci_high": [naive_ci_high, cuped_ci_high],
    })

    # Points
    points = (
        alt.Chart(interval_data)
        .mark_circle(size=120)
        .encode(
            x=alt.X("method:N", title="Estimation Method", axis=alt.Axis(labelFontSize=12, titleFontSize=14)),
            y=alt.Y("effect:Q", title="Treatment Effect Estimate", axis=alt.Axis(labelFontSize=12, titleFontSize=14)),
            color=alt.Color(
                "method:N",
                scale=alt.Scale(domain=["Naive", "CUPED"], range=["#6366f1", "#10b981"]),
                legend=alt.Legend(title="Method"),
            ),
        )
    )

    # Error bars
    error_bars = (
        alt.Chart(interval_data)
        .mark_errorbar()
        .encode(
            x="method:N",
            y=alt.Y("ci_low:Q", title=""),
            y2="ci_high:Q",
            color=alt.Color(
                "method:N",
                scale=alt.Scale(domain=["Naive", "CUPED"], range=["#6366f1", "#10b981"]),
                legend=None,
            ),
        )
    )

    # True effect reference line
    true_effect_data = pd.DataFrame({"label": ["True Effect (τ)"], "value": [tau]})
    true_effect_rule = (
        alt.Chart(true_effect_data)
        .mark_rule(strokeDash=[5, 5], size=2)
        .encode(
            y="value:Q",
            color=alt.Color(
                "label:N",
                scale=alt.Scale(domain=["True Effect (τ)"], range=["#1e293b"]),
                legend=alt.Legend(title=None),
            ),
        )
    )

    # Combine into final chart
    tab1_chart = (
        (points + error_bars + true_effect_rule)
        .resolve_scale(color="independent")
        .properties(
            title="Single Experiment: Treatment Effect Estimates with 95% CI",
            width=500,
            height=350,
        )
        .configure_title(fontSize=16, anchor="middle")
    )

    # CI % width reduction callout
    naive_width = naive_ci_high - naive_ci_low
    cuped_width = cuped_ci_high - cuped_ci_low
    ci_reduction_pct = ((naive_width - cuped_width) / naive_width) * 100
    ci_callout = mo.md(
        f"**CI Width Reduction:** CUPED reduced the 95% confidence interval width by **{ci_reduction_pct:.1f}%** (from {naive_width:.3f} to {cuped_width:.3f})."
    ).callout(kind="success")

    # Tab 1 content
    tab1_content = mo.vstack([
        tab1_scorecard,
        mo.ui.altair_chart(tab1_chart),
        ci_callout,
    ])
    return (tab1_content,)


@app.cell
def _(
    alt,
    cuped_sampling_se,
    mo,
    naive_sampling_se,
    pd,
    replicated_results,
    tau,
    variance_reduction,
):
    # ==========================================================================
    # TAB 2: REPLICATED POWER
    # ==========================================================================

    # Melt data for Altair
    melted_results = replicated_results.melt(var_name="method", value_name="ate")
    melted_results["method_label"] = melted_results["method"].map({
        "naive_ate": "Naive",
        "cuped_ate": "CUPED"
    })

    # Layered histogram
    histogram = (
        alt.Chart(melted_results)
        .mark_bar(opacity=0.6)
        .encode(
            x=alt.X("ate:Q", bin=alt.Bin(maxbins=50), title="Treatment Effect Estimate"),
            y=alt.Y("count():Q", title="Frequency"),
            color=alt.Color(
                "method_label:N",
                scale=alt.Scale(domain=["Naive", "CUPED"], range=["#6366f1", "#10b981"]),
                legend=alt.Legend(title="Method"),
            ),
        )
    )

    # True effect reference line
    true_effect_line = (
        alt.Chart(pd.DataFrame({"x": [tau]}))
        .mark_rule(color="#1e293b", strokeDash=[5, 5], size=2)
        .encode(x="x:Q")
    )

    # Combined chart
    tab2_chart = (
        alt.layer(histogram, true_effect_line)
        .properties(
            title=f"Sampling Distributions: Naive (SE={naive_sampling_se:.3f}) vs CUPED (SE={cuped_sampling_se:.3f})",
            width=700,
            height=400,
        )
        .configure_title(fontSize=16, anchor="middle")
    )

    # Calculate time savings (variance reduction = time savings percentage)
    time_savings_pct = variance_reduction
    # Example: if a baseline experiment takes 14 days
    baseline_days = 14
    reduced_days = baseline_days * (1 - time_savings_pct / 100)

    # Business impact callout
    business_callout = mo.md(f"""
    **Why variance reduction matters:** On average, you could have cut down the time your A/B test or experiment takes by **{time_savings_pct:.1f}%** with CUPED. We get faster decision making with the same power and type 1 error rate.

    For example, a **{baseline_days}-day experiment** could conclude in just **{reduced_days:.1f} days**.
    """).callout(kind="success")

    # Explanation of time savings calculation
    time_explanation = mo.vstack([
        mo.md(f"""
        **How is time savings calculated?**

        The variance reduction was {time_savings_pct:.1f}%.
        """),
        mo.md(
        r'''
        The sample size of each group n is proportional to the variance of our outcome metric Y. In math, this is:

        \[
        n \propto Var(Y)
        \]

        But after applying CUPED, we are working with the covariate adjusted outcome $Y_{cv}$

        \[
        n \propto Var(Y_{cv})
        \]
        '''),
        mo.md(f"""
         Remember, the variance of the covariate adjusted outcome is {time_savings_pct:.1f}% smaller than the original outcome. That means our required sample size will also be  {time_savings_pct:.1f}% smaller. This also means the time we need to wait for the experiment to run is reduced by {time_savings_pct:.1f}% because we need less samples to achieve the same power and type 1 error rate.
        """)
    ])

    # Tab 2 content
    tab2_content = mo.vstack([
        mo.ui.altair_chart(tab2_chart),
        business_callout,
        time_explanation,
    ])
    return (tab2_content,)


@app.cell
def _(cuped_effect, cuped_se, cuped_ttest, mo, pd, single_exp_data, smf):
    # ==========================================================================
    # TAB 3: TECHNICAL DEEP DIVE (LIN ESTIMATOR)
    # ==========================================================================

    # Center the covariate
    lin_data = single_exp_data.copy()
    lin_data["x_centered"] = lin_data["x"] - lin_data["x"].mean()

    # Run OLS with interaction term: y ~ t * x_centered
    lin_model = smf.ols("y ~ t * x_centered", data=lin_data).fit()
    lin_estimate = lin_model.params["t"]
    lin_se = lin_model.bse["t"]
    lin_pvalue = lin_model.pvalues["t"]

    # Get CUPED p-value from ttest
    _cuped_pvalue = cuped_ttest.pvalue

    # Comparison table
    comparison_data = pd.DataFrame({
        "Estimator": ["CUPED", "Lin Regression"],
        "Point Estimate": [f"{cuped_effect:.4f}", f"{lin_estimate:.4f}"],
        "Standard Error": [f"{cuped_se:.4f}", f"{lin_se:.4f}"],
        "P-Value": [f"{_cuped_pvalue:.4f}", f"{lin_pvalue:.4f}"],
    })
    comparison_table = mo.ui.table(comparison_data, selection=None)

    # Explanation
    lin_explanation = mo.md("""
    ### Lin Estimator (Regression Adjustment)

    The results from CUPED can also be achieved through regression. The Lin estimator (Lin, 2013) adjusts for covariates using OLS regression with an interaction term:

    ```
    y ~ treatment + x_centered + treatment * x_centered
    ```

    **CUPED is asymptotically equivalent to the Lin estimator.** Both methods:
    - Reduce variance by leveraging pre-experiment covariate information
    - Produce unbiased estimates of the average treatment effect
    - Achieve the same variance reduction in large samples

    The coefficient on the `treatment` term in the Lin regression should closely match the CUPED estimate shown above.

    Reference: Lin, W. (2013). "Agnostic notes on regression adjustments to experimental data: Reexamining Freedman's critique." *The Annals of Applied Statistics*.
    """)

    # Tab 3 content
    tab3_content = mo.vstack([
        mo.md("### Comparison: CUPED vs Lin Estimator"),
        comparison_table,
        lin_explanation,
    ])
    return (tab3_content,)


@app.cell
def _(mo):
    # ==========================================================================
    # GLOSSARY (COLLAPSIBLE ACCORDION)
    # ==========================================================================

    glossary = mo.accordion({
        "Standard Error (SE)": mo.md("""
    The standard error measures the standard deviation of an estimate. A smaller SE means the estimate is more precise and reliable. In A/B testing, lower SE means faster experiments.
        """),
        "Covariate": mo.md("""
    A covariate is a variable that we have data for, but are not primarily interested in. In CUPED, we use pre-experiment data (like last week's metrics) as a covariate. This data was tracked, but we are primarily interested in the treatment and outcome variable.
        """),
        "CUPED (Controlled-experiment Using Pre-Experiment Data)": mo.md("""
    CUPED is a variance reduction technique developed at Microsoft. It adjusts the outcome variable using pre-experiment covariates to produce more precise treatment effect estimates without increasing sample size.
        """),
        "Variance Reduction": mo.md("""
    Variance reduction refers to techniques that make estimates more precise. With CUPED, we can achieve the same statistical power with fewer samples, leading to faster experiments. Visually, reducing variance makes confidence intervals narrower.
        """),
        "Lin Estimator": mo.md("""
    The Lin estimator is a regression-based approach to covariate adjustment proposed by Winston Lin in 2013. It is asymptotically equivalent to CUPED and provides theoretical justification for the method.
        """),
    })
    return (glossary,)


@app.cell
def _(np, pd, simulate_correlated_data):
    # ==========================================================================
    # CUPED VISUALIZED - three-step transformation frames
    # ==========================================================================

    # Fixed data on purpose: this figure teaches the mechanism and must not move
    # when someone explores the simulator's sliders further down the page.
    #
    # mean_x = 80 is load-bearing. Step 1 -> step 2 is a lift of exactly
    # theta * mean(X), so with the covariate centred on zero that step would be a
    # no-op. At 80 the lift is about 1.7x the cloud's own spread -- clearly visible.
    #
    # tau = 0 because this figure never shows the treatment split. A non-zero
    # effect would band the cloud in two with nothing on screen to explain it.
    np.random.seed(7)
    viz_data = simulate_correlated_data(150, 0, [80, 60], [40, 40], 0.80)

    # Same estimator the simulator uses for theta
    viz_theta = float(
        np.cov(viz_data["x"], viz_data["y"], ddof=1)[0, 1]
        / np.var(viz_data["x"], ddof=1)
    )
    viz_xbar = float(viz_data["x"].mean())
    viz_ybar = float(viz_data["y"].mean())

    # The three states, stacked long-form; the chart filters on `step`.
    # Step 2 differs from step 1 by the constant theta * mean(X), so it moves the
    # cloud without touching its spread -- that is the point of the last frame.
    viz_stage_labels = ["Raw outcome (Y)", "Y − θX", "Y − θX + θX̄"]
    viz_frames = pd.concat(
        [
            pd.DataFrame({
                "x": viz_data["x"].values,
                "value": _vals.values,
                "step": _k,
                "stage": viz_stage_labels[_k],
            })
            for _k, _vals in enumerate([
                viz_data["y"],
                viz_data["y"] - viz_theta * viz_data["x"],
                viz_data["y"] - viz_theta * viz_data["x"] + viz_theta * viz_xbar,
            ])
        ],
        ignore_index=True,
    )

    # Pinned across all three frames. Let any one of them autoscale and both the
    # flatten and the lift normalise away, which is the entire figure.
    _lo, _hi = viz_frames["value"].min(), viz_frames["value"].max()
    _pad = (_hi - _lo) * 0.06
    viz_y_domain = [float(_lo - _pad), float(_hi + _pad)]

    _xlo, _xhi = viz_data["x"].min(), viz_data["x"].max()
    _xpad = (_xhi - _xlo) * 0.05
    viz_x_domain = [float(_xlo - _xpad), float(_xhi + _xpad)]
    return (
        viz_frames,
        viz_stage_labels,
        viz_theta,
        viz_x_domain,
        viz_xbar,
        viz_y_domain,
        viz_ybar,
    )


@app.cell
def _(mo):
    # ==========================================================================
    # CUPED VISUALIZED - animation ticker
    # ==========================================================================

    # In its own cell on purpose: a UI element recreated inside the cell it drives
    # would reset on every tick.
    #
    # The interval is deliberately SHORTER than the 1.5s frame period in the next
    # cell. If they matched, clock jitter could advance the frame index by two and
    # silently skip a step -- the one failure this figure cannot tolerate.
    viz_ticker = mo.ui.refresh(default_interval="1s", label="")
    return (viz_ticker,)


@app.cell
def _(
    alt,
    mo,
    pd,
    time,
    viz_frames,
    viz_stage_labels,
    viz_theta,
    viz_ticker,
    viz_x_domain,
    viz_xbar,
    viz_y_domain,
    viz_ybar,
):
    # ==========================================================================
    # CUPED VISUALIZED - the looping animation
    # ==========================================================================

    viz_ticker  # dependency: this cell re-runs on every tick of the refresh element

    # Frame period (2s) is twice the ticker interval (1s), so every frame is caught
    # at least once and the index cannot jump by two. Simulated over 200k ticks:
    # at a 1.5s period a skip shows up roughly once in 100k, at 2s never.
    viz_step = int(time.monotonic() / 2.0) % 3


    def viz_build_chart(step):
        """Build the chart for one transformation step.

        A function rather than inline code so that `scripts/export_steps.py` can
        pull it out through marimo's `app.run()` and render the same three frames
        to disk. The exported PNGs then cannot drift from what the notebook shows.
        """
        frame = viz_frames[viz_frames["step"] == step]
        y_scale = alt.Scale(domain=viz_y_domain, nice=False)
        x_scale = alt.Scale(domain=viz_x_domain, nice=False)

        points = (
            alt.Chart(frame)
            .mark_circle(size=60, opacity=0.6)
            .encode(
                x=alt.X("x:Q", title="Pre-experiment covariate (X)", scale=x_scale),
                y=alt.Y("value:Q", title="Outcome (Y)", scale=y_scale),
                color=alt.Color(
                    "stage:N",
                    scale=alt.Scale(
                        domain=viz_stage_labels, range=["#6366f1", "#f59e0b", "#10b981"]
                    ),
                    legend=None,
                ),
                tooltip=[
                    alt.Tooltip("x:Q", title="Covariate", format=".1f"),
                    alt.Tooltip("value:Q", title="Value", format=".1f"),
                    alt.Tooltip("stage:N", title="Stage"),
                ],
            )
        )

        # Mean of Y, dashed. Step 0 straddles it tilted, step 1 sits well below it,
        # step 2 lands flat on it -- which is what "restored to scale" means.
        mean_rule = (
            alt.Chart(pd.DataFrame({"y": [viz_ybar]}))
            .mark_rule(color="#1e293b", strokeDash=[6, 4], strokeWidth=2)
            .encode(y=alt.Y("y:Q", scale=y_scale))
        )

        # Slope theta on the raw frame, flat on the other two. Makes "the cloud
        # flattens" something you watch rather than something you infer.
        slope = viz_theta if step == 0 else 0.0
        intercept = float(frame["value"].mean()) - slope * viz_xbar
        fit = (
            alt.Chart(pd.DataFrame({"x": viz_x_domain}))
            .transform_calculate(yl=f"{intercept} + {slope} * datum.x")
            .mark_line(color="#1e293b", strokeWidth=2, opacity=0.8)
            .encode(x=alt.X("x:Q", scale=x_scale), y=alt.Y("yl:Q", scale=y_scale))
        )

        return alt.layer(mean_rule, fit, points).properties(
            width=620,
            height=360,
            title=f"Step {step} of 2  ·  {viz_stage_labels[step]}",
        )


    viz_chart = viz_build_chart(viz_step)

    # ---- commentary ----------------------------------------------------------
    # Deliberately static: all three steps are described at once rather than
    # swapping with the frame. Text that changes underneath a changing picture is
    # hard to read, and a reader who missed a frame would have to wait a full cycle.
    #
    # Inline maths uses literal theta / X-bar characters rather than LaTeX, to match
    # the chart's own stage labels and keep the sentences readable in the source.
    # The one display formula below is LaTeX, which marimo renders with KaTeX.

    _sd_raw = float(viz_frames.loc[viz_frames.step == 0, "value"].std(ddof=1))
    _sd_flat = float(viz_frames.loc[viz_frames.step == 1, "value"].std(ddof=1))
    _lift = viz_theta * viz_xbar

    cuped_viz = mo.vstack([
        mo.md("## CUPED Visualized"),
        mo.md(
            "Some of the noise in an experiment's outcome is predictable from data you "
            "already had before it started. CUPED removes that predictable part and keeps "
            "the rest. The three frames below cycle on their own."
        ),
        viz_ticker,
        viz_chart,
        mo.md(
            f"""
    **Step 0 · Raw outcome (Y).** The outcome climbs with the covariate — the solid line
    is the best fit, slope θ = {viz_theta:.2f}, and the dashed line is the average
    outcome. That tilt is variation you could have predicted *before the experiment
    started*: it is about who these users already were, not about any treatment. Spread:
    **{_sd_raw:.1f}**.

    **Step 1 · Y − θX.** Subtract θX from every point and the slope disappears — the
    cloud goes flat and its spread falls to **{_sd_flat:.1f}**. That drop is the variance
    reduction, and it is the whole reason to do this. But everything has also fallen by
    θX̄ = {_lift:.1f}, far below the dashed average, so the numbers no longer read as
    outcomes.

    **Step 2 · Y − θX + θX̄.** Add θX̄ back. Every point lifts by the same constant, so
    the spread is untouched — still **{_sd_flat:.1f}** — while the cloud settles flat on
    the average outcome. Same scale as step 0, far less scatter.

    \\[
    \\hat{{Y}}_{{cv}} = Y_i - \\theta X_i + \\theta\\,E[X]
    \\]

    The slope removal bought the precision; the mean addition bought back the units.
    """
        ),
    ])
    return (cuped_viz,)


@app.cell
def _(
    controls_panel,
    cuped_viz,
    glossary,
    mo,
    tab1_content,
    tab2_content,
    tab3_content,
):
    # ==========================================================================
    # MAIN LAYOUT
    # ==========================================================================

    # Header
    header = mo.md("""
    # CUPED Experimentation Simulator

    **CUPED** (Controlled-experiment Using Pre-Experiment Data) is a variance reduction technique that makes A/B tests faster and more sensitive. By adjusting for pre-experiment covariates, CUPED can detect the same treatment effects with fewer samples or find smaller effects with the same sample size. The more correlated the pre-experiment data is with the outcome metric, the more the variance will be reduced!
    """)

    # Result tabs (without Technical Deep Dive)
    result_tabs = mo.ui.tabs({
        "Single Experiment": tab1_content,
        "Replicated Results": tab2_content,
    })

    # Practical Considerations accordion
    practical_considerations = mo.accordion({
        "Practical Considerations": mo.md("""
    **Choosing a good covariate:**

    The pre-experiment value of the outcome metric is usually an excellent covariate for CUPED. For example, if you're measuring revenue per user, using each user's revenue from the week before the experiment started tends to be highly correlated with their revenue during the experiment.

    Reference: Kohavi, Ron & Deng, Alex & Xu, Ya & Walker, Toby. (2013). Improving the Sensitivity of Online Controlled Experiments by Utilizing Pre-Experiment Data. 10.1145/2433396.2433413. 

    **Avoiding bias:**

    Be careful not to use covariates that could be affected by the treatment. The covariate must be measured *before* treatment assignment or be otherwise unaffected by it. Using a post-treatment variable as a covariate introduces bias in the causal estimate. 
        """),
    })

    # Advanced Info dropdown (contains Technical Deep Dive content)
    advanced_info = mo.accordion({
        "Advanced Info": tab3_content,
    })

    # Right side content (tabs only)
    right_content = mo.vstack([
        result_tabs,
    ])

    # Controls and results together in one collapsible block
    simulator = mo.accordion({
        "How CUPED Tightens Confidence Intervals": mo.hstack([
            controls_panel,
            right_content,
        ], gap=4, align="start"),
    }, expanded=True)

    # Main layout: collapsible simulator, then practical considerations, advanced info, and glossary full-width
    main_content = mo.vstack([
        header,
        cuped_viz,
        simulator,
        mo.md("---"),
        practical_considerations,
        advanced_info,
        mo.md("### Glossary"),
        glossary,
    ])

    main_content
    return


if __name__ == "__main__":
    app.run()
