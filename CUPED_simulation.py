import marimo

__generated_with = "0.18.4"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import altair as alt
    import statsmodels.formula.api as smf
    from scipy.stats import ttest_ind
    return alt, mo, np, pd, smf, ttest_ind


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

    # Build controls panel layout
    controls_panel = mo.vstack([
        mo.md("## Experiment Settings"),
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
        mo.md("## Simulation Settings"),
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
        <div style="flex: 1; padding: 1rem; background: #f1f5f9; border-radius: 8px; border-left: 4px solid #6366f1;">
            <div style="color: #64748b; font-size: 0.875rem; margin-bottom: 0.25rem;">Naive Estimate</div>
            <div style="font-size: 1.5rem; font-weight: 600; color: #1e293b;">{naive_effect:.3f}</div>
            <div style="color: #64748b; font-size: 0.75rem;">95% CI: [{naive_ci_low:.3f}, {naive_ci_high:.3f}]</div>
            <div style="color: #64748b; font-size: 0.75rem;">Std. error: {naive_se:.3f}</div>
            <div style="color: #64748b; font-size: 0.75rem;">p-value: {naive_pvalue:.4f}</div>
        </div>
        <div style="flex: 1; padding: 1rem; background: #f0fdf4; border-radius: 8px; border-left: 4px solid #10b981;">
            <div style="color: #64748b; font-size: 0.875rem; margin-bottom: 0.25rem;">CUPED Estimate</div>
            <div style="font-size: 1.5rem; font-weight: 600; color: #1e293b;">{cuped_effect:.3f}</div>
            <div style="color: #64748b; font-size: 0.75rem;">95% CI: [{cuped_ci_low:.3f}, {cuped_ci_high:.3f}]</div>
            <div style="color: #64748b; font-size: 0.75rem;">Std. error: {cuped_se:.3f}</div>
            <div style="color: #64748b; font-size: 0.75rem;">p-value: {cuped_pvalue:.4f}</div>
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

    # Tab 1 content
    tab1_content = mo.vstack([
        tab1_scorecard,
        mo.ui.altair_chart(tab1_chart),
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
def _(controls_panel, glossary, mo, tab1_content, tab2_content, tab3_content):
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

    # Main layout: controls on left, content on right, then practical considerations, advanced info, and glossary full-width
    main_content = mo.vstack([
        header,
        mo.hstack([
            controls_panel,
            right_content,
        ], gap=4, align="start"),
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
