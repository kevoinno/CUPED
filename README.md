# CUPED Simulator

An interactive educational tool for understanding Controlled Experiment Using Pre-Experiment Data (CUPED), a statistical technique that reduces variance in A/B testing by leveraging pre-experiment covariates.

**Why it exists:** A/B testing often requires large sample sizes due to high outcome variance. CUPED mathematically reduces this variance, enabling faster experimentation with the same statistical power.

**Target users:** Data scientists, analysts, and researchers wanting to understand and apply CUPED in their experimentation platforms.

## Features

### Core CUPED Learning (Main Tab)

The primary interface focuses on building intuition for CUPED's variance reduction:

- **Interactive parameter exploration**: Adjust sample size, treatment effect, and covariate correlation
- **CUPED vs naive comparison**: Side-by-side histograms showing variance reduction
- **Real-time results**: See standard error reduction and confidence interval improvements
- **Educational guidance**: Clear explanations of CUPED mechanics and benefits

### Advanced Method Comparison (Secondary Tab)

For users wanting deeper statistical understanding:

- **Multi-method comparison**: Compare variance reduction across:
  - Multi-linear regression adjustment
  - Winston Lin centered regression
  - CUPED covariate adjustment
- **Equivalence demonstration**: Show mathematical relationship between methods
- **Performance insights**: Understand when each method performs best

### Production Planning

Scale planning for real-world A/B testing:

- **Time savings calculator**: Estimate experiment duration reduction
- **Sample size optimization**: Calculate required samples with CUPED
- **ROI quantification**: Demonstrate engineering time savings

## Quick Start

```bash
pip install marimo matplotlib numpy pandas scipy
marimo run CUPED_test.py
```

Navigate the tabbed interface to explore CUPED interactively.

## Technical Stack

- **UI Framework**: Marimo for reactive, interactive notebooks
- **Visualization**: matplotlib for statistical plots
- **Computation**: numpy, pandas, scipy for data processing and statistics
- **Architecture**: Functional programming with pure data transformations

## Roadmap

- [ ] Clean up A/B testing function
- [ ] Build main page tab 
- [ ] Build core CUPED learning interface with interactive controls
- [ ] Add multi-method comparison functions and visualizations
- [ ] Polish user experience and educational content
- [ ] Add production planning calculator
