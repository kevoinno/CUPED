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

## Setup

### Prerequisites
- Python 3.13+
- uv package manager (`pip install uv` or `brew install uv`)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd cuped
   ```

2. **Install dependencies**
   ```bash
   uv sync
   ```

3. **Run the simulator**
   ```bash
   uv run python CUPED_test.py
   ```

### Development (Optional)

Add development tools:
```bash
uv add --dev black ruff pytest
```

Run development commands:
```bash
uv run black .          # Format code
uv run ruff check .     # Lint code
uv run pytest           # Run tests (when available)
```

Navigate the tabbed interface to explore CUPED interactively.

## Technical Stack

- **UI Framework**: Marimo for reactive, interactive notebooks
- **Visualization**: Altair for interactive statistical plots, matplotlib for additional plotting
- **Computation**: numpy, pandas, scipy for data processing and statistics
- **Package Management**: uv for modern Python dependency management
- **Architecture**: Functional programming with pure data transformations

## To-do list
- Fix constant reruns when I change the slider by just a little 
  - Playing around with threshold?
  - Find some way to keep reactivity without taking so long

- Add results for a single experiment
- Show equivalence with Lin (2013) regression


## Notes
- Vectorizing the simulation code led to ~63% faster simulation speed
