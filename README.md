# CUPED Simulator

An interactive educational tool for understanding Controlled Experiment Using Pre-Experiment Data (CUPED), a statistical technique that reduces variance in A/B testing by leveraging pre-experiment covariates.

When running A/B tests at scale, typically companies are looking to detect extremely small effects, meaning that the A/B tests they run require large sample sizes. This makes experiments last longer as we have to wait for users to accumulate. CUPED mathematically reduces this variance, enabling faster experimentation with the same statistical power.

This simulation tool is for product mangaers, data scientists, and students who an introduction to the intuition behind CUPED.

## Features

### Simulation Tool
The primary interface focuses on building intuition for CUPED's variance reduction:

- **Interactive parameter exploration**: Adjust sample size, treatment effect, and covariate correlation
- **CUPED vs Standard A/B Test Comparisons**: Side-by-side histograms showing variance reduction for both single-experiment and replicated experiment results
- **Educational guidance**: Clear explanations of CUPED mechanics and benefits
- **Clear business implications**: Clearly shows how CUPED can improve the speed of experiments

### Advanced Information and Connections
For users wanting deeper statistical understanding:

- **Multi-method comparison**: Show asymptotic equivalence between
  - Winston Lin centered regression
  - CUPED covariate adjustment

### Practical Considerations
Explains common pitfalls when implementing CUPED

### Glossary
Defines basic terms associated with CUPED

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
   uv run python CUPED_simulation.py
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

## Tech Stack

- **UI Framework**: Marimo for reactive, interactive notebooks
- **Visualization**: Altair for interactive statistical plots, matplotlib for additional plotting
- **Computation**: numpy, pandas, scipy, statsmodels for data processing and statistics
- **Package Management**: uv for modern Python dependency management
- **Architecture**: Functional programming with pure data transformations

## Notes
- Vectorized operations improved simulation speed by ~63%

## To-do list
- deploy notebook
