# CARM: Counterfactual Attribution Regret Minimization in Spatial Public Goods Games

This repository contains the complete implementation, simulation data, and analysis code for the research paper **"Dynamics of Counterfactual Attribution Regret Minimization in Spatial Public Goods Games"**.

## Overview

This project investigates the evolutionary dynamics of cooperation in spatial public goods games using a novel decision-making mechanism called **Counterfactual Attribution Regret Minimization (CARM)**. The CARM algorithm models how agents learn from counterfactual scenarios by considering both the accessibility of alternative outcomes and their attribution to individual decisions rather than luck.

### Key Features

- **Spatial Public Goods Game**: Agents on a lattice participate in public goods games with their neighbors
- **CARM Learning Mechanism**: Agents update strategies based on weighted counterfactual regret with distance-based accessibility and attribution sensitivity
- **Comprehensive Parameter Analysis**: Systematic exploration of model parameters (β, γ, κ) and their effects on cooperation dynamics
- **Multiple Dynamic Regimes**: From stable cooperation to complex oscillations and phase transitions

## Repository Structure

```
├── src/                          # Core simulation code
│   ├── core/                     # Core algorithm implementation
│   ├── run_fig2_simulations.py   # Time series experiments
│   ├── run_fig3_simulations.py   # Parameter sensitivity analysis
│   ├── run_fig4_simulations.py   # 2D parameter space exploration
|   ...
├── config/                       # Experiment configuration files
├── data/                         # Generated simulation data
├── analysis/                     # Plotting and analysis scripts
├── figures/                      # Generated plots and figures
├── Demo.py                       # Standalone demonstration script
└── submit_*.sh                   # Job submission scripts for HPC
```


## Experiments and Figures

### fig2_simulation: Time Series Dynamics
**Script**: `src/run_fig2_simulations.py`
**Analysis**: `analysis/plot_figure_2.py`

Demonstrates three distinct behavioral regimes of the CARM model. By reading the corresponding configuration file(e.g. `fig2_combo1_low_coop,json`), the parameters can be freely adjusted.

### fig3_simulation: Parameter Sensitivity Analysis
**Scripts**: `src/run_fig3_simulations.py`
**Analysis**: `analysis/plot_figure_3.py`

Systematic exploration of how each CARM parameter affects equilibrium cooperation levels:

Each subplot shows cooperation levels across different synergy factors (R) for multiple parameter values.

Note that the generated illustrations here have different numbering from those in the paper. Please refer to the paper for the correct names.

### fig4_simulation: Two-Dimensional Parameter Landscapes
**Scripts**: `src/run_fig4_simulations.py`
**Analysis**: `analysis/plot_figure_4.py`

Heat maps revealing the complex parameter space structure:
- **Fig 4a**: β vs γ parameter space (κ fixed)
- **Fig 4b**: γ vs κ parameter space (β fixed)
- **Fig 4c**: κ vs β parameter space (γ fixed)

Shows both mean cooperation levels and their standard deviations across the parameter space.

### Figure 5: Dynamic Regime Classification
**Scripts**: `src/run_fig5_simulations.py`
**Analysis**: `analysis/plot_figure_5.py`

Detailed time series analysis of different dynamic regimes.

Note that the variable names in the code are different from those in the paper. Please refer to the paper for the correct names.

### Figure 8: Strategy Switching Behavior
**Scripts**: `src/run_fig8_simulations.py`
**Analysis**: `analysis/plot_figure_8.py`

Microscopic analysis of individual agent decision-making:
- Probability of strategy switching as a function of local environment
- Environmental frequency distributions for cooperators vs defectors
- Transition rates between strategies under different conditions

## Quick Start

### Running a Demonstration
```bash
python Demo.py
```

This script runs a basic CARM simulation with default parameters and provides real-time visualization of the evolution.

### Running Full Experiments

1. **Generate simulation data**:
```bash
python src/run_fig2_simulations.py  # Time series experiments
python src/run_fig3_simulations.py  # Parameter sensitivity
# ... etc for other figures
```

2. **Generate plots**:
```bash
python analysis/plot_figure_2.py   # Create Figure 2
python analysis/plot_figure_3.py   # Create Figure 3
# ... etc for other figures
```

### High-Performance Computing
For large-scale simulations, use the provided submission scripts:
```bash
sbatch submit_fig3.sh  # Submit Figure 3 experiments to SLURM
sbatch submit_fig7.sh  # Submit Figure 7 experiments to SLURM
sbatch submit_fig8.sh  # Submit Figure 8 experiments to SLURM
```

## Configuration

Experimental parameters are stored in JSON files in the `config/` directory. Each file contains:
- Grid and agent parameters (L, N, initialization)
- Game parameters (R_FACTOR, COST)
- CARM parameters (BETA, GAMMA, KAPPA)
- Simulation parameters (duration, sampling intervals)
- Analysis parameters (replicas, parameter ranges)

## Data Output

Simulation results are automatically saved to the `data/` directory, organized by figure:
- Time series data: cooperation fraction (ρ_C) and strategy correlation (L_CD) over time
- Parameter sweep data: equilibrium values across parameter ranges
- Individual run data: detailed agent-level information when needed

## Dependencies

- Python 3.7+
- NumPy
- Pandas  
- Matplotlib
- SciPy (for some analyses)
- Concurrent.futures (for parallel simulations)

