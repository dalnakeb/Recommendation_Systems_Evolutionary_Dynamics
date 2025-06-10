# Recommendation Systems Evolutionary Dynamics

A Python package and collection of Jupyter notebooks for simulating evolutionary dynamics in recommendation systems. These tools were developed as part of a thesis project and include a variety of Evolutionary Game Theory (EGT) simulations, model implementations, and result visualizations.

## Table of Contents

1. [Repository Structure](#repository-structure)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Usage](#usage)

   * [Notebooks](#notebooks)
   * [Scripts](#scripts)
5. [Output](#output)
6. [Contributing](#contributing)
7. [License](#license)

## Repository Structure

```plain
├── notebooks/                      # Jupyter notebooks for EGT simulations
│   ├── youtube_evol_dyn.ipynb      # Simulations and unused code (Moran process, gradient of selection, replicator dynamics)
│   ├── models_1.ipynb              # First class of models (View Count)
│   ├── models_2.ipynb              # Second class of models (Watch Time)
│   └── models_3.ipynb              # Third class of models (Valued Watch Time)
├── src/                            # Package source code
│   ├── recommendation_systems_evolutionary_dynamics/  # Core modules and functions
│       └── __init__.py             # Package initialization (global variables, constants)
│       └── EGT.py                  # Main code containing the EGT Game class simulations
│   └── setup.py                    # Installation script
├── output/                         # Generated plots and figures categorized by model class
│   └── ...                         # Additional output categories
├── requirements.txt                # Dependency list with specific versions
└── README.md                       # This file
```

## Requirements

All required Python packages and version constraints are listed in `requirements.txt`. Install them via:

```bash
pip install -r requirements.txt
```

## Installation

To install the package in editable mode (allowing quick code modifications and making it available to the notebooks):

```bash
pip install -e .
```

This creates a link from your Python environment to the local source directory.

## Usage

### Notebooks

1. Navigate to the `notebooks/` directory:

   ```bash
   cd notebooks
   ```

2. Launch Jupyter Lab or Notebook:

   ```bash
   jupyter lab
   ```

3. For the models_*.ipynb notebook, run then and the outputs will be saved automatically into the corresponding subfolder under `output/`.
4. For the youtube_evol_dyn.ipynb, choose a simulation to run given the initial definition of parameters and model 
## Output

All generated figures and data visualizations from the notebooks are stored in the `output/` directory, organized by model class:

* `output/view_count/`
* ...