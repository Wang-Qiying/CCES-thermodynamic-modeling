# Requirements

## Required software

- Python 3.10 or newer (Python 3.12 was used for the archived run)
- A 64-bit operating system
- Internet access during dependency installation, unless wheels are cached

## Required Python packages

The runtime dependency list is maintained in `requirements.txt`:

- NumPy -- array operations
- SciPy -- integration, root finding, and nonlinear solvers
- CoolProp -- HEOS real-fluid CO2 properties
- Matplotlib -- simulation and publication figures
- Pandas -- tabular outputs and post-processing
- PyYAML -- configuration and parameter snapshots

`requirements-lock.txt` records the exact tested package versions, including
transitive dependencies.

## Optional software

- A LaTeX distribution is needed only to compile manuscript source; it is not
  required to run the simulations or generate PDF figures with Matplotlib.
- Git is needed only for version control and publication on GitHub.

## Hardware and runtime

No GPU is required. Runtime depends strongly on CPU performance and the selected
integration tolerances. The constant-power scan and wellbore ablation execute
multiple nonlinear real-fluid simulations and are substantially slower than the
installation check.

## Numerical configuration

The archived case uses relative integration tolerance `1e-7`, a 120 s maximum
step for fixed-flow cycling, and a 300 s maximum step for the constant-power
scan. These values are stored in the case configuration and study driver.
