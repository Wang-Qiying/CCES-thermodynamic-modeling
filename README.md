# Dynamic sCO2 Dual-Cavern Energy Storage Model

This repository contains the reproducible simulation code and complete parameter
set used for the application studies of a dynamic supercritical carbon dioxide
(sCO2) energy storage system. The model combines real-fluid CO2 properties,
lumped salt-cavern mass and energy balances, cavern-wall heat transfer, dynamic
wellbore pressure/temperature mapping, compressor and turbine models, and
stage-dependent operating constraints.

## Included studies

The repository exposes three independent application functions:

1. `run_three_cycle_case()` -- three fixed-flow charge/standby/discharge/standby
   cycles with a 2% normalized operating-margin floor.
2. `run_constant_power_envelope()` -- constant-power discharge scans from 2 to
   9 MW using the saved charge endpoint from cycle 3.
3. `run_well_ablation()` -- comparison of the full wellbore, frictionless
   adiabatic wellbore, and no-wellbore formulations.

All simulation inputs use SI units and are defined in
`studies/application_analysis/case_config.yaml`.

## Repository layout

```text
.
|-- co2sim/                         # thermodynamic and dynamic model
|-- studies/application_analysis/
|   |-- application_cases.py        # three public study functions
|   |-- case_config.yaml             # complete PESIM case parameters
|   |-- run_application_analysis.py  # secondary command-line entry point
|   `-- paper_case_analysis/
|       `-- build_case_analysis_assets.py
|-- run_cases.py                    # main command-line entry point
|-- check_installation.py           # fast dependency/configuration check
|-- requirements.txt                # supported dependency ranges
|-- requirements-lock.txt           # exact tested environment snapshot
|-- REQUIREMENTS.md                 # software and hardware notes
`-- CITATION.md                     # citation templates
```

Generated results are written below
`studies/application_analysis/results/` and are intentionally not included in
the source package.

## Installation

Python 3.10 or newer is required. From the repository root:

```bash
python -m venv .venv
```

Activate the environment on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

or on Linux/macOS:

```bash
source .venv/bin/activate
```

Then install the dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python check_installation.py
```

For the exact package versions used for the archived calculations, replace
`requirements.txt` with `requirements-lock.txt` in the installation command.

## Running the studies

Run the three cases in sequence and then generate the publication figures:

```bash
python run_cases.py all
```

Run one case at a time:

```bash
python run_cases.py cycles
python run_cases.py envelope
python run_cases.py ablation
python run_cases.py figures
```

The envelope case depends on the cycle-3 charge endpoint, so `cycles` must be
run before `envelope`. Use `--quiet` to suppress progress output.

The functions can also be called directly:

```python
from studies.application_analysis import (
    run_three_cycle_case,
    run_constant_power_envelope,
    run_well_ablation,
)

run_three_cycle_case()
run_constant_power_envelope()
run_well_ablation()
```

## Baseline study configuration

- High-pressure cavern: depth 2.5 km; pressure range 16.50--44.00 MPa;
  initial temperature 95 degC.
- Low-pressure cavern: depth 1.2 km; pressure range 7.92--21.12 MPa;
  initial temperature 56 degC.
- Fixed-flow cycling rate: 100 kg/s.
- Compressor motor and turbine-generator ratings: 10 MW.
- Constant-power scan: 2, 3, 4, 5, 6, 7, 8, and 9 MW.
- Adopted mass-flow range: 50--500 kg/s.
- Standby duration after every active stage: 2 h.
- Active-stage endpoint: minimum normalized modeled/evaluated margin of 0.02.
- Water-based thermal storage: disabled for this pressure-ratio configuration.

## Outputs

Each study writes CSV trajectories and summaries, JSON event records, parameter
snapshots, and diagnostic plots. After all three cases have completed,
`python run_cases.py figures` creates the compact manuscript figures and their
panel-level CSV source data under
`studies/application_analysis/paper_case_analysis/`.

## Scope and limitations

The code is a research model, not a certified cavern or turbomachinery design
tool. A nonnegative normalized margin guarantees only the constraints that are
implemented, active, and numerically evaluated in the selected stage. Site-
specific geomechanical integrity, completion design, equipment maps, and other
unevaluated engineering constraints require separate assessment.

## Citation and license

See `CITATION.md` for an IEEE `bibitem` and `CITATION.cff` for GitHub citation
metadata. No software license has been selected in this package; add an
appropriate license before public release if reuse or redistribution is
intended.
