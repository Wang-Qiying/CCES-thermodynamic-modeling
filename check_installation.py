"""Fast installation check; no dynamic simulation is performed."""

from __future__ import annotations

from pathlib import Path

import CoolProp
import matplotlib
import numpy
import pandas
import scipy
import yaml

from co2sim.model import Model, load_config


ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "studies" / "application_analysis" / "case_config.yaml"


def main() -> None:
    model = Model(load_config(CONFIG))
    states = model.states(model.x0)
    print("Installation check passed.")
    print(f"Fluid backend: {model.cfg['fluid']}")
    print(f"Initial total CO2 mass: {model.total_mass:.6f} kg")
    print(
        "Initial cavern states: "
        f"H={states['H'].p/1e6:.3f} MPa, {states['H'].T-273.15:.2f} degC; "
        f"L={states['L'].p/1e6:.3f} MPa, {states['L'].T-273.15:.2f} degC"
    )
    print(
        "Versions: "
        f"CoolProp={CoolProp.__version__}, NumPy={numpy.__version__}, "
        f"SciPy={scipy.__version__}, Pandas={pandas.__version__}, "
        f"Matplotlib={matplotlib.__version__}, PyYAML={yaml.__version__}"
    )


if __name__ == "__main__":
    main()
