"""Run the PESIM application studies individually or as a reproducible suite."""

from __future__ import annotations

import argparse
import runpy
from pathlib import Path

from studies.application_analysis import (
    run_constant_power_envelope,
    run_three_cycle_case,
    run_well_ablation,
)


ROOT = Path(__file__).resolve().parent
FIGURE_SCRIPT = (
    ROOT
    / "studies"
    / "application_analysis"
    / "paper_case_analysis"
    / "build_case_analysis_assets.py"
)


def build_publication_figures() -> None:
    """Build manuscript figures from existing outputs without rerunning cases."""
    runpy.run_path(str(FIGURE_SCRIPT), run_name="__main__")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "case",
        nargs="?",
        default="all",
        choices=("cycles", "envelope", "ablation", "figures", "all"),
        help="study or post-processing task to run",
    )
    parser.add_argument("--quiet", action="store_true", help="suppress progress output")
    args = parser.parse_args()
    progress = not args.quiet

    if args.case in ("cycles", "all"):
        run_three_cycle_case(progress=progress)
    if args.case in ("envelope", "all"):
        run_constant_power_envelope(progress=progress)
    if args.case in ("ablation", "all"):
        run_well_ablation(progress=progress)
    if args.case in ("figures", "all"):
        build_publication_figures()


if __name__ == "__main__":
    main()
