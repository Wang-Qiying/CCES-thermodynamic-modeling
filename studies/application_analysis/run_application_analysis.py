"""Run the PESIM application cases individually or together."""
import argparse

from application_cases import (
    run_three_cycle_case,
    run_constant_power_envelope,
    run_well_ablation,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "case", nargs="?", default="all",
        choices=("cycles", "envelope", "ablation", "all"),
        help="application case to run",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    progress = not args.quiet
    if args.case in ("cycles", "all"):
        run_three_cycle_case(progress=progress)
    if args.case in ("envelope", "all"):
        run_constant_power_envelope(progress=progress)
    if args.case in ("ablation", "all"):
        run_well_ablation(progress=progress)


if __name__ == "__main__":
    main()
