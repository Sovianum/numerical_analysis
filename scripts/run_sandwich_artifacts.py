#!/usr/bin/env python3
"""Run unified FDM/FEM Sandwich artifact generation locally or in CI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sandwich_numerical import artifacts  # noqa: E402
from sandwich_numerical.fdm.integration import (  # noqa: E402
    solve_case as solve_fdm_case,
)
from sandwich_numerical.fem.integration import (  # noqa: E402
    solve_case as solve_fem_case,
)


LOCAL_DEFAULT_CASE = "grad_factors_1_1_1_load_sin"
LOCAL_DEFAULT_BLOCK_WIDTH = 50
SOLVERS = {
    "fdm": solve_fdm_case,
    "fem": solve_fem_case,
}


def main() -> None:
    args = parse_args()
    output_dir = selected_output_dir(args).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.aggregate_only:
        print(f"Building aggregate comparisons from {output_dir}")
        artifacts.write_all_cases_comparison_figures(
            output_dir,
            case_names=selected_case_names(args, aggregate=True),
        )
        print("Done.")
        return

    runs = artifacts.prepare_runs(prepared_run_args(args))
    solver_names = selected_solver_names(args.solver)
    print(f"Writing artifacts to {output_dir}")
    print(f"Configured sandwiches: {len(runs)}")
    print(f"Configured solvers: {', '.join(solver_names)}")

    for index, run in enumerate(runs, start=1):
        case_dir = output_dir / run.name
        for solver_name in solver_names:
            artifacts.run_case_to_dir(
                run,
                case_dir / solver_name,
                solve_case=SOLVERS[solver_name],
                write_figures=not args.csv_only,
                case_index=index,
                case_count=len(runs),
                label=f"{run.name} [{solver_name}]",
            )

        if solver_names == artifacts.SOLVER_NAMES:
            artifacts.write_case_comparison_figures(
                case_dir,
                run,
                write_figures=not args.csv_only,
            )

    if solver_names == artifacts.SOLVER_NAMES and not args.csv_only and len(runs) > 1:
        artifacts.write_all_cases_comparison_figures(
            output_dir,
            case_names=tuple(run.name for run in runs),
        )

    print("Done.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Sandwich FDM/FEM CSV and PNG artifacts."
    )
    parser.add_argument(
        "--mode",
        choices=("local", "ci"),
        default="local",
        help="Use local debug defaults or CI defaults.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for generated CSV and PNG files.",
    )
    parser.add_argument(
        "--case",
        choices=artifacts.case_choices(),
        action="append",
        help="Run only the selected case or load-specific scenario.",
    )
    parser.add_argument(
        "--all-cases",
        action="store_true",
        help="Run all CI load-specific cases.",
    )
    parser.add_argument(
        "--solver",
        choices=("fdm", "fem", "both"),
        default="both",
        help="Select which solver artifacts to generate.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        help="Override iteration count for all selected sandwiches.",
    )
    parser.add_argument(
        "--block-width",
        type=int,
        help="Override block width; useful for local smoke runs.",
    )
    parser.add_argument(
        "--gradient-relaxation",
        type=float,
        help="Under-relaxation for copied interface gradients.",
    )
    parser.add_argument(
        "--gradient-profile",
        choices=artifacts.GRADIENT_PROFILES,
        help="Override the boundary gradient profile for selected sandwiches.",
    )
    overlap_group = parser.add_mutually_exclusive_group()
    overlap_group.add_argument(
        "--enforce-overlap-continuity",
        dest="enforce_overlap_continuity",
        action="store_true",
        default=None,
        help="Average duplicate real/ghost rows shared by adjacent blocks.",
    )
    overlap_group.add_argument(
        "--no-enforce-overlap-continuity",
        dest="enforce_overlap_continuity",
        action="store_false",
        help="Disable averaging of duplicate real/ghost rows.",
    )
    parser.add_argument(
        "--csv-only",
        action="store_true",
        help="Write CSV data without PNG figures.",
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Read existing per-case CSV files and build only all-case comparisons.",
    )
    return parser.parse_args()


def selected_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir
    if args.mode == "ci":
        return Path("artifacts/sandwich_integration")
    return Path("artifacts/local_sandwich_integration")


def selected_case_names(
    args: argparse.Namespace, aggregate: bool = False
) -> tuple[str, ...]:
    if args.case:
        return tuple(args.case)
    if args.mode == "ci" or args.all_cases or aggregate:
        return artifacts.ci_case_names()
    return (LOCAL_DEFAULT_CASE,)


def prepared_run_args(args: argparse.Namespace) -> argparse.Namespace:
    block_width = args.block_width
    if args.mode == "local" and not args.all_cases and block_width is None:
        block_width = LOCAL_DEFAULT_BLOCK_WIDTH

    return argparse.Namespace(
        case=list(selected_case_names(args)),
        iterations=args.iterations,
        block_width=block_width,
        gradient_relaxation=args.gradient_relaxation,
        gradient_profile=args.gradient_profile,
        enforce_overlap_continuity=args.enforce_overlap_continuity,
    )


def selected_solver_names(solver: str) -> tuple[str, ...]:
    if solver == "both":
        return artifacts.SOLVER_NAMES
    return (solver,)


if __name__ == "__main__":
    main()
