#!/usr/bin/env python3
"""Run the Sandwich integration scenarios and save reproducible artifacts."""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys
import time
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/sandwich_matplotlib_config")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/sandwich_cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sandwich_numerical.integration import (
    GRADIENT_PROFILES,
    SANDWICH_RUNS,
    SandwichRun,
    SandwichSolution,
    solve_case,
    write_solution_csvs,
)


FIGURE_WIDTH = 1120
FIGURE_HEIGHT = 650


def main() -> None:
    args = parse_args()
    runs = prepare_runs(args)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Writing artifacts to {output_dir}")
    print(f"Configured sandwiches: {len(runs)}")

    for index, run in enumerate(runs, start=1):
        run_case(run, index, len(runs), output_dir, write_figures=not args.csv_only)

    print("Done.")


def parse_args() -> argparse.Namespace:
    parser_description = (
        "Run configured Sandwich integration scenarios and save CSV/PNG artifacts."
    )
    parser = argparse.ArgumentParser(description=parser_description)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/sandwich_integration"),
        help="Directory for generated CSV and PNG files.",
    )
    parser.add_argument(
        "--case",
        choices=[run.name for run in SANDWICH_RUNS],
        action="append",
        help="Run only the selected case. Can be passed more than once.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        help="Override iteration count for all selected sandwiches.",
    )
    parser.add_argument(
        "--block-width",
        type=int,
        help="Override block width for all selected sandwiches; useful for smoke runs.",
    )
    parser.add_argument(
        "--gradient-relaxation",
        type=float,
        help="Under-relaxation for copied interface gradients.",
    )
    parser.add_argument(
        "--gradient-profile",
        choices=GRADIENT_PROFILES,
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
    return parser.parse_args()


def prepare_runs(args: argparse.Namespace) -> tuple[SandwichRun, ...]:
    if args.iterations is not None and args.iterations < 0:
        raise SystemExit("--iterations must be non-negative.")
    if args.block_width is not None and args.block_width < 2:
        raise SystemExit("--block-width must be at least 2.")
    if args.gradient_relaxation is not None:
        if not (0 < args.gradient_relaxation <= 1):
            raise SystemExit("--gradient-relaxation must be in the interval (0, 1].")

    runs: Iterable[SandwichRun] = SANDWICH_RUNS
    if args.case:
        selected_names = set(args.case)
        runs = [run for run in runs if run.name in selected_names]

    prepared = []
    for run in runs:
        replacements = {}
        if args.iterations is not None:
            replacements["iterations"] = args.iterations
            interval = max(1, min(run.residual_every, args.iterations))
            replacements["residual_every"] = interval
            replacements["progress_every"] = interval
        if args.block_width is not None:
            replacements["block_width"] = args.block_width
            replacements["heatmap_columns"] = min(run.heatmap_columns, args.block_width)
            replacements["detail_heatmap_columns"] = min(
                run.detail_heatmap_columns, args.block_width
            )
        if args.gradient_relaxation is not None:
            replacements["gradient_relaxation"] = args.gradient_relaxation
        if args.gradient_profile is not None:
            replacements["gradient_profile"] = args.gradient_profile
            if args.gradient_profile != run.gradient_profile:
                replacements["name"] = f"{run.name}_{args.gradient_profile}"
        if args.enforce_overlap_continuity is not None:
            replacements["enforce_overlap_continuity"] = args.enforce_overlap_continuity
        prepared.append(dataclasses.replace(run, **replacements))

    if not prepared:
        raise SystemExit("No sandwich cases selected.")
    return tuple(prepared)


def run_case(
    run: SandwichRun,
    case_index: int,
    case_count: int,
    output_dir: Path,
    write_figures: bool = True,
) -> None:
    case_dir = output_dir / run.name
    case_dir.mkdir(parents=True, exist_ok=True)

    print("")
    print(f"[{case_index}/{case_count}] {run.name}")
    print(
        "  block_size="
        f"({run.block_height}, {run.block_width}), "
        f"grid_step={run.grid_step}, "
        f"grad_factors={run.grad_factors}, "
        f"gradient_profile={run.gradient_profile}, "
        f"gradient_relaxation={run.gradient_relaxation}, "
        f"enforce_overlap_continuity={run.enforce_overlap_continuity}, "
        f"iterations={run.iterations}"
    )

    started_at = time.perf_counter()

    solution = solve_case(
        run,
        progress_callback=lambda iteration, residual: report_progress(
            iteration, residual, run.iterations, started_at
        ),
    )

    write_solution_csvs(case_dir, run, solution)
    if write_figures:
        write_solution_figures(case_dir, run, solution)

    print(f"  wrote {case_dir}")


def report_progress(
    iteration: int, residual: float | None, total_iterations: int, started_at: float
) -> None:
    elapsed = time.perf_counter() - started_at
    if residual is None:
        print(f"  iter {iteration:>6}/{total_iterations}: {elapsed:6.1f}s")
    else:
        print(
            f"  iter {iteration:>6}/{total_iterations}: "
            f"residual={residual:.6e}, elapsed={elapsed:6.1f}s"
        )


def write_solution_figures(
    case_dir: Path, run: SandwichRun, solution: SandwichSolution
) -> None:
    displacement = solution.displacement.drop(
        columns=["x2_mesh_id", "x2_bottom", "x2_top"]
    ).to_numpy()
    heatmap_columns = min(run.heatmap_columns, displacement.shape[1])
    detail_columns = min(run.detail_heatmap_columns, displacement.shape[1])
    write_figure_png(
        make_heatmap_figure(
            displacement[:, :heatmap_columns],
            f"Displacement {run.name}: first {heatmap_columns} columns",
            layer_boundary_rows(run),
        ),
        case_dir / "displacement_heatmap.png",
    )
    write_figure_png(
        make_heatmap_figure(
            displacement[:, :detail_columns],
            f"Displacement {run.name}: first {detail_columns} columns",
            layer_boundary_rows(run),
        ),
        case_dir / "displacement_detail_heatmap.png",
    )
    write_figure_png(
        make_samples_figure(
            solution.samples,
            f"Samples {run.name}",
            layer_boundary_x2(run),
        ),
        case_dir / "samples.png",
    )
    write_figure_png(
        make_residuals_figure(solution.residuals, f"Residual {run.name}"),
        case_dir / "residuals.png",
    )


def layer_boundary_rows(run: SandwichRun) -> np.ndarray:
    return np.arange(1, len(run.grad_factors)) * run.block_height - 0.5


def layer_boundary_x2(run: SandwichRun) -> np.ndarray:
    return layer_boundary_rows(run) * run.grid_step


def make_heatmap_figure(
    data: np.ndarray, title: str, layer_boundaries: np.ndarray | None = None
):
    fig, ax = plt.subplots(figsize=(11.2, 6.5))
    image = ax.imshow(
        data,
        aspect="auto",
        cmap="RdBu_r",
        norm=make_displacement_norm(data),
        origin="lower",
    )
    ax.set_title(title)
    ax.set_xlabel("x1 column")
    ax.set_ylabel("x2 row")
    add_horizontal_layer_boundaries(ax, layer_boundaries, data.shape[0])
    fig.colorbar(image, ax=ax, label="displacement")
    return fig


def make_samples_figure(
    samples: pd.DataFrame, title: str, layer_boundaries: np.ndarray | None = None
):
    fig, ax = plt.subplots(figsize=(11.2, 6.5))
    for column in samples.columns:
        if column == "x2":
            continue
        ax.plot(samples["x2"], samples[column], label=column)
    add_vertical_layer_boundaries(ax, layer_boundaries)
    ax.set_title(title)
    ax.set_xlabel("x2")
    ax.set_ylabel("displacement")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return fig


def add_horizontal_layer_boundaries(
    ax, layer_boundaries: np.ndarray | None, row_count: int
) -> None:
    if layer_boundaries is None:
        return
    for boundary in layer_boundaries:
        if 0 < boundary < row_count - 1:
            ax.axhline(
                boundary, color="black", linestyle="--", linewidth=0.8, alpha=0.5
            )


def add_vertical_layer_boundaries(ax, layer_boundaries: np.ndarray | None) -> None:
    if layer_boundaries is None:
        return
    for index, boundary in enumerate(layer_boundaries):
        ax.axvline(
            boundary,
            color="black",
            linestyle="--",
            linewidth=0.8,
            alpha=0.5,
            label="layer boundary" if index == 0 else "_nolegend_",
        )


def make_residuals_figure(residuals: pd.DataFrame, title: str):
    fig, ax = plt.subplots(figsize=(11.2, 6.5))
    ax.plot(residuals["iteration"], residuals["residual"])
    ax.set_title(title)
    ax.set_xlabel("iteration")
    ax.set_ylabel("residual")
    ax.grid(True, alpha=0.3)
    return fig


def make_displacement_norm(data: np.ndarray):
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return None
    min_value = float(finite.min())
    max_value = float(finite.max())
    if min_value < 0 < max_value:
        limit = max(abs(min_value), abs(max_value))
        return TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    return None


def write_figure_png(
    fig, path: Path, width: int = FIGURE_WIDTH, height: int = FIGURE_HEIGHT
) -> None:
    fig.set_size_inches(width / 100, height / 100)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
