#!/usr/bin/env python3
"""Run the Sandwich notebook scenario and save reproducible artifacts.

The configured cases mirror ``notebooks/Sandwich.ipynb``: two sandwiches with
the same sinusoidal boundary gradient and different middle-layer gradient
factors (``mu``).
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys
import time
from dataclasses import dataclass
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

from sandwich_numerical.sandwich import Sandwich


@dataclass(frozen=True)
class SandwichRun:
    name: str
    block_height: int
    block_width: int
    grid_step: float
    mu: float
    iterations: int
    residual_every: int
    progress_every: int
    heatmap_columns: int
    detail_heatmap_columns: int
    sample_x1_positions: tuple[float, ...]
    gradient_profile: str = "sine"


SANDWICH_RUNS: tuple[SandwichRun, ...] = (
    SandwichRun(
        name="mu_1",
        block_height=21,
        block_width=3000,
        grid_step=0.005,
        mu=1.0,
        iterations=10_000,
        residual_every=1_000,
        progress_every=1_000,
        heatmap_columns=1_000,
        detail_heatmap_columns=100,
        sample_x1_positions=(0.0, 0.2, 0.4, 1.0),
    ),
    SandwichRun(
        name="mu_1000",
        block_height=21,
        block_width=3000,
        grid_step=0.005,
        mu=1_000.0,
        iterations=10_000,
        residual_every=1_000,
        progress_every=1_000,
        heatmap_columns=1_000,
        detail_heatmap_columns=100,
        sample_x1_positions=(0.0, 0.2, 0.4, 1.0),
    ),
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
        run_case(run, index, len(runs), output_dir)

    print("Done.")


def parse_args() -> argparse.Namespace:
    parser_description = (
        "Run configured Sandwich notebook scenarios and save CSV/PNG artifacts."
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
    return parser.parse_args()


def prepare_runs(args: argparse.Namespace) -> tuple[SandwichRun, ...]:
    if args.iterations is not None and args.iterations < 0:
        raise SystemExit("--iterations must be non-negative.")
    if args.block_width is not None and args.block_width < 2:
        raise SystemExit("--block-width must be at least 2.")

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
        prepared.append(dataclasses.replace(run, **replacements))

    if not prepared:
        raise SystemExit("No sandwich cases selected.")
    return tuple(prepared)


def run_case(
    run: SandwichRun, case_index: int, case_count: int, output_dir: Path
) -> None:
    case_dir = output_dir / run.name
    case_dir.mkdir(parents=True, exist_ok=True)

    print("")
    print(f"[{case_index}/{case_count}] {run.name}")
    print(
        "  block_size="
        f"({run.block_height}, {run.block_width}), "
        f"grid_step={run.grid_step}, "
        f"mu={run.mu:g}, "
        f"iterations={run.iterations}"
    )

    mesh = create_sandwich(run)
    residual_rows: list[dict[str, float | int]] = [
        {"iteration": 0, "residual": float(mesh.get_residual())}
    ]
    started_at = time.perf_counter()

    for iteration in range(1, run.iterations + 1):
        mesh.step()

        should_sample_residual = (
            iteration % run.residual_every == 0 or iteration == run.iterations
        )
        should_report_progress = (
            iteration % run.progress_every == 0 or iteration == run.iterations
        )

        if should_sample_residual:
            residual = float(mesh.get_residual())
            residual_rows.append({"iteration": iteration, "residual": residual})
        else:
            residual = None

        if should_report_progress:
            elapsed = time.perf_counter() - started_at
            if residual is None:
                print(f"  iter {iteration:>6}/{run.iterations}: {elapsed:6.1f}s")
            else:
                print(
                    f"  iter {iteration:>6}/{run.iterations}: "
                    f"residual={residual:.6e}, elapsed={elapsed:6.1f}s"
                )

    displacement = mesh.get_displacement_array()
    residuals = pd.DataFrame(residual_rows)

    residuals.to_csv(case_dir / "residuals.csv", index=False)
    samples_df = samples_data_to_df(run, displacement)
    samples_df.to_csv(case_dir / "samples.csv", index=False)
    displacement_df = displacement_data_to_df(run, displacement)
    displacement_df.to_csv(case_dir / "displacement.csv", index=False)
    write_case_parameters_csv(case_dir / "parameters.csv", run)

    heatmap_columns = min(run.heatmap_columns, displacement.shape[1])
    detail_columns = min(run.detail_heatmap_columns, displacement.shape[1])
    write_figure_png(
        make_heatmap_figure(
            displacement[:, :heatmap_columns],
            f"Displacement {run.name}: first {heatmap_columns} columns",
        ),
        case_dir / "displacement_heatmap.png",
    )
    write_figure_png(
        make_heatmap_figure(
            displacement[:, :detail_columns],
            f"Displacement {run.name}: first {detail_columns} columns",
        ),
        case_dir / "displacement_detail_heatmap.png",
    )
    write_figure_png(
        make_samples_figure(samples_df, f"Samples {run.name}"),
        case_dir / "samples.png",
    )
    write_figure_png(
        make_residuals_figure(residuals, f"Residual {run.name}"),
        case_dir / "residuals.png",
    )

    print(f"  wrote {case_dir}")


def create_sandwich(run: SandwichRun) -> Sandwich:
    return Sandwich(
        block_size=(run.block_height, run.block_width),
        grad_vec=build_gradient_vector(run),
        grid_step=run.grid_step,
        grad_factor=run.mu,
    )


def build_gradient_vector(run: SandwichRun) -> np.ndarray:
    mesh_height = run.block_height * 3
    if run.gradient_profile != "sine":
        raise ValueError(f"Unsupported gradient profile: {run.gradient_profile}")
    return np.sin((2 * np.pi) / (mesh_height - 1) * np.arange(mesh_height))


def samples_data_to_df(run: SandwichRun, displacement: np.ndarray) -> pd.DataFrame:
    columns: dict[str, np.ndarray] = {
        "x2": np.arange(displacement.shape[0]) * run.grid_step
    }
    for x1_position in run.sample_x1_positions:
        column_index = int(round(x1_position / run.grid_step))
        if column_index >= displacement.shape[1]:
            print(
                f"  skipping sample x1={x1_position:g}: "
                f"column {column_index} is outside width {displacement.shape[1]}"
            )
            continue
        columns[f"x1={x1_position:.1f}"] = displacement[:, column_index]
    return pd.DataFrame(columns)


def displacement_data_to_df(run: SandwichRun, displacement: np.ndarray) -> pd.DataFrame:
    x2_mesh_id = np.arange(displacement.shape[0])
    result: dict[str, np.ndarray] = {
        "x2_mesh_id": x2_mesh_id,
        "x2_bottom": x2_mesh_id * run.grid_step,
        "x2_top": (x2_mesh_id + 1) * run.grid_step,
    }

    for index in range(displacement.shape[1]):
        result[f"x1={index * run.grid_step:.4f}"] = displacement[:, index]
    return pd.DataFrame(result)


def write_case_parameters_csv(path: Path, run: SandwichRun) -> None:
    pd.DataFrame([dataclasses.asdict(run)]).to_csv(path, index=False)


def make_heatmap_figure(data: np.ndarray, title: str):
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
    fig.colorbar(image, ax=ax, label="displacement")
    return fig


def make_samples_figure(samples: pd.DataFrame, title: str):
    fig, ax = plt.subplots(figsize=(11.2, 6.5))
    for column in samples.columns:
        if column == "x2":
            continue
        ax.plot(samples["x2"], samples[column], label=column)
    ax.set_title(title)
    ax.set_xlabel("x2")
    ax.set_ylabel("displacement")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return fig


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
