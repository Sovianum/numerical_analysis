"""Reusable Sandwich integration scenario runner."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

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


@dataclass(frozen=True)
class SandwichSolution:
    residuals: pd.DataFrame
    samples: pd.DataFrame
    displacement: pd.DataFrame


ProgressCallback = Callable[[int, float | None], None]


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


def solve_case(
    run: SandwichRun, progress_callback: ProgressCallback | None = None
) -> SandwichSolution:
    mesh = create_sandwich(run)
    residual_rows: list[dict[str, float | int]] = [
        {"iteration": 0, "residual": float(mesh.get_residual())}
    ]

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

        if should_report_progress and progress_callback is not None:
            progress_callback(iteration, residual)

    displacement = mesh.get_displacement_array()
    residuals = pd.DataFrame(residual_rows)
    samples = samples_data_to_df(run, displacement)
    displacement_data = displacement_data_to_df(run, displacement)

    return SandwichSolution(
        residuals=residuals,
        samples=samples,
        displacement=displacement_data,
    )


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
    gradient = np.sin((2 * np.pi) / (mesh_height - 1) * np.arange(mesh_height))
    return np.asarray(gradient, dtype=float)


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


def parameters_data_to_df(run: SandwichRun) -> pd.DataFrame:
    return pd.DataFrame([dataclasses.asdict(run)])


def write_solution_csvs(
    case_dir: Path, run: SandwichRun, solution: SandwichSolution
) -> None:
    solution.residuals.to_csv(case_dir / "residuals.csv", index=False)
    solution.samples.to_csv(case_dir / "samples.csv", index=False)
    solution.displacement.to_csv(case_dir / "displacement.csv", index=False)
    parameters_data_to_df(run).to_csv(case_dir / "parameters.csv", index=False)
