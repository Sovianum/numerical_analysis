"""Shared Sandwich integration scenario and artifact helpers."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SandwichRun:
    name: str
    block_height: int
    block_width: int
    grid_step: float
    grad_factors: tuple[float, ...]
    iterations: int
    residual_every: int
    progress_every: int
    heatmap_columns: int
    detail_heatmap_columns: int
    sample_x1_positions: tuple[float, ...]
    gradient_profile: str = "sine"
    gradient_relaxation: float = 1.0
    enforce_overlap_continuity: bool = True


@dataclass(frozen=True)
class SandwichSolution:
    residuals: pd.DataFrame
    samples: pd.DataFrame
    displacement: pd.DataFrame


ProgressCallback = Callable[[int, float | None], None]

GRADIENT_PROFILE_SINE = "sine"
GRADIENT_PROFILE_PARABOLIC_ZERO_MEAN = "parabolic_zero_mean"
GRADIENT_PROFILES = (
    GRADIENT_PROFILE_SINE,
    GRADIENT_PROFILE_PARABOLIC_ZERO_MEAN,
)


SANDWICH_RUNS: tuple[SandwichRun, ...] = (
    SandwichRun(
        name="grad_factors_1_1_1",
        block_height=21,
        block_width=3000,
        grid_step=0.005,
        grad_factors=(1.0, 1.0, 1.0),
        iterations=10_000,
        residual_every=1_000,
        progress_every=1_000,
        heatmap_columns=1_000,
        detail_heatmap_columns=100,
        sample_x1_positions=(0.0, 0.2, 0.4, 1.0),
    ),
    SandwichRun(
        name="grad_factors_1_1000_1",
        block_height=21,
        block_width=3000,
        grid_step=0.005,
        grad_factors=(1.0, 1_000.0, 1.0),
        iterations=10_000,
        residual_every=1_000,
        progress_every=1_000,
        heatmap_columns=1_000,
        detail_heatmap_columns=100,
        sample_x1_positions=(0.0, 0.2, 0.4, 1.0),
    ),
    SandwichRun(
        name="grad_factors_1000_1_1000_1_1000",
        block_height=21,
        block_width=3000,
        grid_step=0.005,
        grad_factors=(1_000.0, 1.0, 1_000.0, 1.0, 1_000.0),
        iterations=10_000,
        residual_every=500,
        progress_every=500,
        heatmap_columns=1_000,
        detail_heatmap_columns=100,
        sample_x1_positions=(0.0, 0.2, 0.4, 1.0),
        gradient_relaxation=0.001,
    ),
)


def mesh_height(run: SandwichRun) -> int:
    return run.block_height * len(run.grad_factors)


def sample_x1_coordinates(run: SandwichRun) -> np.ndarray:
    return np.arange(run.block_width, dtype=float) * run.grid_step


def sample_x2_coordinates(run: SandwichRun) -> np.ndarray:
    return np.arange(mesh_height(run), dtype=float) * run.grid_step


def layer_boundary_rows(run: SandwichRun) -> np.ndarray:
    return np.arange(1, len(run.grad_factors), dtype=float) * run.block_height - 0.5


def layer_boundary_x2(run: SandwichRun) -> np.ndarray:
    return layer_boundary_rows(run) * run.grid_step


def build_gradient_vector(run: SandwichRun) -> np.ndarray:
    height = mesh_height(run)

    if run.gradient_profile == GRADIENT_PROFILE_SINE:
        gradient = np.sin((2 * np.pi) / (height - 1) * np.arange(height))
    elif run.gradient_profile == GRADIENT_PROFILE_PARABOLIC_ZERO_MEAN:
        full_thickness = (height - 1) * run.grid_step
        x = np.linspace(-full_thickness / 2, full_thickness / 2, height)
        gradient = x**2 - full_thickness**2 / 12
    else:
        raise ValueError(f"Unsupported gradient profile: {run.gradient_profile}")

    return np.asarray(gradient, dtype=float)


def samples_data_to_df(run: SandwichRun, displacement: np.ndarray) -> pd.DataFrame:
    columns: dict[str, np.ndarray] = {"x2": sample_x2_coordinates(run)}
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
    result = pd.DataFrame([dataclasses.asdict(run)])
    result["grad_factors"] = result["grad_factors"].astype(str)
    return result


def solution_from_displacement(
    run: SandwichRun,
    residuals: pd.DataFrame,
    displacement: np.ndarray,
) -> SandwichSolution:
    return SandwichSolution(
        residuals=residuals,
        samples=samples_data_to_df(run, displacement),
        displacement=displacement_data_to_df(run, displacement),
    )


def write_solution_csvs(
    case_dir: Path, run: SandwichRun, solution: SandwichSolution
) -> None:
    solution.residuals.to_csv(case_dir / "residuals.csv", index=False)
    solution.samples.to_csv(case_dir / "samples.csv", index=False)
    solution.displacement.to_csv(case_dir / "displacement.csv", index=False)
    parameters_data_to_df(run).to_csv(case_dir / "parameters.csv", index=False)
