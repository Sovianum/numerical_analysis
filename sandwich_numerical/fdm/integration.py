"""Reusable FDM Sandwich integration scenario runner."""

from __future__ import annotations

import pandas as pd

from sandwich_numerical.integration import (
    ProgressCallback,
    SandwichRun,
    SandwichSolution,
    build_gradient_vector,
    solution_from_displacement,
)

from .sandwich import Sandwich


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

    residuals = pd.DataFrame(residual_rows)
    return solution_from_displacement(run, residuals, mesh.get_displacement_array())


def create_sandwich(run: SandwichRun) -> Sandwich:
    return Sandwich(
        num_mid_blocks=len(run.grad_factors) - 2,
        block_size=(run.block_height, run.block_width),
        grad_vec=build_gradient_vector(run),
        grid_step=run.grid_step,
        grad_factors=run.grad_factors,
        gradient_relaxation=run.gradient_relaxation,
        enforce_overlap_continuity=run.enforce_overlap_continuity,
    )
