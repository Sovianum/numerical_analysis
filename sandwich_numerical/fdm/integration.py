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

from .sparse import solve_displacement as solve_sparse_displacement


def solve_case(
    run: SandwichRun,
    progress_callback: ProgressCallback | None = None,
) -> SandwichSolution:
    displacement, residual = solve_sparse_displacement(run, build_gradient_vector(run))
    if progress_callback is not None:
        progress_callback(0, residual)

    residuals = pd.DataFrame([{"iteration": 0, "residual": residual}])
    return solution_from_displacement(run, residuals, displacement)
