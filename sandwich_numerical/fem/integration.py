"""Reusable FEM Sandwich integration scenario runner."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from skfem import (
    Basis,
    BilinearForm,
    ElementQuad1,
    FacetBasis,
    LinearForm,
    MeshQuad,
    asm,
    condense,
    solve,
)
from skfem.helpers import dot, grad

from sandwich_numerical.integration import (
    ProgressCallback,
    SandwichRun,
    SandwichSolution,
    build_gradient_vector,
    layer_boundary_x2,
    sample_x1_coordinates,
    sample_x2_coordinates,
    solution_from_displacement,
)


COORDINATE_KEY_DECIMALS = 12


@BilinearForm
def stiffness(u: Any, v: Any, w: dict[str, Any]) -> Any:
    return w["coefficient"] * dot(grad(u), grad(v))


@LinearForm
def left_neumann(v: Any, w: dict[str, Any]) -> Any:
    return -w["coefficient"] * w["left_gradient"] * v


def solve_case(
    run: SandwichRun, progress_callback: ProgressCallback | None = None
) -> SandwichSolution:
    displacement, residual = solve_displacement(run, build_gradient_vector(run))
    if progress_callback is not None:
        progress_callback(0, residual)

    residuals = pd.DataFrame([{"iteration": 0, "residual": residual}])
    return solution_from_displacement(run, residuals, displacement)


def solve_displacement(
    run: SandwichRun, gradient_vector: np.ndarray
) -> tuple[np.ndarray, float]:
    basis = Basis(create_rectangular_mesh(run), ElementQuad1())

    A = assemble_stiffness(run, basis)
    b = assemble_left_boundary_load(run, basis, gradient_vector)
    right_dofs = right_boundary_dofs(run, basis)

    solution = solve(*condense(A, b, D=right_dofs))
    residual = algebraic_residual_norm(A, b, solution, constrained_dofs=right_dofs)
    return sample_solution_on_grid(run, basis, solution), residual


def create_rectangular_mesh(run: SandwichRun) -> MeshQuad:
    x2 = np.concatenate((sample_x2_coordinates(run), layer_boundary_x2(run)))
    return MeshQuad.init_tensor(sample_x1_coordinates(run), np.unique(np.sort(x2)))


def assemble_stiffness(run: SandwichRun, basis: Basis) -> Any:
    return asm(
        stiffness,
        basis,
        coefficient=layer_coefficient(run, basis.global_coordinates()[1]),
    )


def assemble_left_boundary_load(
    run: SandwichRun, basis: Basis, gradient_vector: np.ndarray
) -> np.ndarray:
    left_basis = FacetBasis(
        basis.mesh,
        basis.elem,
        facets=basis.mesh.facets_satisfying(
            lambda x: np.isclose(x[0], 0.0),
            boundaries_only=True,
        ),
    )
    x2 = left_basis.global_coordinates()[1]
    return np.asarray(
        asm(
            left_neumann,
            left_basis,
            coefficient=layer_coefficient(run, x2),
            left_gradient=left_boundary_gradient(run, gradient_vector, x2),
        ),
        dtype=float,
    )


def right_boundary_dofs(run: SandwichRun, basis: Basis) -> np.ndarray:
    x_right = (run.block_width - 1) * run.grid_step
    return np.asarray(
        basis.get_dofs(lambda x: np.isclose(x[0], x_right)).flatten(),
        dtype=np.int64,
    )


def layer_coefficient(run: SandwichRun, x2: np.ndarray) -> np.ndarray:
    boundaries = layer_boundary_x2(run)
    layer_ids = np.searchsorted(boundaries, x2, side="right")
    factors = np.asarray(run.grad_factors, dtype=float)
    return 1.0 / factors[layer_ids]


def left_boundary_gradient(
    run: SandwichRun, gradient_vector: np.ndarray, x2: np.ndarray
) -> np.ndarray:
    return np.interp(x2, sample_x2_coordinates(run), gradient_vector)


def algebraic_residual_norm(
    A: Any,
    b: np.ndarray,
    solution: np.ndarray,
    constrained_dofs: np.ndarray,
) -> float:
    residual = A @ solution - b
    all_dofs = np.arange(A.shape[0])
    free_dofs = np.setdiff1d(all_dofs, constrained_dofs, assume_unique=False)
    return float(np.linalg.norm(residual[free_dofs]))


def sample_solution_on_grid(
    run: SandwichRun, basis: Basis, solution: np.ndarray
) -> np.ndarray:
    values_by_coordinate = {
        coordinate_key(x1, x2): value
        for (x1, x2), value in zip(basis.doflocs.T, solution)
    }

    x1_coordinates = sample_x1_coordinates(run)
    x2_coordinates = sample_x2_coordinates(run)
    displacement = np.empty((x2_coordinates.size, x1_coordinates.size), dtype=float)

    for row_index, x2 in enumerate(x2_coordinates):
        for column_index, x1 in enumerate(x1_coordinates):
            displacement[row_index, column_index] = values_by_coordinate[
                coordinate_key(x1, x2)
            ]

    return displacement


def coordinate_key(x1: float, x2: float) -> tuple[float, float]:
    return (
        round(float(x1), COORDINATE_KEY_DECIMALS),
        round(float(x2), COORDINATE_KEY_DECIMALS),
    )
