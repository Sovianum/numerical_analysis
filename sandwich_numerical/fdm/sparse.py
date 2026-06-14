"""Sparse global finite-difference solver for the Sandwich problem."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse import linalg

from sandwich_numerical.integration import SandwichRun, mesh_height


@dataclass(frozen=True)
class SparseSystem:
    matrix: sparse.csr_matrix
    rhs: np.ndarray
    constrained_dofs: np.ndarray


def solve_displacement(
    run: SandwichRun, gradient_vector: np.ndarray
) -> tuple[np.ndarray, float]:
    system = assemble_system(run, gradient_vector)
    solution = linalg.spsolve(system.matrix, system.rhs)
    residual = algebraic_residual_norm(
        system.matrix,
        system.rhs,
        np.asarray(solution, dtype=float),
        constrained_dofs=system.constrained_dofs,
    )
    displacement = np.asarray(solution, dtype=float).reshape(
        mesh_height(run), run.block_width
    )
    return displacement, residual


def assemble_system(run: SandwichRun, gradient_vector: np.ndarray) -> SparseSystem:
    height = mesh_height(run)
    width = run.block_width
    gradient = validate_gradient_vector(gradient_vector, height)
    coefficients = layer_coefficients(run)
    row_count = height * width
    matrix = sparse.lil_matrix((row_count, row_count), dtype=float)
    rhs = np.zeros(row_count, dtype=float)
    h_square = run.grid_step * run.grid_step

    for row in range(height):
        for column in range(width):
            dof = node_index(row, column, width)
            if column == width - 1:
                matrix[dof, dof] = 1.0
                continue

            diagonal = 0.0
            center_coefficient = coefficients[row]

            if column == 0:
                rhs[dof] -= center_coefficient * gradient[row] / run.grid_step
            else:
                west = center_coefficient / h_square
                diagonal += west
                matrix[dof, node_index(row, column - 1, width)] = -west

            east = center_coefficient / h_square
            diagonal += east
            matrix[dof, node_index(row, column + 1, width)] = -east

            if row > 0:
                south = (
                    harmonic_mean(center_coefficient, coefficients[row - 1]) / h_square
                )
                # Physical top/bottom rows have half-height control volumes.
                if row == height - 1:
                    south *= 2.0
                diagonal += south
                matrix[dof, node_index(row - 1, column, width)] = -south

            if row < height - 1:
                north = (
                    harmonic_mean(center_coefficient, coefficients[row + 1]) / h_square
                )
                # Physical top/bottom rows have half-height control volumes.
                if row == 0:
                    north *= 2.0
                diagonal += north
                matrix[dof, node_index(row + 1, column, width)] = -north

            matrix[dof, dof] = diagonal

    constrained = np.asarray(
        [node_index(row, width - 1, width) for row in range(height)],
        dtype=np.int64,
    )
    return SparseSystem(matrix=matrix.tocsr(), rhs=rhs, constrained_dofs=constrained)


def validate_gradient_vector(gradient_vector: np.ndarray, height: int) -> np.ndarray:
    gradient = np.asarray(gradient_vector, dtype=float)
    if gradient.ndim != 1 or gradient.shape != (height,):
        raise ValueError(
            f"gradient_vector must have shape ({height},), got {gradient.shape}"
        )
    return gradient


def layer_coefficients(run: SandwichRun) -> np.ndarray:
    factors = np.asarray(run.grad_factors, dtype=float)
    if np.any(~np.isfinite(factors)) or np.any(factors <= 0):
        raise ValueError("grad_factors must be finite positive values")
    return np.repeat(1.0 / factors, run.block_height)


def harmonic_mean(left: float, right: float) -> float:
    if left <= 0 or right <= 0:
        raise ValueError("coefficients must be positive")
    return 2.0 * left * right / (left + right)


def node_index(row: int, column: int, width: int) -> int:
    return row * width + column


def algebraic_residual_norm(
    matrix: sparse.spmatrix,
    rhs: np.ndarray,
    solution: np.ndarray,
    constrained_dofs: np.ndarray,
) -> float:
    residual = matrix @ solution - rhs
    all_dofs = np.arange(matrix.shape[0])
    free_dofs = np.setdiff1d(all_dofs, constrained_dofs, assume_unique=False)
    return float(np.linalg.norm(residual[free_dofs]))
