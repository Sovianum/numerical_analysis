"""Tests for the sparse global FDM Sandwich solver."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from sandwich_numerical.fdm.integration import solve_case
from sandwich_numerical.fdm.sparse import (
    assemble_system,
    node_index,
    solve_displacement,
)
from sandwich_numerical.integration import (
    SandwichRun,
    mesh_height,
    sample_x1_coordinates,
)


def small_run(**overrides) -> SandwichRun:
    values = {
        "name": "small",
        "block_height": 4,
        "block_width": 6,
        "grid_step": 0.25,
        "grad_factors": (1.0, 1.0, 1.0),
        "iterations": 2,
        "residual_every": 1,
        "progress_every": 1,
        "heatmap_columns": 6,
        "detail_heatmap_columns": 6,
        "sample_x1_positions": (0.0, 0.5, 1.0),
    }
    values.update(overrides)
    return SandwichRun(**values)


def test_sparse_matrix_has_expected_size() -> None:
    run = small_run()
    gradient = np.zeros(mesh_height(run))

    system = assemble_system(run, gradient)

    expected_size = mesh_height(run) * run.block_width
    assert system.matrix.shape == (expected_size, expected_size)
    assert system.rhs.shape == (expected_size,)
    assert system.constrained_dofs.shape == (mesh_height(run),)


def test_top_and_bottom_rows_use_half_control_volume_stencil() -> None:
    run = small_run(grad_factors=(1.0, 1.0, 1.0))
    gradient = np.zeros(mesh_height(run))

    system = assemble_system(run, gradient)
    matrix = system.matrix
    width = run.block_width
    h_square = run.grid_step * run.grid_step
    bottom = node_index(0, 1, width)
    bottom_neighbor = node_index(1, 1, width)
    top = node_index(mesh_height(run) - 1, 1, width)
    top_neighbor = node_index(mesh_height(run) - 2, 1, width)

    assert matrix[bottom, bottom_neighbor] == pytest.approx(-2.0 / h_square)
    assert matrix[top, top_neighbor] == pytest.approx(-2.0 / h_square)


def test_sparse_displacement_has_artifact_grid_shape() -> None:
    run = small_run(grad_factors=(1.0, 2.0, 3.0))
    gradient = np.linspace(-1.0, 1.0, mesh_height(run))

    displacement, residual = solve_displacement(run, gradient)

    assert displacement.shape == (mesh_height(run), run.block_width)
    assert residual == pytest.approx(0.0, abs=1e-10)


def test_homogeneous_constant_gradient_matches_linear_solution() -> None:
    run = small_run(block_width=7, sample_x1_positions=(0.0,))
    gradient = np.full(mesh_height(run), 2.0)

    displacement, residual = solve_displacement(run, gradient)

    x_right = (run.block_width - 1) * run.grid_step
    expected_row = gradient[0] * (sample_x1_coordinates(run) - x_right)
    expected = np.tile(expected_row, (mesh_height(run), 1))

    np.testing.assert_allclose(displacement, expected, rtol=0, atol=1e-12)
    assert residual == pytest.approx(0.0, abs=1e-12)


def test_zero_neumann_top_and_bottom_keeps_constant_rows_constant() -> None:
    run = small_run(block_width=7)
    gradient = np.full(mesh_height(run), -1.5)

    displacement, _ = solve_displacement(run, gradient)

    expected = np.tile(displacement[0, :], (mesh_height(run), 1))
    np.testing.assert_allclose(displacement, expected, rtol=0, atol=1e-12)


def test_sparse_case_uses_existing_artifact_schema() -> None:
    run = small_run()

    solution = solve_case(run)

    displacement_values = solution.displacement.drop(
        columns=["x2_mesh_id", "x2_bottom", "x2_top"]
    ).to_numpy()
    assert displacement_values.shape == (mesh_height(run), run.block_width)
    assert list(solution.residuals.columns) == ["iteration", "residual"]
    assert solution.residuals.shape == (1, 2)
    assert list(solution.samples.columns) == ["x2", "x1=0.0", "x1=0.5", "x1=1.0"]


def test_fdm_cli_writes_csv_artifacts(tmp_path: Path) -> None:
    output_dir = tmp_path / "sparse"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/fdm/run_sandwich_integration.py",
            "--case",
            "grad_factors_1_1_1",
            "--block-width",
            "5",
            "--iterations",
            "1",
            "--csv-only",
            "--output-dir",
            str(output_dir),
        ],
        cwd=Path(__file__).resolve().parents[2],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    case_dir = output_dir / "grad_factors_1_1_1"
    assert (case_dir / "residuals.csv").exists()
    assert (case_dir / "samples.csv").exists()
    assert (case_dir / "displacement.csv").exists()
    assert (case_dir / "parameters.csv").exists()
