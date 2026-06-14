"""Tests for the FEM Sandwich integration runner."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from sandwich_numerical.fem.integration import (
    create_rectangular_mesh,
    solve_case,
    solve_displacement,
)
from sandwich_numerical.integration import (
    SandwichRun,
    layer_boundary_x2,
    mesh_height,
    sample_x1_coordinates,
)


def small_run(**overrides) -> SandwichRun:
    values = {
        "name": "small",
        "block_height": 4,
        "block_width": 5,
        "grid_step": 0.25,
        "grad_factors": (1.0, 1.0, 1.0),
        "iterations": 0,
        "residual_every": 1,
        "progress_every": 1,
        "heatmap_columns": 5,
        "detail_heatmap_columns": 5,
        "sample_x1_positions": (0.0, 0.5, 1.0),
    }
    values.update(overrides)
    return SandwichRun(**values)


def test_rectangular_mesh_contains_layer_interfaces() -> None:
    run = small_run(grad_factors=(1.0, 2.0, 3.0))

    mesh = create_rectangular_mesh(run)
    x2_coordinates = np.unique(mesh.p[1])

    assert mesh.t.shape[0] == 4
    for boundary in layer_boundary_x2(run):
        assert np.any(np.isclose(x2_coordinates, boundary))
        assert np.any(np.isclose(mesh.p[1], boundary))


def test_fem_solution_uses_fdm_displacement_artifact_shape() -> None:
    run = small_run()

    solution = solve_case(run)
    displacement_values = solution.displacement.drop(
        columns=["x2_mesh_id", "x2_bottom", "x2_top"]
    ).to_numpy()

    assert displacement_values.shape == (mesh_height(run), run.block_width)
    assert list(solution.residuals.columns) == ["iteration", "residual"]
    assert solution.residuals.shape == (1, 2)
    assert list(solution.samples.columns) == ["x2", "x1=0.0", "x1=0.5", "x1=1.0"]


def test_homogeneous_constant_gradient_matches_linear_solution() -> None:
    run = small_run(block_width=6, sample_x1_positions=(0.0,))
    gradient = np.full(mesh_height(run), 2.0)

    displacement, residual = solve_displacement(run, gradient)

    x_right = (run.block_width - 1) * run.grid_step
    expected_row = gradient[0] * (sample_x1_coordinates(run) - x_right)
    expected = np.tile(expected_row, (mesh_height(run), 1))

    np.testing.assert_allclose(displacement, expected, rtol=0, atol=1e-12)
    assert residual == pytest.approx(0.0, abs=1e-12)


def test_fem_cli_writes_csv_artifacts(tmp_path: Path) -> None:
    output_dir = tmp_path / "fem_artifacts"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/fem/run_sandwich_integration.py",
            "--case",
            "grad_factors_1_1_1",
            "--block-width",
            "5",
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
