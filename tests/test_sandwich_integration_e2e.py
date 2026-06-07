"""End-to-end regression checks for the scripted Sandwich integration runs."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sandwich_numerical.integration import (
    GRADIENT_PROFILE_PARABOLIC_ZERO_MEAN,
    SANDWICH_RUNS,
    SandwichRun,
    build_gradient_vector,
    parameters_data_to_df,
    solve_case,
)
from scripts.run_sandwich_integration import layer_boundary_rows, layer_boundary_x2


BASELINE_DIR = Path(__file__).resolve().parent / "fixtures/sandwich_integration"
RTOL = 1e-12
ATOL = 1e-14


@pytest.mark.parametrize("run", SANDWICH_RUNS, ids=lambda run: run.name)
def test_sandwich_integration_matches_baseline(run: SandwichRun) -> None:
    solution = solve_case(run)
    case_dir = BASELINE_DIR / run.name

    assert_numeric_frame_matches_csv(solution.residuals, case_dir / "residuals.csv")
    assert_numeric_frame_matches_csv(solution.samples, case_dir / "samples.csv")
    assert_numeric_frame_matches_csv(
        solution.displacement, case_dir / "displacement.csv"
    )
    assert_parameters_match_csv(run, case_dir / "parameters.csv")


def test_layer_boundary_sample_coordinates_match_discrete_row_boundaries() -> None:
    run = SandwichRun(
        name="five_layer",
        block_height=21,
        block_width=3000,
        grid_step=0.005,
        grad_factors=(1000.0, 1.0, 1000.0, 1.0, 1000.0),
        iterations=0,
        residual_every=1,
        progress_every=1,
        heatmap_columns=100,
        detail_heatmap_columns=10,
        sample_x1_positions=(0.0,),
    )

    assert layer_boundary_rows(run).tolist() == [20.5, 41.5, 62.5, 83.5]
    assert layer_boundary_x2(run).tolist() == pytest.approx(
        [0.1025, 0.2075, 0.3125, 0.4175]
    )


def test_parabolic_zero_mean_gradient_profile_matches_formula() -> None:
    run = SandwichRun(
        name="parabolic_zero_mean",
        block_height=7,
        block_width=10,
        grid_step=0.25,
        grad_factors=(1.0, 1.0, 1.0),
        iterations=0,
        residual_every=1,
        progress_every=1,
        heatmap_columns=10,
        detail_heatmap_columns=10,
        sample_x1_positions=(0.0,),
        gradient_profile=GRADIENT_PROFILE_PARABOLIC_ZERO_MEAN,
    )

    actual = build_gradient_vector(run)
    mesh_height = run.block_height * len(run.grad_factors)
    full_thickness = (mesh_height - 1) * run.grid_step
    x = np.linspace(-full_thickness / 2, full_thickness / 2, mesh_height)
    expected = x**2 - full_thickness**2 / 12

    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-15)
    np.testing.assert_allclose(actual, actual[::-1], rtol=0, atol=1e-15)
    analytic_integral = (
        full_thickness**3 / 12 - full_thickness * full_thickness**2 / 12
    )
    assert analytic_integral == pytest.approx(0.0, abs=1e-15)


def assert_numeric_frame_matches_csv(actual: pd.DataFrame, path: Path) -> None:
    assert path.exists(), f"Missing baseline CSV: {path}"
    expected = pd.read_csv(path)

    assert list(actual.columns) == list(expected.columns)
    assert actual.shape == expected.shape
    pd.testing.assert_frame_equal(
        actual.reset_index(drop=True),
        expected,
        check_dtype=False,
        check_exact=False,
        rtol=RTOL,
        atol=ATOL,
    )


def assert_parameters_match_csv(run: SandwichRun, path: Path) -> None:
    assert path.exists(), f"Missing baseline CSV: {path}"
    expected = pd.read_csv(path)
    actual = parameters_data_to_df(run)
    actual["sample_x1_positions"] = actual["sample_x1_positions"].astype(str)
    actual["grad_factors"] = actual["grad_factors"].astype(str)

    assert list(actual.columns) == list(expected.columns)
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False)
