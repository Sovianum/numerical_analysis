"""End-to-end regression checks for the scripted Sandwich integration runs."""

from pathlib import Path

import pandas as pd
import pytest

from sandwich_numerical.integration import (
    SANDWICH_RUNS,
    SandwichRun,
    parameters_data_to_df,
    solve_case,
)


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

    assert list(actual.columns) == list(expected.columns)
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False)
