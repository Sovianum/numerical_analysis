"""Tests for the unified Sandwich artifact pipeline."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from sandwich_numerical import artifacts


def sample_table(scale: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x2": [0.0, 0.5, 1.0],
            "x1=0.0": [0.0, scale, 0.0],
            "x1=0.5": [scale, 0.0, -scale],
        }
    )


def residual_table(value: float) -> pd.DataFrame:
    return pd.DataFrame({"iteration": [0], "residual": [value]})


def write_solver_tables(case_dir: Path) -> None:
    for solver_name, scale in (("fdm", 1.0), ("fem", 0.9)):
        solver_dir = case_dir / solver_name
        solver_dir.mkdir(parents=True)
        sample_table(scale).to_csv(solver_dir / "samples.csv", index=False)
        residual_table(scale).to_csv(solver_dir / "residuals.csv", index=False)


def test_residual_comparison_accepts_single_iteration_point() -> None:
    fig = artifacts.make_solver_residuals_comparison_figure(
        {
            "fdm": residual_table(1.0),
            "fem": residual_table(0.5),
        },
        "residuals",
    )
    try:
        assert len(fig.axes[0].lines) == 2
        assert [line.get_marker() for line in fig.axes[0].lines] == ["o", "s"]
    finally:
        plt.close(fig)


def test_all_case_comparison_writes_expected_pngs(tmp_path: Path) -> None:
    output_dir = tmp_path / "artifacts"
    case_names = artifacts.ci_case_names()[:2]
    for case_name in case_names:
        write_solver_tables(output_dir / case_name)

    artifacts.write_all_cases_comparison_figures(output_dir, case_names=case_names)

    comparison_dir = output_dir / "comparison"
    assert (comparison_dir / "displacement_sections_all_cases.png").exists()
    assert (comparison_dir / "residuals_all_cases.png").exists()


def test_unified_cli_local_csv_smoke(tmp_path: Path) -> None:
    output_dir = tmp_path / "local"

    result = run_artifact_cli(
        "--mode",
        "local",
        "--case",
        "grad_factors_1_1_1_load_sin",
        "--block-width",
        "5",
        "--csv-only",
        "--output-dir",
        str(output_dir),
    )

    assert result.returncode == 0, result.stderr
    case_dir = output_dir / "grad_factors_1_1_1_load_sin"
    for solver_name in artifacts.SOLVER_NAMES:
        assert (case_dir / solver_name / "samples.csv").exists()
        assert (case_dir / solver_name / "displacement.csv").exists()
        assert (case_dir / solver_name / "residuals.csv").exists()
        assert (case_dir / solver_name / "parameters.csv").exists()
    assert (case_dir / "comparison").is_dir()


def test_unified_cli_ci_like_case_writes_comparison_pngs(tmp_path: Path) -> None:
    output_dir = tmp_path / "ci"

    result = run_artifact_cli(
        "--mode",
        "ci",
        "--case",
        "grad_factors_1_1_1_load_sin",
        "--block-width",
        "5",
        "--output-dir",
        str(output_dir),
    )

    assert result.returncode == 0, result.stderr
    comparison_dir = output_dir / "grad_factors_1_1_1_load_sin" / "comparison"
    assert (comparison_dir / "displacement_sections.png").exists()
    assert (comparison_dir / "residuals.png").exists()


def run_artifact_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "scripts/run_sandwich_artifacts.py", *args],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        text=True,
        capture_output=True,
    )
