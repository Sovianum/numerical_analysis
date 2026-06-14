"""Tests for the unified Sandwich artifact pipeline."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

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


def test_all_case_comparison_writes_only_displacement_png(tmp_path: Path) -> None:
    output_dir = tmp_path / "artifacts"
    case_names = artifacts.ci_case_names()[:2]
    for case_name in case_names:
        write_solver_tables(output_dir / case_name)

    artifacts.write_all_cases_comparison_figures(output_dir, case_names=case_names)

    comparison_dir = output_dir / "comparison"
    assert (comparison_dir / "displacement_sections_all_cases.png").exists()
    assert not (comparison_dir / "residuals_all_cases.png").exists()


def test_layer_aligned_ticks_subdivide_layer_thickness() -> None:
    ticks = artifacts.layer_aligned_ticks(
        np.array([0.1025, 0.2075]),
        lower=0.0,
        upper=0.31,
    )

    np.testing.assert_allclose(np.diff(ticks), 0.105 / 5, rtol=0, atol=1e-14)
    assert ticks.tolist() == pytest.approx(
        [
            0.0185,
            0.0395,
            0.0605,
            0.0815,
            0.1025,
            0.1235,
            0.1445,
            0.1655,
            0.1865,
            0.2075,
            0.2285,
            0.2495,
            0.2705,
            0.2915,
        ]
    )


def test_samples_figure_uses_layer_aligned_grid_and_bright_boundaries() -> None:
    fig = artifacts.make_samples_figure(
        sample_table(1.0),
        "samples",
        layer_boundaries=np.array([0.5]),
    )
    try:
        boundary = fig.axes[0].lines[-1]

        assert boundary.get_label() == "layer boundary"
        assert boundary.get_linestyle() == "-"
        assert boundary.get_linewidth() == pytest.approx(1.4)
        assert boundary.get_alpha() == pytest.approx(0.9)
        assert fig.axes[0].get_xticks(minor=True).size > 0
    finally:
        plt.close(fig)


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
    readme = case_dir / "README.md"
    assert readme.exists()
    assert "Layers: 3" in readme.read_text(encoding="utf-8")
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
    assert not (comparison_dir / "residuals.png").exists()

    for solver_name in artifacts.SOLVER_NAMES:
        solver_dir = output_dir / "grad_factors_1_1_1_load_sin" / solver_name
        assert (solver_dir / "residuals.csv").exists()
        assert not (solver_dir / "residuals.png").exists()


def run_artifact_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "scripts/run_sandwich_artifacts.py", *args],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        text=True,
        capture_output=True,
    )
