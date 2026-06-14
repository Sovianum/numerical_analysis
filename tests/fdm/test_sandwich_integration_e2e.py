"""End-to-end regression checks for the scripted Sandwich integration runs."""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pytest

from sandwich_numerical.integration import (
    GRADIENT_PROFILE_SINE,
    GRADIENT_PROFILE_PARABOLIC_ZERO_MEAN,
    SandwichRun,
    build_gradient_vector,
)
from scripts.fdm.run_sandwich_integration import (
    DISPLACEMENT_CMAP,
    layer_boundary_rows,
    layer_boundary_x2,
    make_displacement_norm,
    make_heatmap_figure,
    prepare_runs,
)


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


@pytest.mark.parametrize(
    ("gradient_profile", "expected_name"),
    [
        (GRADIENT_PROFILE_SINE, "grad_factors_1_1_1_load_sin"),
        (
            GRADIENT_PROFILE_PARABOLIC_ZERO_MEAN,
            "grad_factors_1_1_1_load_parabolic",
        ),
    ],
)
def test_prepare_runs_adds_load_shape_to_name_for_explicit_profile(
    gradient_profile: str, expected_name: str
) -> None:
    args = argparse.Namespace(
        case=["grad_factors_1_1_1"],
        iterations=None,
        block_width=None,
        gradient_relaxation=None,
        gradient_profile=gradient_profile,
        enforce_overlap_continuity=None,
    )

    (run,) = prepare_runs(args)

    assert run.name == expected_name


@pytest.mark.parametrize(
    ("case_name", "expected_gradient_profile"),
    [
        ("grad_factors_1_1_1_load_sin", GRADIENT_PROFILE_SINE),
        (
            "grad_factors_1_1_1_load_parabolic",
            GRADIENT_PROFILE_PARABOLIC_ZERO_MEAN,
        ),
    ],
)
def test_prepare_runs_accepts_load_specific_case_names(
    case_name: str, expected_gradient_profile: str
) -> None:
    args = argparse.Namespace(
        case=[case_name],
        iterations=None,
        block_width=None,
        gradient_relaxation=None,
        gradient_profile=None,
        enforce_overlap_continuity=None,
    )

    (run,) = prepare_runs(args)

    assert run.name == case_name
    assert run.gradient_profile == expected_gradient_profile


def test_prepare_runs_rejects_mismatched_load_specific_case_profile() -> None:
    args = argparse.Namespace(
        case=["grad_factors_1_1_1_load_sin"],
        iterations=None,
        block_width=None,
        gradient_relaxation=None,
        gradient_profile=GRADIENT_PROFILE_PARABOLIC_ZERO_MEAN,
        enforce_overlap_continuity=None,
    )

    with pytest.raises(SystemExit, match="implies gradient profile"):
        prepare_runs(args)


@pytest.mark.parametrize(
    "data",
    [
        np.array([[-2.0, 0.0, 1.0]]),
        np.array([[0.0, 1.0, 2.0]]),
        np.array([[-2.0, -1.0, 0.0]]),
        np.zeros((2, 3)),
    ],
)
def test_displacement_heatmap_norm_centers_zero(data: np.ndarray) -> None:
    norm = make_displacement_norm(data)

    assert norm is not None
    assert norm(0.0) == pytest.approx(0.5)
    assert norm.vmin == pytest.approx(-norm.vmax)
    assert norm.vcenter == 0


def test_displacement_heatmap_uses_white_center_colormap() -> None:
    fig = make_heatmap_figure(np.array([[-1.0, 0.0, 1.0]]), "displacement")
    try:
        image = fig.axes[0].images[0]

        assert image.cmap.name == DISPLACEMENT_CMAP
        assert image.cmap(image.norm(0.0))[:3] == pytest.approx(
            (1.0, 1.0, 1.0),
            abs=0.01,
        )
    finally:
        plt.close(fig)
