#!/usr/bin/env python3
"""Run the FDM Sandwich integration scenarios and save reproducible artifacts."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sandwich_numerical import artifacts as _artifacts  # noqa: E402
from sandwich_numerical.fdm.integration import solve_case  # noqa: E402


DISPLACEMENT_CMAP = _artifacts.DISPLACEMENT_CMAP
FIGURE_HEIGHT = _artifacts.FIGURE_HEIGHT
FIGURE_WIDTH = _artifacts.FIGURE_WIDTH
add_horizontal_layer_boundaries = _artifacts.add_horizontal_layer_boundaries
add_vertical_layer_boundaries = _artifacts.add_vertical_layer_boundaries
case_choices = _artifacts.case_choices
case_name_for_gradient_profile = _artifacts.case_name_for_gradient_profile
layer_boundary_rows = _artifacts.layer_boundary_rows
layer_boundary_x2 = _artifacts.layer_boundary_x2
load_specific_case_aliases = _artifacts.load_specific_case_aliases
make_displacement_norm = _artifacts.make_displacement_norm
make_heatmap_figure = _artifacts.make_heatmap_figure
make_residuals_figure = _artifacts.make_residuals_figure
make_samples_figure = _artifacts.make_samples_figure
prepare_runs = _artifacts.prepare_runs
report_progress = _artifacts.report_progress
resolve_case_selection = _artifacts.resolve_case_selection
write_figure_png = _artifacts.write_figure_png
write_solution_figures = _artifacts.write_solution_figures


def parse_args():
    return _artifacts.parse_args(
        Path("artifacts/sandwich_integration"),
        (
            "Run configured FDM Sandwich integration scenarios and save "
            "CSV/PNG artifacts."
        ),
    )


def run_case(
    run,
    case_index: int,
    case_count: int,
    output_dir: Path,
    write_figures: bool = True,
) -> None:
    _artifacts.run_case(
        run,
        case_index,
        case_count,
        output_dir,
        solve_case=solve_case,
        write_figures=write_figures,
    )


def main() -> None:
    _artifacts.main(
        solve_case,
        default_output_dir=Path("artifacts/sandwich_integration"),
        parser_description=(
            "Run configured FDM Sandwich integration scenarios and save "
            "CSV/PNG artifacts."
        ),
    )


if __name__ == "__main__":
    main()
