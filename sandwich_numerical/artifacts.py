"""Shared CLI and plotting helpers for Sandwich integration artifacts."""

from __future__ import annotations

import argparse
import dataclasses
import os
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Protocol

os.environ.setdefault("MPLCONFIGDIR", "/tmp/sandwich_matplotlib_config")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/sandwich_cache")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.colors import Normalize, TwoSlopeNorm  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from sandwich_numerical.integration import (  # noqa: E402
    build_gradient_vector,
    GRADIENT_PROFILE_PARABOLIC_ZERO_MEAN,
    GRADIENT_PROFILE_SINE,
    GRADIENT_PROFILES,
    SANDWICH_RUNS,
    ProgressCallback,
    SandwichRun,
    SandwichSolution,
    layer_boundary_rows,
    layer_boundary_x2,
    write_solution_csvs,
)


class CaseSolver(Protocol):
    def __call__(
        self,
        run: SandwichRun,
        progress_callback: ProgressCallback | None = None,
    ) -> SandwichSolution:
        ...


FIGURE_WIDTH = 1120
FIGURE_HEIGHT = 650
DISPLACEMENT_CMAP = "bwr"
LAYER_GRID_SUBDIVISIONS = 5
LAYER_BOUNDARY_COLOR = "black"
LAYER_BOUNDARY_LINESTYLE = "-"
LAYER_BOUNDARY_LINEWIDTH = 1.4
LAYER_BOUNDARY_ALPHA = 0.9
LAYER_GRID_COLOR = "black"
LAYER_GRID_LINESTYLE = "-"
LAYER_GRID_LINEWIDTH = 0.45
LAYER_GRID_ALPHA = 0.18
LOAD_SHAPE_NAME_BY_GRADIENT_PROFILE = {
    GRADIENT_PROFILE_SINE: "load_sin",
    GRADIENT_PROFILE_PARABOLIC_ZERO_MEAN: "load_parabolic",
}
LOAD_DESCRIPTION_BY_GRADIENT_PROFILE = {
    GRADIENT_PROFILE_SINE: (
        "sin(2*pi*i/(height - 1)); zero at the lower and upper boundaries, "
        "with max absolute value close to 1 on the discrete mesh"
    ),
    GRADIENT_PROFILE_PARABOLIC_ZERO_MEAN: (
        "(x2^2 - L^2/12) normalized by its max absolute value; symmetric, "
        "zero mean, with max absolute value 1"
    ),
}
SOLVER_NAMES = ("fdm", "fem")


def case_name_for_gradient_profile(case_name: str, gradient_profile: str) -> str:
    return f"{case_name}_{LOAD_SHAPE_NAME_BY_GRADIENT_PROFILE[gradient_profile]}"


def load_specific_case_aliases() -> dict[str, tuple[str, str]]:
    return {
        case_name_for_gradient_profile(run.name, gradient_profile): (
            run.name,
            gradient_profile,
        )
        for run in SANDWICH_RUNS
        for gradient_profile in GRADIENT_PROFILES
    }


def ci_case_names() -> tuple[str, ...]:
    return tuple(
        case_name_for_gradient_profile(run.name, gradient_profile)
        for run in SANDWICH_RUNS
        for gradient_profile in GRADIENT_PROFILES
    )


def case_choices() -> tuple[str, ...]:
    base_cases = tuple(run.name for run in SANDWICH_RUNS)
    load_specific_cases = tuple(sorted(load_specific_case_aliases()))
    return base_cases + load_specific_cases


def main(
    solve_case: CaseSolver,
    default_output_dir: Path,
    parser_description: str,
) -> None:
    args = parse_args(default_output_dir, parser_description)
    runs = prepare_runs(args)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Writing artifacts to {output_dir}")
    print(f"Configured sandwiches: {len(runs)}")

    for index, run in enumerate(runs, start=1):
        run_case(
            run,
            index,
            len(runs),
            output_dir,
            solve_case=solve_case,
            write_figures=not args.csv_only,
        )

    print("Done.")


def parse_args(default_output_dir: Path, parser_description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=parser_description)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help="Directory for generated CSV and PNG files.",
    )
    parser.add_argument(
        "--case",
        choices=case_choices(),
        action="append",
        help=(
            "Run only the selected case or load-specific scenario. "
            "Can be passed more than once."
        ),
    )
    parser.add_argument(
        "--iterations",
        type=int,
        help="Override iteration count for all selected sandwiches.",
    )
    parser.add_argument(
        "--block-width",
        type=int,
        help="Override block width for all selected sandwiches; useful for smoke runs.",
    )
    parser.add_argument(
        "--gradient-relaxation",
        type=float,
        help="Under-relaxation for copied interface gradients.",
    )
    parser.add_argument(
        "--gradient-profile",
        choices=GRADIENT_PROFILES,
        help="Override the boundary gradient profile for selected sandwiches.",
    )
    overlap_group = parser.add_mutually_exclusive_group()
    overlap_group.add_argument(
        "--enforce-overlap-continuity",
        dest="enforce_overlap_continuity",
        action="store_true",
        default=None,
        help="Average duplicate real/ghost rows shared by adjacent blocks.",
    )
    overlap_group.add_argument(
        "--no-enforce-overlap-continuity",
        dest="enforce_overlap_continuity",
        action="store_false",
        help="Disable averaging of duplicate real/ghost rows.",
    )
    parser.add_argument(
        "--csv-only",
        action="store_true",
        help="Write CSV data without PNG figures.",
    )
    return parser.parse_args()


def prepare_runs(args: argparse.Namespace) -> tuple[SandwichRun, ...]:
    if args.iterations is not None and args.iterations < 0:
        raise SystemExit("--iterations must be non-negative.")
    if args.block_width is not None and args.block_width < 2:
        raise SystemExit("--block-width must be at least 2.")
    if args.gradient_relaxation is not None:
        if not (0 < args.gradient_relaxation <= 1):
            raise SystemExit("--gradient-relaxation must be in the interval (0, 1].")

    runs_by_name = {run.name: run for run in SANDWICH_RUNS}
    selected_cases = resolve_case_selection(args.case, args.gradient_profile)
    prepared = []
    for run_name, gradient_profile in selected_cases:
        run = runs_by_name[run_name]
        replacements = {}
        if args.iterations is not None:
            replacements["iterations"] = args.iterations
            interval = max(1, min(run.residual_every, args.iterations))
            replacements["residual_every"] = interval
            replacements["progress_every"] = interval
        if args.block_width is not None:
            replacements["block_width"] = args.block_width
            replacements["heatmap_columns"] = min(run.heatmap_columns, args.block_width)
            replacements["detail_heatmap_columns"] = min(
                run.detail_heatmap_columns, args.block_width
            )
        if args.gradient_relaxation is not None:
            replacements["gradient_relaxation"] = args.gradient_relaxation
        if gradient_profile is not None:
            replacements["gradient_profile"] = gradient_profile
            replacements["name"] = case_name_for_gradient_profile(
                run.name, gradient_profile
            )
        if args.enforce_overlap_continuity is not None:
            replacements["enforce_overlap_continuity"] = args.enforce_overlap_continuity
        prepared.append(dataclasses.replace(run, **replacements))

    if not prepared:
        raise SystemExit("No sandwich cases selected.")
    return tuple(prepared)


def resolve_case_selection(
    case_names: Iterable[str] | None, gradient_profile: str | None
) -> tuple[tuple[str, str | None], ...]:
    if not case_names:
        return tuple((run.name, gradient_profile) for run in SANDWICH_RUNS)

    aliases = load_specific_case_aliases()
    selected_cases = []
    for case_name in case_names:
        if case_name not in aliases:
            selected_cases.append((case_name, gradient_profile))
            continue

        base_case_name, case_gradient_profile = aliases[case_name]
        if gradient_profile is not None and gradient_profile != case_gradient_profile:
            raise SystemExit(
                f"--case {case_name!r} implies gradient profile "
                f"{case_gradient_profile!r}, but --gradient-profile "
                f"{gradient_profile!r} was requested."
            )
        selected_cases.append((base_case_name, case_gradient_profile))

    return tuple(selected_cases)


def run_case(
    run: SandwichRun,
    case_index: int,
    case_count: int,
    output_dir: Path,
    solve_case: CaseSolver,
    write_figures: bool = True,
) -> None:
    case_dir = output_dir / run.name
    write_run_readme(
        case_dir,
        run,
        output_lines=(
            "`residuals.csv`, `samples.csv`, `displacement.csv`, and "
            "`parameters.csv` contain this solver's tabular artifacts.",
            "PNG figures are written beside the CSV files when figures are enabled.",
        ),
    )
    run_case_to_dir(
        run,
        case_dir,
        solve_case=solve_case,
        write_figures=write_figures,
        case_index=case_index,
        case_count=case_count,
    )


def run_case_to_dir(
    run: SandwichRun,
    case_dir: Path,
    solve_case: CaseSolver,
    write_figures: bool = True,
    case_index: int = 1,
    case_count: int = 1,
    label: str | None = None,
) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)

    print("")
    print(f"[{case_index}/{case_count}] {label or run.name}")
    print(
        "  block_size="
        f"({run.block_height}, {run.block_width}), "
        f"grid_step={run.grid_step}, "
        f"grad_factors={run.grad_factors}, "
        f"gradient_profile={run.gradient_profile}, "
        f"gradient_relaxation={run.gradient_relaxation}, "
        f"enforce_overlap_continuity={run.enforce_overlap_continuity}, "
        f"iterations={run.iterations}"
    )

    started_at = time.perf_counter()

    solution = solve_case(
        run,
        progress_callback=lambda iteration, residual: report_progress(
            iteration, residual, run.iterations, started_at
        ),
    )

    write_solution_csvs(case_dir, run, solution)
    if write_figures:
        write_solution_figures(case_dir, run, solution)

    print(f"  wrote {case_dir}")


def write_run_readme(
    case_dir: Path,
    run: SandwichRun,
    output_lines: Sequence[str] | None = None,
) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    gradient = build_gradient_vector(run)
    row_count = run.block_height * len(run.grad_factors)
    full_thickness = (row_count - 1) * run.grid_step
    layer_count = len(run.grad_factors)
    if output_lines is None:
        output_lines = (
            "`fdm/` contains finite-difference CSV/PNG artifacts.",
            "`fem/` contains finite-element CSV/PNG artifacts.",
            "`comparison/` contains cross-solver comparison figures.",
        )
    readme = "\n".join(
        [
            f"# {run.name}",
            "",
            "## Scenario",
            "",
            f"- Layers: {layer_count}",
            f"- Rows per layer: {run.block_height}",
            f"- Columns: {run.block_width}",
            f"- Grid step: {run.grid_step:g}",
            f"- Full thickness: {full_thickness:g}",
            f"- Gradient factors by layer: {run.grad_factors}",
            "",
            "## Boundary Load",
            "",
            f"- Profile: {run.gradient_profile}",
            f"- Definition: {LOAD_DESCRIPTION_BY_GRADIENT_PROFILE[run.gradient_profile]}",
            f"- Discrete min/max: {gradient.min():.12g} / {gradient.max():.12g}",
            f"- Discrete max |load|: {np.max(np.abs(gradient)):.12g}",
            "",
            "## Outputs",
            "",
            *(f"- {line}" for line in output_lines),
            "",
        ]
    )
    (case_dir / "README.md").write_text(readme, encoding="utf-8")


def report_progress(
    iteration: int, residual: float | None, total_iterations: int, started_at: float
) -> None:
    elapsed = time.perf_counter() - started_at
    if residual is None:
        print(f"  iter {iteration:>6}/{total_iterations}: {elapsed:6.1f}s")
    else:
        print(
            f"  iter {iteration:>6}/{total_iterations}: "
            f"residual={residual:.6e}, elapsed={elapsed:6.1f}s"
        )


def write_solution_figures(
    case_dir: Path, run: SandwichRun, solution: SandwichSolution
) -> None:
    displacement = solution.displacement.drop(
        columns=["x2_mesh_id", "x2_bottom", "x2_top"]
    ).to_numpy()
    heatmap_columns = min(run.heatmap_columns, displacement.shape[1])
    detail_columns = min(run.detail_heatmap_columns, displacement.shape[1])
    write_figure_png(
        make_heatmap_figure(
            displacement[:, :heatmap_columns],
            f"Displacement {run.name}: first {heatmap_columns} columns",
            layer_boundary_rows(run),
        ),
        case_dir / "displacement_heatmap.png",
    )
    write_figure_png(
        make_heatmap_figure(
            displacement[:, :detail_columns],
            f"Displacement {run.name}: first {detail_columns} columns",
            layer_boundary_rows(run),
        ),
        case_dir / "displacement_detail_heatmap.png",
    )
    write_figure_png(
        make_samples_figure(
            solution.samples,
            f"Samples {run.name}",
            layer_boundary_x2(run),
        ),
        case_dir / "samples.png",
    )


def make_heatmap_figure(
    data: np.ndarray, title: str, layer_boundaries: np.ndarray | None = None
) -> Figure:
    fig, ax = plt.subplots(figsize=(11.2, 6.5))
    image = ax.imshow(
        data,
        aspect="auto",
        cmap=DISPLACEMENT_CMAP,
        norm=make_displacement_norm(data),
        origin="lower",
    )
    ax.set_title(title)
    ax.set_xlabel("x1 column")
    ax.set_ylabel("x2 row")
    add_layer_aligned_y_grid(ax, layer_boundaries)
    add_horizontal_layer_boundaries(ax, layer_boundaries, data.shape[0])
    fig.colorbar(image, ax=ax, label="displacement")
    return fig


def make_samples_figure(
    samples: pd.DataFrame, title: str, layer_boundaries: np.ndarray | None = None
) -> Figure:
    fig, ax = plt.subplots(figsize=(11.2, 6.5))
    for column in samples.columns:
        if column == "x2":
            continue
        ax.plot(samples["x2"], samples[column], label=column)
    add_layer_aligned_x_grid(ax, layer_boundaries)
    add_vertical_layer_boundaries(ax, layer_boundaries)
    ax.set_title(title)
    ax.set_xlabel("x2")
    ax.set_ylabel("displacement")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    return fig


def add_horizontal_layer_boundaries(
    ax: Axes, layer_boundaries: np.ndarray | None, row_count: int
) -> None:
    if layer_boundaries is None:
        return
    for boundary in layer_boundaries:
        if 0 < boundary < row_count - 1:
            ax.axhline(
                boundary,
                color=LAYER_BOUNDARY_COLOR,
                linestyle=LAYER_BOUNDARY_LINESTYLE,
                linewidth=LAYER_BOUNDARY_LINEWIDTH,
                alpha=LAYER_BOUNDARY_ALPHA,
                zorder=4,
            )


def add_vertical_layer_boundaries(
    ax: Axes, layer_boundaries: np.ndarray | None
) -> None:
    if layer_boundaries is None:
        return
    for index, boundary in enumerate(layer_boundaries):
        ax.axvline(
            boundary,
            color=LAYER_BOUNDARY_COLOR,
            linestyle=LAYER_BOUNDARY_LINESTYLE,
            linewidth=LAYER_BOUNDARY_LINEWIDTH,
            alpha=LAYER_BOUNDARY_ALPHA,
            label="layer boundary" if index == 0 else "_nolegend_",
            zorder=4,
        )


def make_residuals_figure(residuals: pd.DataFrame, title: str) -> Figure:
    fig, ax = plt.subplots(figsize=(11.2, 6.5))
    ax.plot(residuals["iteration"], residuals["residual"])
    ax.set_title(title)
    ax.set_xlabel("iteration")
    ax.set_ylabel("residual")
    ax.grid(True, alpha=0.3)
    return fig


def write_case_comparison_figures(
    case_dir: Path, run: SandwichRun, write_figures: bool = True
) -> None:
    comparison_dir = case_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    if not write_figures:
        return

    samples_by_solver = load_solver_csvs(case_dir, "samples.csv")
    write_figure_png(
        make_solver_samples_comparison_figure(
            samples_by_solver,
            f"Displacement sections {run.name}",
            layer_boundary_x2(run),
        ),
        comparison_dir / "displacement_sections.png",
    )


def write_all_cases_comparison_figures(
    output_dir: Path, case_names: Sequence[str] | None = None
) -> None:
    cases = tuple(case_names or discover_case_directories(output_dir))
    if not cases:
        raise FileNotFoundError(f"No case directories found in {output_dir}")

    samples_by_case = {
        case_name: load_solver_csvs(output_dir / case_name, "samples.csv")
        for case_name in cases
    }
    runs_by_case = {case_name: run_for_case_name(case_name) for case_name in cases}

    comparison_dir = output_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    write_figure_png(
        make_all_cases_samples_comparison_figure(samples_by_case, runs_by_case),
        comparison_dir / "displacement_sections_all_cases.png",
        width=1600,
        height=1800,
    )


def discover_case_directories(output_dir: Path) -> tuple[str, ...]:
    known_cases = [
        case_name for case_name in ci_case_names() if (output_dir / case_name).is_dir()
    ]
    if known_cases:
        return tuple(known_cases)
    return tuple(sorted(path.name for path in output_dir.iterdir() if path.is_dir()))


def load_solver_csvs(case_dir: Path, file_name: str) -> dict[str, pd.DataFrame]:
    tables = {}
    for solver_name in SOLVER_NAMES:
        path = case_dir / solver_name / file_name
        if path.exists():
            tables[solver_name] = pd.read_csv(path)
    missing = sorted(set(SOLVER_NAMES) - set(tables))
    if missing:
        raise FileNotFoundError(
            f"Missing {file_name} for solvers {missing} in {case_dir}"
        )
    return tables


def run_for_case_name(case_name: str) -> SandwichRun:
    aliases = load_specific_case_aliases()
    runs_by_name = {run.name: run for run in SANDWICH_RUNS}
    if case_name in aliases:
        base_case_name, gradient_profile = aliases[case_name]
        return dataclasses.replace(
            runs_by_name[base_case_name],
            name=case_name,
            gradient_profile=gradient_profile,
        )
    return runs_by_name[case_name]


def make_solver_samples_comparison_figure(
    samples_by_solver: Mapping[str, pd.DataFrame],
    title: str,
    layer_boundaries: np.ndarray | None = None,
) -> Figure:
    fig, ax = plt.subplots(figsize=(11.2, 6.5))
    plot_samples_comparison(ax, samples_by_solver, layer_boundaries)
    ax.set_title(title)
    ax.legend(ncol=2, fontsize="small")
    return fig


def make_all_cases_samples_comparison_figure(
    samples_by_case: Mapping[str, Mapping[str, pd.DataFrame]],
    runs_by_case: Mapping[str, SandwichRun],
) -> Figure:
    case_names = tuple(samples_by_case)
    fig, axes = make_case_subplots(case_names)
    for ax, case_name in zip(axes, case_names):
        plot_samples_comparison(
            ax,
            samples_by_case[case_name],
            layer_boundary_x2(runs_by_case[case_name]),
        )
        ax.set_title(case_name)
    add_shared_legend(fig, axes)
    return fig


def make_case_subplots(case_names: Sequence[str]) -> tuple[Figure, list[Axes]]:
    row_count = int(np.ceil(len(case_names) / 2))
    fig, axes_grid = plt.subplots(
        row_count,
        2,
        figsize=(16, max(5, row_count * 4.5)),
        squeeze=False,
    )
    axes = axes_grid.ravel()
    for ax in axes[len(case_names) :]:
        ax.set_visible(False)
    return fig, list(axes[: len(case_names)])


def plot_samples_comparison(
    ax: Axes,
    samples_by_solver: Mapping[str, pd.DataFrame],
    layer_boundaries: np.ndarray | None = None,
) -> None:
    linestyles = {"fdm": "-", "fem": "--"}
    for solver_name in SOLVER_NAMES:
        samples = samples_by_solver[solver_name]
        for column in samples.columns:
            if column == "x2":
                continue
            ax.plot(
                samples["x2"],
                samples[column],
                label=f"{solver_name} {column}",
                linestyle=linestyles[solver_name],
            )
    ax.set_xlabel("x2")
    ax.set_ylabel("displacement")
    add_layer_aligned_x_grid(ax, layer_boundaries)
    add_vertical_layer_boundaries(ax, layer_boundaries)
    ax.grid(True, axis="y", alpha=0.3)


def add_layer_aligned_x_grid(ax: Axes, layer_boundaries: np.ndarray | None) -> None:
    ticks = layer_aligned_ticks(layer_boundaries, *ax.get_xlim())
    if ticks.size == 0:
        ax.grid(True, alpha=0.3)
        return

    ax.set_xticks(ticks, minor=True)
    ax.tick_params(axis="x", which="minor", length=0)
    ax.grid(False, axis="x", which="major")
    ax.grid(
        True,
        axis="x",
        which="minor",
        color=LAYER_GRID_COLOR,
        linestyle=LAYER_GRID_LINESTYLE,
        linewidth=LAYER_GRID_LINEWIDTH,
        alpha=LAYER_GRID_ALPHA,
    )


def add_layer_aligned_y_grid(ax: Axes, layer_boundaries: np.ndarray | None) -> None:
    ticks = layer_aligned_ticks(layer_boundaries, *ax.get_ylim())
    if ticks.size == 0:
        return

    ax.set_yticks(ticks, minor=True)
    ax.tick_params(axis="y", which="minor", length=0)
    ax.grid(
        True,
        axis="y",
        which="minor",
        color=LAYER_GRID_COLOR,
        linestyle=LAYER_GRID_LINESTYLE,
        linewidth=LAYER_GRID_LINEWIDTH,
        alpha=LAYER_GRID_ALPHA,
    )


def layer_aligned_ticks(
    layer_boundaries: np.ndarray | None,
    lower: float,
    upper: float,
    subdivisions: int = LAYER_GRID_SUBDIVISIONS,
) -> np.ndarray:
    if layer_boundaries is None or subdivisions <= 0:
        return np.array([], dtype=float)

    boundaries = np.asarray(layer_boundaries, dtype=float)
    boundaries = np.sort(boundaries[np.isfinite(boundaries)])
    if boundaries.size == 0:
        return np.array([], dtype=float)

    axis_min = min(lower, upper)
    axis_max = max(lower, upper)
    layer_thickness = infer_layer_thickness(boundaries, axis_min, axis_max)
    if layer_thickness <= 0:
        return np.array([], dtype=float)

    spacing = layer_thickness / subdivisions
    layers_before = int(np.ceil((boundaries[0] - axis_min) / layer_thickness))
    first_layer_start = boundaries[0] - layers_before * layer_thickness
    count = int(np.ceil((axis_max - first_layer_start) / spacing)) + 1
    ticks = first_layer_start + spacing * np.arange(count + 1, dtype=float)
    tolerance = spacing * 1e-6
    return np.asarray(
        ticks[(axis_min - tolerance <= ticks) & (ticks <= axis_max + tolerance)],
        dtype=float,
    )


def infer_layer_thickness(
    layer_boundaries: np.ndarray, axis_min: float, axis_max: float
) -> float:
    if layer_boundaries.size >= 2:
        return float(np.median(np.diff(layer_boundaries)))

    return float(max(layer_boundaries[0] - axis_min, axis_max - layer_boundaries[0]))


def add_shared_legend(fig: Figure, axes: Sequence[Axes]) -> None:
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize="small")


def make_displacement_norm(data: np.ndarray) -> Normalize | None:
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return None
    min_value = float(finite.min())
    max_value = float(finite.max())
    limit = max(abs(min_value), abs(max_value))
    if limit == 0:
        limit = 1.0
    return TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)


def write_figure_png(
    fig: Figure, path: Path, width: int = FIGURE_WIDTH, height: int = FIGURE_HEIGHT
) -> None:
    fig.set_size_inches(width / 100, height / 100)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
