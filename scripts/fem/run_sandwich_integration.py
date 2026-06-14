#!/usr/bin/env python3
"""Run the FEM Sandwich integration scenarios and save reproducible artifacts."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sandwich_numerical.artifacts import main as run_artifact_main  # noqa: E402
from sandwich_numerical.fem.integration import solve_case  # noqa: E402


def main() -> None:
    run_artifact_main(
        solve_case,
        default_output_dir=Path("artifacts/sandwich_fem"),
        parser_description=(
            "Run configured FEM Sandwich integration scenarios and save "
            "CSV/PNG artifacts."
        ),
    )


if __name__ == "__main__":
    main()
