"""Finite-difference implementation of the Sandwich numerical method."""

from .sandwich import (  # noqa: F401
    Sandwich,
    set_boundary_conditions_bottom_block,
    set_boundary_conditions_middle_block,
    set_boundary_conditions_top_block,
)
