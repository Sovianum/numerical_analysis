"""Finite-difference implementation of the Sandwich numerical method."""

from .sandwich import (
    Sandwich,
    set_boundary_conditions_bottom_block,
    set_boundary_conditions_middle_block,
    set_boundary_conditions_top_block,
)

__all__ = [
    "Sandwich",
    "set_boundary_conditions_bottom_block",
    "set_boundary_conditions_middle_block",
    "set_boundary_conditions_top_block",
]
