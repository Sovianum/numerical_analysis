from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import plotly.graph_objects as go

from sandwich_numerical.solver.mesh_block import BoundaryType, MeshBlock
from .solver.laplace import set_laplace_update


def set_boundary_conditions_bottom_block(
    state: MeshBlock, grad_vec: np.ndarray, grid_step: float
) -> None:
    """
    Set boundary conditions for the bottom block of the sandwich structure.

    This function enforces three types of boundary conditions:
    1. Fixed boundary: The far end (right side) is clamped to zero
    2. Zero gradient: The bottom side has zero normal derivative (df/dx2 = 0)
    3. Prescribed gradient: The near end (left side) has a known gradient
       from grad_vec
    """

    state.set_boundary_values(BoundaryType.RIGHT, 0)
    state.set_boundary_gradients(BoundaryType.BOTTOM, 0)
    state.set_boundary_gradients(BoundaryType.LEFT, grad_vec * grid_step)


def set_boundary_conditions_top_block(
    state: MeshBlock, grad_vec: np.ndarray, grid_step: float
) -> None:
    """
    Set boundary conditions for the top block of the sandwich structure.

    This function enforces three types of boundary conditions:
    1. Fixed boundary: The far end (right side) is clamped to zero
    2. Zero gradient: The top side has zero normal derivative (df/dx2 = 0)
    3. Prescribed gradient: The near end (left side) has a known gradient
       from grad_vec
    """

    state.set_boundary_values(BoundaryType.RIGHT, 0)
    state.set_boundary_gradients(BoundaryType.TOP, 0)
    state.set_boundary_gradients(BoundaryType.LEFT, grad_vec * grid_step)


def set_boundary_conditions_middle_block(
    state: MeshBlock, grad_vec: np.ndarray, grid_step: float
) -> None:
    """
    Set boundary conditions for a middle block of the sandwich structure.

    This function enforces a fixed far end and a prescribed gradient at the
    near end. Middle blocks do not have zero-gradient top/bottom boundaries.
    """

    state.set_boundary_values(BoundaryType.RIGHT, 0)
    state.set_boundary_gradients(BoundaryType.LEFT, grad_vec * grid_step)


class Sandwich:
    """
    A multi-block numerical solver implementing the generalized Sandwich method.
    """

    def __init__(
        self,
        num_mid_blocks: int,
        block_size: tuple[int, int],
        grad_vec: np.ndarray,
        grid_step: float,
        grad_factors: Sequence[float],
        learning_rate: float = 1.0,
    ) -> None:
        self.block_size = self._validate_block_size(block_size)
        self.block_height = self.block_size[0]
        self.grid_step = self._validate_positive_number(grid_step, "grid_step")
        self.learning_rate = self._validate_positive_number(
            learning_rate, "learning_rate"
        )
        self.num_mid_blocks = self._validate_num_mid_blocks(num_mid_blocks)
        self.total_blocks = 2 + self.num_mid_blocks
        self.grad_factors = self._validate_grad_factors(grad_factors, self.total_blocks)

        self.grad_vec = self._validate_grad_vec(
            grad_vec, self.block_height, self.total_blocks
        )

        self.blocks = [MeshBlock(self.block_size)]
        self.blocks.extend(
            MeshBlock(self._pad_block_size(self.block_size, padding=1))
            for _ in range(self.num_mid_blocks)
        )
        self.blocks.append(MeshBlock(self.block_size))

        self.bottom = self.blocks[0]
        self.top = self.blocks[-1]
        self.mid_blocks = self.blocks[1:-1]
        self.mid = self.mid_blocks[0]

    @staticmethod
    def _validate_block_size(block_size: tuple[int, int]) -> tuple[int, int]:
        if not isinstance(block_size, tuple):
            raise TypeError("block_size must be a tuple (height, width)")

        if len(block_size) != 2:
            raise ValueError("block_size must be 2-dimensional (height, width)")

        height, width = block_size
        if not isinstance(height, int) or not isinstance(width, int):
            raise TypeError("block_size values must be integers")

        if height < 2 or width < 2:
            raise ValueError("block_size values must be at least 2")

        return block_size

    @staticmethod
    def _validate_num_mid_blocks(num_mid_blocks: int) -> int:
        if not isinstance(num_mid_blocks, int):
            raise TypeError("num_mid_blocks must be an integer")

        if num_mid_blocks < 1:
            raise ValueError("Number of mid blocks must be positive")

        if num_mid_blocks % 2 == 0:
            raise ValueError("Number of mid blocks must be odd")

        return num_mid_blocks

    @staticmethod
    def _validate_grad_vec(
        grad_vec: np.ndarray, block_height: int, total_blocks: int
    ) -> np.ndarray:
        grad_array = np.asarray(grad_vec, dtype=float)
        expected_length = total_blocks * block_height

        if grad_array.ndim != 1:
            raise ValueError("grad_vec must be a 1-dimensional array")

        if grad_array.shape != (expected_length,):
            raise ValueError(
                f"grad_vec must have length {expected_length}, "
                f"got {grad_array.shape[0]}"
            )

        return grad_array

    @classmethod
    def _validate_grad_factors(
        cls, grad_factors: Sequence[float], total_blocks: int
    ) -> list[float]:
        factors = [
            cls._validate_positive_number(value, "grad_factors")
            for value in grad_factors
        ]

        if len(factors) != total_blocks:
            raise ValueError(
                f"grad_factors must have length {total_blocks}, got {len(factors)}"
            )

        return factors

    @staticmethod
    def _validate_positive_number(value: float, name: str) -> float:
        numeric_value = float(value)

        if not np.isfinite(numeric_value):
            raise ValueError(f"{name} must be finite")

        if numeric_value <= 0:
            raise ValueError(f"{name} must be positive")

        return numeric_value

    @staticmethod
    def _pad_block_size(block_size: tuple[int, int], padding: int) -> tuple[int, int]:
        return (block_size[0] + 2 * padding, block_size[1])

    def step(self) -> None:
        """
        Execute one complete iteration step of the Sandwich solver.
        """
        self._set_boundary_conditions_generalized()
        self._run_laplace_on_all_blocks()
        self._transfer_saved_gradients_to_adjacent_blocks(
            self._get_gradients_of_all_blocks()
        )
        self._run_laplace_on_all_blocks()
        self._transfer_boundary_values_to_adjacent_blocks(
            self._get_all_boundary_values()
        )

    def plot(self, plot_abs: bool = False) -> go.Figure:
        """
        Generate a heatmap visualization of the current displacement field.
        """
        displacement = self.get_displacement_array()[::-1]

        if plot_abs:
            displacement = np.abs(displacement)

        return go.Figure(data=go.Heatmap(z=displacement))

    def get_displacement_array(self) -> np.ndarray:
        """
        Get the complete displacement field across all blocks.
        """
        displacement_parts = [self.bottom._state]
        displacement_parts.extend(
            mid_block._state[1:-1] for mid_block in self.mid_blocks
        )
        displacement_parts.append(self.top._state)

        return np.concatenate(displacement_parts)

    def get_residual(self) -> float:
        """
        Calculate the residual of the current solution.
        """
        results: list[float] = []
        for block in self.blocks:
            res_block = np.copy(block._state)
            set_laplace_update(res_block)

            results.append(
                float(np.sum(np.abs(block._state[1:-1, 1:-1] - res_block[1:-1, 1:-1])))
            )

        return sum(results)

    def _set_boundary_conditions_generalized(self) -> None:
        set_boundary_conditions_bottom_block(
            self.bottom,
            self._get_block_grad_vec(block_id=0, padding=0),
            self.grid_step,
        )

        for block_id, mid_block in enumerate(self.mid_blocks, start=1):
            set_boundary_conditions_middle_block(
                mid_block,
                self._get_block_grad_vec(block_id=block_id, padding=1),
                self.grid_step,
            )

        set_boundary_conditions_top_block(
            self.top,
            self._get_block_grad_vec(block_id=len(self.blocks) - 1, padding=0),
            self.grid_step,
        )

    def _get_block_grad_vec(self, block_id: int, padding: int) -> np.ndarray:
        left_idx = block_id * self.block_height - padding
        right_idx = (block_id + 1) * self.block_height + padding

        return self.grad_vec[left_idx:right_idx]

    def _run_laplace_on_all_blocks(self) -> None:
        for block in self.blocks:
            set_laplace_update(block._state, self.learning_rate)

    def _get_gradients_of_all_blocks(
        self,
    ) -> dict[int, dict[BoundaryType, np.ndarray]]:
        stored_gradients: dict[int, dict[BoundaryType, np.ndarray]] = {}

        for index, block in enumerate(self.blocks):
            stored_gradients[index] = {}
            for boundary in [BoundaryType.TOP, BoundaryType.BOTTOM]:
                stored_gradients[index][boundary] = block.get_boundary_gradients(
                    boundary
                )

        return stored_gradients

    def _get_all_boundary_values(
        self,
    ) -> dict[int, dict[BoundaryType, np.ndarray]]:
        stored_boundary_values: dict[int, dict[BoundaryType, np.ndarray]] = {}

        for index, block in enumerate(self.blocks):
            stored_boundary_values[index] = {}
            for boundary in [BoundaryType.TOP, BoundaryType.BOTTOM]:
                stored_boundary_values[index][boundary] = block.get_boundary_values(
                    boundary
                )

        return stored_boundary_values

    def _transfer_boundary_values_to_adjacent_blocks(
        self, stored_boundary_values: dict[int, dict[BoundaryType, np.ndarray]]
    ) -> None:
        for index in range(len(self.blocks) - 1):
            self.blocks[index + 1].set_boundary_values(
                BoundaryType.BOTTOM,
                stored_boundary_values[index][BoundaryType.TOP],
            )
            self.blocks[index].set_boundary_values(
                BoundaryType.TOP,
                stored_boundary_values[index + 1][BoundaryType.BOTTOM],
            )

    def _transfer_saved_gradients_to_adjacent_blocks(
        self, stored_gradients: dict[int, dict[BoundaryType, np.ndarray]]
    ) -> None:
        for index in range(len(self.blocks) - 1):
            grad_scale_forward = self._get_grad_scale(index, index + 1)
            if grad_scale_forward <= 1:
                self.blocks[index + 1].set_boundary_gradients(
                    BoundaryType.BOTTOM,
                    stored_gradients[index][BoundaryType.TOP] * grad_scale_forward,
                )

            grad_scale_backward = self._get_grad_scale(index + 1, index)
            if grad_scale_backward <= 1:
                self.blocks[index].set_boundary_gradients(
                    BoundaryType.TOP,
                    stored_gradients[index + 1][BoundaryType.BOTTOM]
                    * grad_scale_backward,
                )

    def _get_grad_scale(self, block_id_curr: int, block_id_next: int) -> float:
        return self.grad_factors[block_id_curr] / self.grad_factors[block_id_next]
