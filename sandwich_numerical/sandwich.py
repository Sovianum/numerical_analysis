import numpy as np
import plotly.graph_objects as go
from sandwich_numerical.solver.mesh_block import BoundaryType, MeshBlock
from sandwich_numerical.solver.mesh_utils import (
    copy_boundary_gradients,
    copy_boundary_values,
)

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

    Args:
        state (MeshBlock): Current state mesh block
        grad_vec (np.ndarray): Gradient vector specifying the prescribed gradient
            at the near end
        grid_step (float): Grid spacing for finite difference calculations

    Note:
        The function modifies state in-place by setting boundary values.
    """

    state.set_boundary_values(
        BoundaryType.RIGHT, 0
    )  # the block is fixed on the far end
    state.set_boundary_gradients(BoundaryType.BOTTOM, 0)  # df/dx2 = 0

    # gradients are known on the near end
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

    Args:
        state (MeshBlock): Current state mesh block
        grad_vec (np.ndarray): Gradient vector specifying the prescribed gradient
            at the near end
        grid_step (float): Grid spacing for finite difference calculations

    Note:
        The function modifies state in-place by setting boundary values.
    """

    state.set_boundary_values(
        BoundaryType.RIGHT, 0
    )  # the block is fixed on the far end
    state.set_boundary_gradients(BoundaryType.TOP, 0)  # df/dx2 = 0

    # gradients are known on the near end
    state.set_boundary_gradients(BoundaryType.LEFT, grad_vec * grid_step)


def set_boundary_conditions_middle_block(
    state: MeshBlock, grad_vec: np.ndarray, grid_step: float
) -> None:
    """
    Set boundary conditions for the middle block of the sandwich structure.

    This function enforces two types of boundary conditions:
    1. Fixed boundary: The far end (right side) is clamped to zero
    2. Prescribed gradient: The near end (left side) has a known gradient
       from grad_vec

    Args:
        state (MeshBlock): Current state mesh block
        grad_vec (np.ndarray): Gradient vector specifying the prescribed gradient
            at the near end
        grid_step (float): Grid spacing for finite difference calculations

    Note:
        The function modifies next_state in-place by setting boundary values.
        The middle block has no zero-gradient conditions on top/bottom sides.
    """

    state.set_boundary_values(
        BoundaryType.RIGHT, 0
    )  # the block is fixed on the far end

    # gradients are known on the near end
    state.set_boundary_gradients(BoundaryType.LEFT, grad_vec * grid_step)


class Sandwich:
    """
    A multi-block numerical solver implementing the Sandwich method.

    The Sandwich method divides the computational domain into three blocks
    (bottom, middle, top) and solves them iteratively with data transfer between
    blocks. This approach allows for efficient solution of large-scale problems
    by breaking them into manageable sub-problems while maintaining solution
    continuity across block boundaries.

    The solver uses finite difference methods with the Laplace operator and implements
    various boundary conditions including fixed boundaries, zero gradients, and
    prescribed gradients.

    Attributes:
        block_size (tuple): Size of each block as (height, width)
        grad_vec (np.ndarray): Vector of prescribed gradients at block boundaries
        grid_step (float): Grid spacing for finite difference calculations
        grad_factor (float): Factor for scaling gradients in the middle block
        block_height (int): Height of each block (extracted from block_size)
        curr_state_* (np.ndarray): Current state arrays for each block
        next_state_* (np.ndarray): Next state arrays for each block (working arrays)
    """

    def __init__(
        self,
        block_size: tuple[int, int],
        grad_vec: np.ndarray,
        grid_step: float,
        grad_factor: float,
    ) -> None:
        """
        Initialize the Sandwich solver with the specified parameters.

        Args:
            block_size (tuple): Size of each block as (height, width)
            grad_vec (np.ndarray): Vector of prescribed gradients at block boundaries.
                                  Must have length 3*block_height.
            grid_step (float): Grid spacing for finite difference calculations
            grad_factor (float): Factor for scaling gradients in the middle block.
                               Values > 1 increase middle block gradients,
                               < 1 decrease them.

        Note:
            The middle block is created with 2 extra rows (top and bottom) to
            accommodate boundary conditions and data transfer requirements.
        """
        self.block_size = self._validate_block_size(block_size)
        self.block_height = self.block_size[0]
        self.grad_vec = self._validate_grad_vec(grad_vec, self.block_height)
        self.grid_step = self._validate_positive_number(grid_step, "grid_step")
        self.grad_factor = self._validate_positive_number(grad_factor, "grad_factor")

        # Current state blocks
        self.top = MeshBlock(self.block_size)
        self.bottom = MeshBlock(self.block_size)

        # we need one more row from each side because of boundary conditions
        mid_block_size = (self.block_size[0] + 2, self.block_size[1])
        self.mid = MeshBlock(mid_block_size)

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
    def _validate_grad_vec(grad_vec: np.ndarray, block_height: int) -> np.ndarray:
        grad_array = np.asarray(grad_vec, dtype=float)
        expected_length = 3 * block_height

        if grad_array.ndim != 1:
            raise ValueError("grad_vec must be a 1-dimensional array")

        if grad_array.shape != (expected_length,):
            raise ValueError(
                "grad_vec must have shape "
                f"({expected_length},), got {grad_array.shape}"
            )

        return grad_array

    @staticmethod
    def _validate_positive_number(value: float, name: str) -> float:
        numeric_value = float(value)

        if not np.isfinite(numeric_value):
            raise ValueError(f"{name} must be finite")

        if numeric_value <= 0:
            raise ValueError(f"{name} must be positive")

        return numeric_value

    def step(self) -> None:
        """
        Execute one complete iteration step of the Sandwich solver.

        This method performs the complete solution cycle:
        1. Set boundary conditions for all blocks
        2. Apply Laplace operator to outer blocks (bottom and top)
        3. Transfer data from outer blocks to middle block
        4. Apply Laplace operator to middle block
        5. Transfer data from middle block back to outer blocks
        6. Swap current and next state arrays

        Note:
            This method modifies the internal state arrays in-place.
            Call this method repeatedly in a loop to converge to the solution.
        """
        self._set_boundary_conditions()
        self._make_laplace_step_outer()
        self._transfer_values_inwards()
        self._make_laplace_step_inner()
        self._transfer_gradients_outwards()

    def plot(self, plot_abs: bool = False) -> go.Figure:
        """
        Generate a heatmap visualization of the current displacement field.

        Args:
            plot_abs (bool, optional): If True, plot absolute displacement values.
                If False (default), plot raw displacement values.

        Returns:
            plotly.graph_objs._figure.Figure: A Plotly heatmap figure showing the
            displacement field across all three blocks.

        Note:
            The displacement array is flipped vertically ([::-1]) to match conventional
            plotting conventions where the origin is at the bottom-left.
        """
        displacement = self.get_displacement_array()[::-1]

        if plot_abs:
            displacement = np.abs(displacement)

        return go.Figure(data=go.Heatmap(z=displacement))

    def get_displacement_array(self) -> np.ndarray:
        """
        Get the complete displacement field across all three blocks.

        Returns:
            np.ndarray: A concatenated array containing the displacement values
                       from all three blocks in the order: bottom, middle, top.
                       The middle block excludes the boundary padding rows.

        Note:
            The returned array has shape (3*block_height, block_width) and represents
            the current state of the entire computational domain.
        """
        return np.concatenate(
            (
                self.bottom._state,
                self.mid._state[1:-1],  # remove boundary conditions paddings
                self.top._state,
            )
        )

    def get_residual(self) -> float:
        """
        Calculate the residual (error) of the current solution.

        The residual measures how well the current state satisfies the Laplace equation.
        It is computed as the sum of absolute differences between the current state
        and what the Laplace operator would predict for that state.

        Returns:
            float: The total residual across all three blocks. Lower values indicate
                   better convergence to the solution.

        Note:
            The residual is computed only for interior points (excluding boundaries)
            and summed across all blocks. This value can be used to monitor convergence
            during iterative solution.
        """
        results: list[float] = []
        for block in [self.bottom._state, self.mid._state, self.top._state]:
            res_block = np.copy(block)
            set_laplace_update(res_block)

            results.append(
                float(np.sum(np.abs(block[1:-1, 1:-1] - res_block[1:-1, 1:-1])))
            )

        return sum(results)

    def _set_boundary_conditions(self) -> None:
        """
        Set boundary conditions for all three blocks.

        This private method applies the appropriate boundary conditions to each block:
        - Bottom block: Uses first block_height elements of grad_vec
        - Middle block: Uses elements from block_height-1 to 2*block_height+1
          of grad_vec
        - Top block: Uses last block_height elements of grad_vec

        Note:
            This method modifies the next_state arrays in-place by calling the
            appropriate boundary condition functions for each block.
        """
        set_boundary_conditions_bottom_block(
            self.bottom, self.grad_vec[: self.block_height], self.grid_step
        )
        set_boundary_conditions_middle_block(
            self.mid,
            self.grad_vec[self.block_height - 1 : 2 * self.block_height + 1],
            self.grid_step,
        )
        set_boundary_conditions_top_block(
            self.top,
            self.grad_vec[2 * self.block_height : 3 * self.block_height],
            self.grid_step,
        )

    def _make_laplace_step_outer(self) -> None:
        """
        Apply Laplace operator to the outer blocks (bottom and top).

        This private method updates the next_state arrays of the bottom and top blocks
        by applying the finite difference Laplace operator to their current states.
        """
        set_laplace_update(self.bottom._state)
        set_laplace_update(self.top._state)

    def _make_laplace_step_inner(self) -> None:
        """
        Apply Laplace operator to the middle block.

        This private method updates the next_state array of the middle block
        by applying the finite difference Laplace operator to its current state.
        """
        set_laplace_update(self.mid._state)

    def _transfer_values_inwards(self) -> None:
        # displacements are continuous
        copy_boundary_values(
            self.bottom, BoundaryType.TOP, self.mid, BoundaryType.BOTTOM
        )
        copy_boundary_values(self.top, BoundaryType.BOTTOM, self.mid, BoundaryType.TOP)

    def _transfer_gradients_outwards(self) -> None:
        # grads are proportional
        grad_scale = 1 / self.grad_factor

        copy_boundary_gradients(
            self.mid, BoundaryType.BOTTOM, self.bottom, BoundaryType.TOP, grad_scale
        )
        copy_boundary_gradients(
            self.mid, BoundaryType.TOP, self.top, BoundaryType.BOTTOM, grad_scale
        )
