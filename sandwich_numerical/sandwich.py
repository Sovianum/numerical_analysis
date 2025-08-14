import numpy as np
import plotly.graph_objects as go
from sandwich_numerical.solver.mesh_block import BoundaryType, MeshBlock
from sandwich_numerical.solver.mesh_utils import copy_boundary_gradients, copy_boundary_values

from .solver.laplace import set_laplace_update


class Sandwich:
    """
    A multi-block numerical solver implementing the generalized Sandwich method for differential equations.
    
    The generalized Sandwich method divides the computational domain into an arbitrary odd number of blocks:
    - 1 bottom block
    - N mid blocks (where N is odd)
    - 1 top block
    
    Each mid block has its own gradient factor, allowing for fine-tuned control over the solution.
    The solver uses finite difference methods with the Laplace operator and implements
    various boundary conditions including fixed boundaries, zero gradients, and prescribed gradients.
    
    Attributes:
        block_size (tuple): Size of each block as (height, width)
        grad_vec (np.ndarray): Vector of prescribed gradients at block boundaries
        grid_step (float): Grid spacing for finite difference calculations
        grad_factors (list): List of gradient factors for each block
        num_mid_blocks (int): Number of middle blocks (must be odd)
        block_height (int): Height of each block (extracted from block_size)
        blocks (list): List of all mesh blocks in order: [bottom, mid1, mid2, ..., midN, top]
    """
    
    def __init__(self, num_mid_blocks, block_size, grad_vec, grid_step, grad_factors, learning_rate=1.0):
        """
        Initialize the generalized Sandwich solver with the specified parameters.
        
        Args:
            block_size (tuple): Size of each block as (height, width)
            grad_vec (np.ndarray): Vector of prescribed gradients at block boundaries.
                                  Must have length (2 + num_mid_blocks) * block_height + 1
            grid_step (float): Grid spacing for finite difference calculations
            grad_factors (list): List of gradient factors for each block.
                               Length must be 2 + num_mid_blocks (bottom + mid blocks + top).
                               Values > 1 increase block gradients, < 1 decrease them.
            num_mid_blocks (int): Number of middle blocks (must be odd)
        
        Note:
            Each mid block is created with 2 extra rows (top and bottom) to accommodate
            boundary conditions and data transfer requirements.
        """
        self.block_size = block_size
        self.grad_vec = grad_vec
        self.grid_step = grid_step
        self.grad_factors = grad_factors
        self.num_mid_blocks = num_mid_blocks
        self.learning_rate = learning_rate
        # Validate that we have an odd number of mid blocks
        if self.num_mid_blocks % 2 == 0:
            raise ValueError("Number of mid blocks must be odd")
        
        self.block_height = block_size[0]
        
        # Calculate total blocks needed
        total_blocks = 2 + self.num_mid_blocks  # bottom + mid blocks + top
        
        # Validate grad_vec length
        expected_length = total_blocks * self.block_height
        if len(grad_vec) != expected_length:
            raise ValueError(f"grad_vec must have length {expected_length}, got {len(grad_vec)}")
        
        # Validate grad_factors length
        if len(grad_factors) != total_blocks:
            raise ValueError(f"grad_factors must have length {total_blocks}, got {len(grad_factors)}")

        # Create all blocks
        self.blocks = []
        
        # Bottom block
        self.blocks.append(MeshBlock(block_size))
        
        # Mid blocks (with padding)
        for i in range(self.num_mid_blocks):
            self.blocks.append(MeshBlock(self._pad_block_size(block_size, padding=1)))
        
        # Top block
        self.blocks.append(MeshBlock(block_size))
        
        # Store references for convenience
        self.bottom = self.blocks[0]
        self.top = self.blocks[-1]
        self.mid_blocks = self.blocks[1:-1]

    def _pad_block_size(self, block_size: tuple, padding: int) -> tuple:
        return (block_size[0] + 2*padding, block_size[1])
        
    def step(self):
        """
        Execute one complete iteration step of the generalized Sandwich solver.
        
        This method performs the complete solution cycle:
        1. Set boundary conditions for all blocks
        2. Run Laplace step on all blocks
        3. Get gradients of all blocks
        4. Transfer saved gradients to adjacent blocks
        5. Run Laplace step again on all blocks
        6. Get all boundary values
        7. Transfer boundary values to adjacent blocks
        
        Note:
            This method modifies the internal state arrays in-place.
            Call this method repeatedly in a loop to converge to the solution.
        """
        self._set_boundary_conditions()

        self._run_laplace_on_all_blocks()

        self._transfer_saved_gradients_to_adjacent_blocks(
            self._get_gradients_of_all_blocks()
        )

        self._run_laplace_on_all_blocks()

        self._transfer_boundary_values_to_adjacent_blocks(
            self._get_all_boundary_values()
        )

    def _run_laplace_on_all_blocks(self):
        """
        Phase 1: Run Laplace step on all blocks.
        
        This method applies the Laplace operator to all blocks in the system.
        """
        for block in self.blocks:
            set_laplace_update(block._state, self.learning_rate)

    def _get_gradients_of_all_blocks(self):
        """
        Phase 2: Get gradients of all blocks.
        
        This method extracts and stores gradients from all blocks for later transfer.
        It computes gradients at TOP and BOTTOM boundaries of each block and returns them
        for use in the gradient transfer phase.
        
        Returns:
            dict: Nested dictionary containing gradients for each block at TOP and BOTTOM boundaries
        """
        # Store gradients for each block at TOP and BOTTOM boundaries only
        stored_gradients = {}
        
        for i, block in enumerate(self.blocks):
            stored_gradients[i] = {}
            
            # Extract gradients only at TOP and BOTTOM boundaries for this block
            for boundary in [BoundaryType.TOP, BoundaryType.BOTTOM]:
                try:
                    gradients = block.get_boundary_gradients(boundary)
                    stored_gradients[i][boundary] = gradients.copy()
                except ValueError:
                    # Block might be too small for gradients, store None
                    stored_gradients[i][boundary] = None
        
        return stored_gradients

    def _get_all_boundary_values(self):
        """
        Get boundary values of all blocks.
        
        This method extracts boundary values from all blocks at TOP and BOTTOM boundaries
        for later transfer to adjacent blocks.
        
        Returns:
            dict: Nested dictionary containing boundary values for each block at TOP and BOTTOM boundaries
        """
        # Store boundary values for each block at TOP and BOTTOM boundaries only
        stored_boundary_values = {}
        
        for i, block in enumerate(self.blocks):
            stored_boundary_values[i] = {}
            
            # Extract boundary values only at TOP and BOTTOM boundaries for this block
            for boundary in [BoundaryType.TOP, BoundaryType.BOTTOM]:
                try:
                    boundary_values = block.get_boundary_values(boundary)
                    stored_boundary_values[i][boundary] = boundary_values.copy()
                except ValueError:
                    # Block might be too small, store None
                    stored_boundary_values[i][boundary] = None
        
        return stored_boundary_values

    def _transfer_boundary_values_to_adjacent_blocks(self, stored_boundary_values):
        """
        Transfer boundary values to adjacent blocks.
        
        This method transfers boundary values from each block to its adjacent blocks
        at the TOP and BOTTOM boundaries.
        
        Args:
            stored_boundary_values (dict): Nested dictionary containing boundary values for each block and boundary
        """
        # Transfer boundary values between adjacent blocks
        for i in range(len(self.blocks) - 1):
            # Transfer from current block to next block (bottom to top direction)
            self.blocks[i + 1].set_boundary_values(
                BoundaryType.BOTTOM, 
                stored_boundary_values[i][BoundaryType.TOP]
            )
            
            # Transfer from next block to current block (top to bottom direction)
            self.blocks[i].set_boundary_values(
                BoundaryType.TOP, 
                stored_boundary_values[i + 1][BoundaryType.BOTTOM]
            )

    def _transfer_saved_gradients_to_adjacent_blocks(self, stored_gradients):
        """
        Phase 3: Transfer saved gradients to adjacent blocks.
        
        This method transfers boundary gradients from each block to its adjacent blocks,
        using the appropriate gradient scaling factors and the pre-computed gradients.
        
        Args:
            stored_gradients (dict): Nested dictionary containing gradients for each block and boundary
        """
        # Transfer gradients between adjacent blocks
        for i in range(len(self.blocks) - 1):
            # Transfer from current block to next block (bottom to top direction)
            self.blocks[i + 1].set_boundary_gradients(
                BoundaryType.BOTTOM, 
                stored_gradients[i][BoundaryType.TOP] * self._get_grad_scale(i, i + 1)
            )
            
            # Transfer from next block to current block (top to bottom direction)
            self.blocks[i].set_boundary_gradients(
                BoundaryType.TOP, 
                stored_gradients[i + 1][BoundaryType.BOTTOM] * self._get_grad_scale(i + 1, i)
            )
        
    def plot(self, plot_abs=False):
        """
        Generate a heatmap visualization of the current displacement field.
        
        Args:
            plot_abs (bool, optional): If True, plot the absolute values of displacement.
                                     If False (default), plot the raw displacement values.
        
        Returns:
            plotly.graph_objs._figure.Figure: A Plotly heatmap figure showing the
            displacement field across all blocks.
        
        Note:
            The displacement array is flipped vertically ([::-1]) to match conventional
            plotting conventions where the origin is at the bottom-left.
        """
        displacement = self.get_displacement_array()[::-1]
        
        if plot_abs:
            displacement = np.abs(displacement)
        
        return go.Figure(data=go.Heatmap(z=displacement))
    
    def get_displacement_array(self):
        """
        Get the complete displacement field across all blocks.
        
        Returns:
            np.ndarray: A concatenated array containing the displacement values
                       from all blocks in order: bottom, mid1, mid2, ..., midN, top.
                       Mid blocks exclude the boundary padding rows.
        
        Note:
            The returned array has shape ((2 + num_mid_blocks) * block_height, block_width)
            and represents the current state of the entire computational domain.
        """
        displacement_parts = []
        
        # Add bottom block
        displacement_parts.append(self.bottom._state)
        
        # Add mid blocks (removing boundary padding)
        for mid_block in self.mid_blocks:
            displacement_parts.append(mid_block._state[1:-1])
        
        # Add top block
        displacement_parts.append(self.top._state)
        
        return np.concatenate(displacement_parts)
    
    def get_residual(self):
        """
        Calculate the residual (error) of the current solution.
        
        The residual measures how well the current state satisfies the Laplace equation.
        It is computed as the sum of absolute differences between the current state
        and what the Laplace operator would predict for that state.
        
        Returns:
            float: The total residual across all blocks. Lower values indicate
                   better convergence to the solution.
        
        Note:
            The residual is computed only for interior points (excluding boundaries)
            and summed across all blocks. This value can be used to monitor convergence
            during iterative solution.
        """
        results = []
        for block in self.blocks:
            res_block = np.copy(block._state)
            set_laplace_update(res_block)
            
            # For mid blocks, exclude padding rows
            if block in self.mid_blocks:
                results.append(
                    np.sum(
                        np.abs(
                            block._state[1:-1, 1:-1] - res_block[1:-1, 1:-1]
                        )
                    )
                )
            else:
                results.append(
                    np.sum(
                        np.abs(
                            block._state[1:-1, 1:-1] - res_block[1:-1, 1:-1]
                        )
                    )
                )
            
        return sum(results)
        
    def _set_boundary_conditions(self):
        """
        Set boundary conditions for all blocks.
        
        This private method applies the appropriate boundary conditions to each block:
        - Bottom block: Uses first block_height elements of grad_vec
        - Mid blocks: Use elements from their respective ranges of grad_vec
        - Top block: Uses remaining elements of grad_vec
        
        Note:
            This method modifies the next_state arrays in-place by calling the
            appropriate boundary condition functions for each block.
        """
        # Set boundary conditions for bottom block
        self._set_boundary_conditions_bottom_block(
            self.bottom, 
            self._get_block_grad_vec(block_id=0, padding=0), 
            self.grid_step
        )
        
        # Set boundary conditions for mid blocks
        for i, mid_block in enumerate(self.mid_blocks):
            self._set_boundary_conditions_middle_block(
                mid_block, 
                self._get_block_grad_vec(block_id=i+1, padding=1), 
                self.grid_step
            )
        
        # Set boundary conditions for top block
        self._set_boundary_conditions_top_block(
            self.top, 
            self._get_block_grad_vec(block_id=len(self.blocks)-1, padding=0), 
            self.grid_step
        )

    def _get_block_grad_vec(self, block_id: int, padding: int) -> np.ndarray:
        left_idx = block_id*self.block_height - padding
        right_idx = (block_id + 1)*self.block_height + padding

        return self.grad_vec[left_idx:right_idx]

    def _set_boundary_conditions_bottom_block(self, state: MeshBlock, 
                                        grad_vec: np.ndarray, grid_step: float):
        state.set_boundary_values(BoundaryType.RIGHT, 0) # the block is fixed on the far end
        state.set_boundary_gradients(BoundaryType.BOTTOM, 0) # df/dx2 = 0 on the bottom side
        
        # gradients are known on the near end
        state.set_boundary_gradients(BoundaryType.LEFT, grad_vec * grid_step)

    def _set_boundary_conditions_middle_block(self, state: MeshBlock,
                                        grad_vec: np.ndarray, grid_step: float):

        state.set_boundary_values(BoundaryType.RIGHT, 0) # the block is fixed on the far end
        
        # gradients are known on the near end
        state.set_boundary_gradients(BoundaryType.LEFT, grad_vec * grid_step)

    def _set_boundary_conditions_top_block(self, state: MeshBlock, 
                                        grad_vec: np.ndarray, grid_step: float):
        state.set_boundary_values(BoundaryType.RIGHT, 0) # the block is fixed on the far end
        state.set_boundary_gradients(BoundaryType.TOP, 0) # df/dx2 = 0 on the top side
        
        # gradients are known on the near end
        state.set_boundary_gradients(BoundaryType.LEFT, grad_vec * grid_step)
        
    def _get_grad_scale(self, block_id_curr: int, block_id_next: int) -> float:
        return self.grad_factors[block_id_curr] / self.grad_factors[block_id_next]
