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
    
    def __init__(self, num_mid_blocks, block_size, grad_vec, grid_step, grad_factors):
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
        2. Go inwards transferring values from outer blocks to inner blocks
        3. Go outwards transferring gradients from inner blocks to outer blocks
        4. Each layer has its own grad factor for gradient scaling
        
        Note:
            This method modifies the internal state arrays in-place.
            Call this method repeatedly in a loop to converge to the solution.
        """
        self._set_boundary_conditions()
        self._transfer_values_inwards()
        self._transfer_gradients_outwards()
        
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
        
    def _transfer_values_inwards(self):
        """
        Transfer values inwards from outer blocks to inner blocks using a pairwise approach.
        
        This method implements the first phase of the step method:
        1. Apply Laplace operator to outer blocks (top and bottom)
        2. Transfer boundary values from outer blocks to adjacent middle blocks
        3. Repeat pairwise until reaching the middle mid block
        4. Apply Laplace operator to the middle mid block
        """
        # Calculate the middle index of blocks
        mid_mid_index = len(self.blocks) // 2
        
        # Continue pairwise until reaching the middle mid block
        left_index = 0
        right_index = len(self.blocks) - 1
        
        while left_index < mid_mid_index and right_index > mid_mid_index:
            # Apply Laplace to current pair of middle blocks
            set_laplace_update(self.blocks[left_index]._state)
            set_laplace_update(self.blocks[right_index]._state)
            
            # Transfer values to adjacent inner blocks
            if left_index + 1 < len(self.blocks):
                copy_boundary_values(self.blocks[left_index], BoundaryType.TOP, 
                                   self.blocks[left_index + 1], BoundaryType.BOTTOM)
            
            if right_index - 1 >= 0:
                copy_boundary_values(self.blocks[right_index], BoundaryType.BOTTOM, 
                                   self.blocks[right_index - 1], BoundaryType.TOP)
            
            left_index += 1
            right_index -= 1
        
        # Apply Laplace to the middle mid block
        set_laplace_update(self.blocks[mid_mid_index]._state)

    def _transfer_gradients_outwards(self):
        """
        Transfer gradients outwards from inner blocks to outer blocks using a pairwise approach.
        
        This method implements the second phase of the step method:
        1. Start from the middle mid block
        2. Transfer gradients to adjacent middle blocks (left and right)
        3. Continue pairwise outward until reaching the outer blocks
        4. Transfer gradients from middle blocks to outer blocks (bottom and top)
        with each layer having its own grad factor for scaling.
        """
        # Calculate the middle index of mid blocks
        mid_mid_index = len(self.blocks) // 2
        
        # Start from the middle mid block and work outward
        left_index = mid_mid_index
        right_index = mid_mid_index
        
        # Continue pairwise outward until reaching the outer blocks
        while left_index > 0 and right_index < len(self.blocks) - 1:
            # Transfer gradients from current middle blocks to adjacent outer blocks
            if left_index > 0:
                grad_scale = self._get_grad_scale(left_index, left_index - 1)
                copy_boundary_gradients(self.blocks[left_index], BoundaryType.BOTTOM, 
                                       self.blocks[left_index - 1], BoundaryType.TOP, grad_scale)
            
            if right_index < len(self.blocks) - 1:
                grad_scale = self._get_grad_scale(right_index, right_index + 1)
                copy_boundary_gradients(self.blocks[right_index], BoundaryType.TOP, 
                                       self.blocks[right_index + 1], BoundaryType.BOTTOM, grad_scale)
            
            left_index -= 1
            right_index += 1

    def _get_grad_scale(self, block_id_curr: int, block_id_next: int) -> float:
        return self.grad_factors[block_id_curr] / self.grad_factors[block_id_next]
