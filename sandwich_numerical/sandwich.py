import numpy as np
import plotly.graph_objects as go
from sandwich_numerical.solver.mesh_block import BoundaryType, MeshBlock
from sandwich_numerical.solver.mesh_utils import copy_boundary_gradients, copy_boundary_values

from .solver.laplace import set_laplace_update


def set_boundary_conditions_bottom_block(curr_state: MeshBlock, next_state: MeshBlock, 
                                       grad_vec: np.ndarray, grid_step: float):
    """
    Set boundary conditions for the bottom block of the sandwich structure.
    
    This function enforces three types of boundary conditions:
    1. Fixed boundary: The far end (right side) is clamped to zero
    2. Zero gradient: The bottom side has zero normal derivative (df/dx2 = 0)
    3. Prescribed gradient: The near end (left side) has a known gradient from grad_vec
    
    Args:
        curr_state (MeshBlock): Current state mesh block
        next_state (MeshBlock): Next state mesh block to be updated
        grad_vec (np.ndarray): Gradient vector specifying the prescribed gradient at the near end
        grid_step (float): Grid spacing for finite difference calculations
        
    Note:
        The function modifies next_state in-place by setting boundary values.
    """

    next_state.set_boundary_values(BoundaryType.RIGHT, 0) # the block is fixed on the far end
    next_state.set_boundary_gradients(BoundaryType.BOTTOM, 0) # df/dx2 = 0 on the bottom side
    
    # gradients are known on the near end
    next_state.set_boundary_values(BoundaryType.LEFT, curr_state._state[:, 1] - grad_vec * grid_step)


def set_boundary_conditions_top_block(curr_state: MeshBlock, next_state: MeshBlock, 
                                    grad_vec: np.ndarray, grid_step: float):
    """
    Set boundary conditions for the top block of the sandwich structure.
    
    This function enforces three types of boundary conditions:
    1. Fixed boundary: The far end (right side) is clamped to zero
    2. Zero gradient: The top side has zero normal derivative (df/dx2 = 0)
    3. Prescribed gradient: The near end (left side) has a known gradient from grad_vec
    
    Args:
        curr_state (MeshBlock): Current state mesh block
        next_state (MeshBlock): Next state mesh block to be updated
        grad_vec (np.ndarray): Gradient vector specifying the prescribed gradient at the near end
        grid_step (float): Grid spacing for finite difference calculations
        
    Note:
        The function modifies next_state in-place by setting boundary values.
        Assumes both mesh blocks have the same 2D shape.
    """
    
    next_state.set_boundary_values(BoundaryType.RIGHT, 0) # the block is fixed on the far end
    next_state.set_boundary_gradients(BoundaryType.TOP, 0) # df/dx2 = 0 on the top side
    
    # gradients are known on the near end
    next_state.set_boundary_values(BoundaryType.LEFT, curr_state._state[:, 1] - grad_vec * grid_step)


def set_boundary_conditions_middle_block(curr_state: MeshBlock, next_state: MeshBlock,
                                       grad_vec: np.ndarray, grid_step: float):
    """
    Set boundary conditions for the middle block of the sandwich structure.
    
    This function enforces two types of boundary conditions:
    1. Fixed boundary: The far end (right side) is clamped to zero
    2. Prescribed gradient: The near end (left side) has a known gradient from grad_vec
    
    Args:
        curr_state (MeshBlock): Current state mesh block
        next_state (MeshBlock): Next state mesh block to be updated
        grad_vec (np.ndarray): Gradient vector specifying the prescribed gradient at the near end
        grid_step (float): Grid spacing for finite difference calculations
        
    Note:
        The function modifies next_state in-place by setting boundary values.
        The middle block has no zero-gradient conditions on top/bottom sides.
    """

    next_state.set_boundary_values(BoundaryType.RIGHT, 0) # the block is fixed on the far end
    
    # gradients are known on the near end
    next_state.set_boundary_values(BoundaryType.LEFT, curr_state._state[:, 1] - grad_vec * grid_step)


class Sandwich:
    """
    A multi-block numerical solver implementing the Sandwich method for differential equations.
    
    The Sandwich method divides the computational domain into three blocks (bottom, middle, top)
    and solves them iteratively with data transfer between blocks. This approach allows for
    efficient solution of large-scale problems by breaking them into manageable sub-problems
    while maintaining solution continuity across block boundaries.
    
    The solver uses finite difference methods with the Laplace operator and implements
    various boundary conditions including fixed boundaries, zero gradients, and prescribed gradients.
    
    Attributes:
        block_size (tuple): Size of each block as (height, width)
        grad_vec (np.ndarray): Vector of prescribed gradients at block boundaries
        grid_step (float): Grid spacing for finite difference calculations
        grad_factor (float): Factor for scaling gradients in the middle block
        block_height (int): Height of each block (extracted from block_size)
        curr_state_* (np.ndarray): Current state arrays for each block
        next_state_* (np.ndarray): Next state arrays for each block (working arrays)
    """
    
    def __init__(self, block_size, grad_vec, grid_step, grad_factor):
        """
        Initialize the Sandwich solver with the specified parameters.
        
        Args:
            block_size (tuple): Size of each block as (height, width)
            grad_vec (np.ndarray): Vector of prescribed gradients at block boundaries.
                                  Must have length 3*block_height + 1 for proper indexing.
            grid_step (float): Grid spacing for finite difference calculations
            grad_factor (float): Factor for scaling gradients in the middle block.
                               Values > 1 increase middle block gradients, < 1 decrease them.
        
        Note:
            The middle block is created with 2 extra rows (top and bottom) to accommodate
            boundary conditions and data transfer requirements.
        """
        self.block_size = block_size
        self.grad_vec = grad_vec
        self.grid_step = grid_step
        self.grad_factor = grad_factor
        
        self.block_height = block_size[0]

        # Current state blocks
        self.top_curr = MeshBlock(block_size)
        self.bottom_curr = MeshBlock(block_size)
        
        # we need one more row from each side because of boundary conditions
        mid_block_size = (block_size[0] + 2, block_size[1])
        self.mid_curr = MeshBlock(mid_block_size)
        
        # Next state blocks
        self.top_next = MeshBlock(block_size)
        self.bottom_next = MeshBlock(block_size)
        self.mid_next = MeshBlock(mid_block_size)
        
    def step(self):
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
        self._swap()
        
    def plot(self, plot_abs=False):
        """
        Generate a heatmap visualization of the current displacement field.
        
        Args:
            plot_abs (bool, optional): If True, plot the absolute values of displacement.
                                     If False (default), plot the raw displacement values.
        
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
    
    def get_displacement_array(self):
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
                self.bottom_curr._state, 
                self.mid_curr._state[1:-1], # remove boundary conditions paddings 
                self.top_curr._state
            )
        )
    
    def get_residual(self):
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
        results = []
        for block in [self.bottom_curr._state, self.mid_curr._state, self.top_curr._state]:
            res_block = np.zeros_like(block)
            set_laplace_update(block, res_block)
            
            results.append(
                np.sum(
                    np.abs(
                        block[1:-1, 1:-1] - res_block[1:-1, 1:-1]
                    )
                )
            )
            
        return sum(results)
        
    def _set_boundary_conditions(self):
        """
        Set boundary conditions for all three blocks.
        
        This private method applies the appropriate boundary conditions to each block:
        - Bottom block: Uses first block_height elements of grad_vec
        - Middle block: Uses elements from block_height-1 to 2*block_height+1 of grad_vec
        - Top block: Uses remaining elements of grad_vec
        
        Note:
            This method modifies the next_state arrays in-place by calling the
            appropriate boundary condition functions for each block.
        """
        set_boundary_conditions_bottom_block(
            self.bottom_curr, 
            self.bottom_next, 
            self.grad_vec[:self.block_height], 
            self.grid_step
        )
        set_boundary_conditions_middle_block(
            self.mid_curr, 
            self.mid_next, 
            self.grad_vec[self.block_height-1:2*self.block_height+1], 
            self.grid_step
        )
        set_boundary_conditions_top_block(
            self.top_curr, 
            self.top_next, 
            self.grad_vec[2*self.block_height:3*self.block_height], 
            self.grid_step
        )
        
    def _make_laplace_step_outer(self):
        """
        Apply Laplace operator to the outer blocks (bottom and top).
        
        This private method updates the next_state arrays of the bottom and top blocks
        by applying the finite difference Laplace operator to their current states.
        """
        set_laplace_update(self.bottom_curr._state, self.bottom_next._state)
        set_laplace_update(self.top_curr._state, self.top_next._state)
        
    def _make_laplace_step_inner(self):
        """
        Apply Laplace operator to the middle block.
        
        This private method updates the next_state array of the middle block
        by applying the finite difference Laplace operator to its current state.
        """
        set_laplace_update(self.mid_curr._state, self.mid_next._state)

    def _transfer_values_inwards(self):
        # displacements are continuous
        copy_boundary_values(self.bottom_next, BoundaryType.TOP, self.mid_next, BoundaryType.BOTTOM)
        copy_boundary_values(self.top_next, BoundaryType.BOTTOM, self.mid_next, BoundaryType.TOP)

    def _transfer_gradients_outwards(self):
        # grads are proportional
        grad_scale = 1 / self.grad_factor

        copy_boundary_gradients(self.mid_next, BoundaryType.BOTTOM, self.bottom_next, BoundaryType.TOP, grad_scale)
        copy_boundary_gradients(self.mid_next, BoundaryType.TOP, self.top_next, BoundaryType.BOTTOM, grad_scale)
        
    def _swap(self):
        """
        Swap current and next state arrays for all blocks.
        
        This private method prepares for the next iteration by making the
        computed next_state arrays the new current_state arrays.
        """
        self.bottom_curr._state, self.bottom_next._state = self.bottom_next._state, self.bottom_curr._state
        self.mid_curr._state, self.mid_next._state = self.mid_next._state, self.mid_curr._state
        self.top_curr._state, self.top_next._state = self.top_next._state, self.top_curr._state
