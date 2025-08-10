import numpy as np
import plotly.express as px


def set_laplace_update(curr_state: np.ndarray, next_state: np.ndarray):
    """
    Apply the Laplace operator to update the next state from the current state.
    
    This function implements a finite difference approximation of the Laplace operator
    using a 5-point stencil. It computes the average of the four neighboring values
    (left, right, bottom, top) for each interior point.
    
    Args:
        curr_state (np.ndarray): Current state array of shape (n, m)
        next_state (np.ndarray): Next state array of shape (n, m) to be updated
        
    Note:
        The function updates only the interior points (1:-1, 1:-1) of next_state.
        Boundary points are assumed to be handled separately by boundary condition functions.
    """
    
    assert len(curr_state.shape) == 2
    assert len(next_state.shape) == 2
    
    values_left = curr_state[1:-1, :-2]
    values_right = curr_state[1:-1, 2:]
    
    values_bottom = curr_state[:-2, 1:-1]
    values_top = curr_state[2:, 1:-1]
    
    next_state[1:-1, 1:-1] = (values_left + values_right + values_bottom + values_top) / 4


def set_boundary_conditions_bottom_block(curr_state: np.ndarray, next_state: np.ndarray, 
                                       grad_vec: np.ndarray, grid_step: float):
    """
    Set boundary conditions for the bottom block of the sandwich structure.
    
    This function enforces three types of boundary conditions:
    1. Fixed boundary: The far end (right side) is clamped to zero
    2. Zero gradient: The bottom side has zero normal derivative (df/dx2 = 0)
    3. Prescribed gradient: The near end (left side) has a known gradient from grad_vec
    
    Args:
        curr_state (np.ndarray): Current state array of shape (n, m)
        next_state (np.ndarray): Next state array of shape (n, m) to be updated
        grad_vec (np.ndarray): Gradient vector specifying the prescribed gradient at the near end
        grid_step (float): Grid spacing for finite difference calculations
        
    Note:
        The function modifies next_state in-place by setting boundary values.
    """
    
    next_state[:, -1] = 0         # the block is fixed on the far end
    next_state[0] = curr_state[1] # df/dx2 = 0 on the bottom side
    
    # gradients are known on the near end
    next_state[:, 0] = curr_state[:, 1] - grad_vec * grid_step


def set_boundary_conditions_top_block(curr_state: np.ndarray, next_state: np.ndarray, 
                                    grad_vec: np.ndarray, grid_step: float):
    """
    Set boundary conditions for the top block of the sandwich structure.
    
    This function enforces three types of boundary conditions:
    1. Fixed boundary: The far end (right side) is clamped to zero
    2. Zero gradient: The top side has zero normal derivative (df/dx2 = 0)
    3. Prescribed gradient: The near end (left side) has a known gradient from grad_vec
    
    Args:
        curr_state (np.ndarray): Current state array of shape (n, m)
        next_state (np.ndarray): Next state array of shape (n, m) to be updated
        grad_vec (np.ndarray): Gradient vector specifying the prescribed gradient at the near end
        grid_step (float): Grid spacing for finite difference calculations
        
    Note:
        The function modifies next_state in-place by setting boundary values.
        Assumes both arrays have the same 2D shape.
    """

    assert len(curr_state.shape) == 2
    assert len(next_state.shape) == 2
    
    next_state[:, -1] = 0            # the block is fixed on the far end
    next_state[-1] = curr_state[-2]  # df/dx2 = 0 on the top side
    
    # gradients are known on the near end
    next_state[:, 0] = curr_state[:, 1] - grad_vec * grid_step


def set_boundary_conditions_middle_block(curr_state: np.ndarray, next_state: np.ndarray,
                                       grad_vec: np.ndarray, grid_step: float):
    """
    Set boundary conditions for the middle block of the sandwich structure.
    
    This function enforces two types of boundary conditions:
    1. Fixed boundary: The far end (right side) is clamped to zero
    2. Prescribed gradient: The near end (left side) has a known gradient from grad_vec
    
    Args:
        curr_state (np.ndarray): Current state array of shape (n, m)
        next_state (np.ndarray): Next state array of shape (n, m) to be updated
        grad_vec (np.ndarray): Gradient vector specifying the prescribed gradient at the near end
        grid_step (float): Grid spacing for finite difference calculations
        
    Note:
        The function modifies next_state in-place by setting boundary values.
        The middle block has no zero-gradient conditions on top/bottom sides.
    """
    
    next_state[:, -1] = 0         # the block is fixed on the far end
    
    # gradients are known on the near end
    next_state[:, 0] = curr_state[:, 1] - grad_vec * grid_step


def transfer_data_inwards(curr_state_bottom: np.ndarray, curr_state_top: np.ndarray, curr_state_mid: np.ndarray,
                              mid_grad_factor: float):
    """
    Transfer displacement and gradient data from outer blocks to the middle block.
    
    This function implements the data transfer mechanism that ensures continuity
    between the three blocks of the sandwich structure. It enforces:
    1. Displacement continuity: Values at block interfaces match exactly
    2. Gradient proportionality: Gradients are scaled by mid_grad_factor for the middle block
    
    Args:
        curr_state_bottom (np.ndarray): Current state of the bottom block
        curr_state_top (np.ndarray): Current state of the top block  
        curr_state_mid (np.ndarray): Current state of the middle block to be updated
        mid_grad_factor (float): Factor to scale gradients in the middle block
        
    Note:
        The function modifies curr_state_mid in-place to ensure data consistency
        between the three blocks. This is a key step in the multi-block solver.
    """
    
    # displacements are continuous
    curr_state_mid[0] = curr_state_bottom[-1]
    curr_state_mid[-1] = curr_state_top[0]
    
    # grads are proportional
    grad_bottom = curr_state_bottom[-1] - curr_state_bottom[-2]
    grad_top = curr_state_top[1] - curr_state_top[0]
    
    curr_state_mid[1] = curr_state_mid[0] + mid_grad_factor * grad_bottom
    curr_state_mid[-2] = curr_state_mid[-1] - mid_grad_factor * grad_top

def transfer_data_outwards(curr_state_bottom: np.ndarray, curr_state_top: np.ndarray, curr_state_mid: np.ndarray,
                              mid_grad_factor: float):
    """
    Transfer displacement and gradient data from the middle block to outer blocks.
    
    This function implements the reverse data transfer mechanism that updates
    the outer blocks based on the middle block's solution. It ensures:
    1. Displacement continuity: Values at block interfaces match exactly
    2. Gradient consistency: Gradients are properly scaled back to outer blocks
    
    Args:
        curr_state_bottom (np.ndarray): Current state of the bottom block to be updated
        curr_state_top (np.ndarray): Current state of the top block to be updated
        curr_state_mid (np.ndarray): Current state of the middle block (source)
        mid_grad_factor (float): Factor used to scale gradients back to outer blocks
        
    Note:
        The function modifies curr_state_bottom and curr_state_top in-place.
        This completes the data exchange cycle in the multi-block solver.
    """
    
    # displacements are continuous
    curr_state_bottom[-1] = curr_state_mid[0]
    curr_state_top[0] = curr_state_mid[-1]
    
    # grads are proportional
    grad_bottom = curr_state_mid[1] - curr_state_mid[0]
    grad_top = curr_state_mid[-1] - curr_state_mid[-2]
    
    curr_state_bottom[-1] = curr_state_bottom[-2] + grad_bottom / mid_grad_factor
    curr_state_top[0] = curr_state_top[1] - grad_top / mid_grad_factor


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
        
        self.curr_state_top = np.zeros(block_size)
        self.next_state_top = np.zeros(block_size)
        
        self.curr_state_bottom = np.zeros(block_size)
        self.next_state_bottom = np.zeros(block_size)
        
        # we need one more row from each side because of boundary conditions
        mid_block_size = (block_size[0] + 2, block_size[1])
        self.curr_state_mid = np.zeros(mid_block_size)
        self.next_state_mid = np.zeros(mid_block_size)
        
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
        self._transfer_info_inwards()
        self._make_laplace_step_inner()
        self._transfer_info_outwards()
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
        
        return px.imshow(displacement)
    
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
                self.curr_state_bottom, 
                self.curr_state_mid[1:-1], # remove boundary conditions paddings 
                self.curr_state_top
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
        for block in [self.curr_state_bottom, self.curr_state_mid, self.curr_state_top]:
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
            self.curr_state_bottom, 
            self.next_state_bottom, 
            self.grad_vec[:self.block_height], 
            self.grid_step
        )
        set_boundary_conditions_middle_block(
            self.curr_state_mid, 
            self.next_state_mid, 
            self.grad_vec[self.block_height-1:2*self.block_height+1], 
            self.grid_step
        )
        set_boundary_conditions_top_block(
            self.curr_state_top, 
            self.next_state_top, 
            self.grad_vec[2*self.block_height:3*self.block_height], 
            self.grid_step
        )
        
    def _make_laplace_step_outer(self):
        """
        Apply Laplace operator to the outer blocks (bottom and top).
        
        This private method updates the next_state arrays of the bottom and top blocks
        by applying the finite difference Laplace operator to their current states.
        """
        set_laplace_update(self.curr_state_bottom, self.next_state_bottom)
        set_laplace_update(self.curr_state_top, self.next_state_top)
        
    def _make_laplace_step_inner(self):
        """
        Apply Laplace operator to the middle block.
        
        This private method updates the next_state array of the middle block
        by applying the finite difference Laplace operator to its current state.
        """
        set_laplace_update(self.curr_state_mid, self.next_state_mid)
        
    def _transfer_info_inwards(self):
        """
        Transfer data from outer blocks to the middle block.
        
        This private method ensures continuity between blocks by transferring
        displacement and gradient information from the bottom and top blocks
        to the middle block using the prescribed grad_factor.
        """
        transfer_data_inwards(
            self.curr_state_bottom,
            self.curr_state_top,
            self.curr_state_mid,
            self.grad_factor
        )
        
    def _transfer_info_outwards(self):
        """
        Transfer data from the middle block to outer blocks.
        
        This private method updates the outer blocks based on the middle block's
        solution, ensuring consistency across all block boundaries.
        """
        transfer_data_outwards(
            self.next_state_bottom,
            self.next_state_top,
            self.next_state_mid,
            self.grad_factor
        )
        
    def _swap(self):
        """
        Swap current and next state arrays for all blocks.
        
        This private method prepares for the next iteration by making the
        computed next_state arrays the new current_state arrays.
        """
        self.curr_state_bottom, self.next_state_bottom = self.next_state_bottom, self.curr_state_bottom
        self.curr_state_mid, self.next_state_mid = self.next_state_mid, self.curr_state_mid
        self.curr_state_top, self.next_state_top = self.next_state_top, self.curr_state_top






