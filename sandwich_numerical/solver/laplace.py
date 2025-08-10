import numpy as np


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
