import numpy as np


def set_laplace_update(state: np.ndarray, learning_rate: float = 1.0):
    """
    Apply the Laplace operator to update the next state from the current state.
    
    This function implements a finite difference approximation of the Laplace operator
    using a 5-point stencil. It computes the average of the four neighboring values
    (left, right, bottom, top) for each interior point.
    
    Args:
        state (np.ndarray): Current state array of shape (n, m)
        
    Note:
        The function updates only the interior points (1:-1, 1:-1) of state.
        Boundary points are assumed to be handled separately by boundary condition functions.
    """
    
    assert len(state.shape) == 2
    
    values_left = state[1:-1, :-2]
    values_right = state[1:-1, 2:]
    
    values_bottom = state[:-2, 1:-1]
    values_top = state[2:, 1:-1]

    old_mid_state = state[1:-1, 1:-1]
    new_mid_state = (values_left + values_right + values_bottom + values_top) / 4 

    state[1:-1, 1:-1] = old_mid_state + learning_rate * (new_mid_state - old_mid_state)
