import numpy as np
from typing import Union


class MeshBlock:
    """
    A class that stores a numpy array and provides methods for state management.
    
    This class encapsulates a 2D numpy array representing a mesh block and provides
    a safe interface for updating the state in-place.
    """
    
    def __init__(self, shape: tuple, dtype: np.dtype = np.float64):
        """
        Initialize a MeshBlock with a numpy array of the specified shape.
        
        Args:
            shape: Shape of the array as a tuple (height, width)
            dtype: Data type for the array (default: np.float64)
        """
        if not isinstance(shape, tuple):
            raise TypeError("Shape must be a tuple (height, width)")
        
        if len(shape) != 2:
            raise ValueError("Shape must be 2-dimensional (height, width)")
        
        self._state = np.zeros(shape, dtype=dtype)
        self._shape = shape
        self._dtype = dtype
    
    @property
    def state(self) -> np.ndarray:
        """
        Get the current state array.
        
        Returns:
            The current state array (read-only view)
        """
        return self._state.view()
    
    @property
    def shape(self) -> tuple:
        """
        Get the shape of the mesh block.
        
        Returns:
            Tuple of (height, width)
        """
        return self._shape
    
    @property
    def dtype(self) -> np.dtype:
        """
        Get the data type of the mesh block.
        
        Returns:
            The numpy data type
        """
        return self._dtype
    
    def set_state(self, new_state: np.ndarray) -> None:
        """
        Update the state array in-place with new values.
        
        Args:
            new_state: New state array to copy into the current state
            
        Raises:
            ValueError: If the new state has incompatible shape
        """
        if not isinstance(new_state, np.ndarray):
            raise TypeError("new_state must be a numpy array")
        
        if new_state.shape != self._shape:
            raise ValueError(f"Shape mismatch: expected {self._shape}, got {new_state.shape}")
        
        # Copy the new state into the existing array (in-place update)
        # This will automatically handle dtype conversion if needed
        np.copyto(self._state, new_state)
    
    def set_boundary_values(self, boundary: str, values: Union[np.ndarray, float, int]) -> None:
        """
        Update boundary values of the mesh block.
        
        Args:
            boundary: Boundary to update ('left', 'right', 'top', 'bottom')
            values: Values to set. Can be:
                - Single number (float/int) to set all boundary points to the same value
                - 1D numpy array with length matching the boundary dimension
                - 2D numpy array with shape matching the boundary
        
        Raises:
            ValueError: If boundary name is invalid or values have incompatible shape
        """
        boundary = boundary.lower()
        
        if boundary not in ['left', 'right', 'top', 'bottom']:
            raise ValueError("Boundary must be one of: 'left', 'right', 'top', 'bottom'")
        
        if isinstance(values, (int, float)):
            # Convert single value to appropriate array
            if boundary in ['left', 'right']:
                values = np.full(self._shape[0], values, dtype=self._dtype)
            else:  # top, bottom
                values = np.full(self._shape[1], values, dtype=self._dtype)
        
        if not isinstance(values, np.ndarray):
            raise TypeError("Values must be a number or numpy array")
        
        # Validate and set boundary values
        if boundary == 'left':
            if values.shape != (self._shape[0],):
                raise ValueError(f"Left boundary values must have shape ({self._shape[0]},), got {values.shape}")
            self._state[:, 0] = values
        elif boundary == 'right':
            if values.shape != (self._shape[0],):
                raise ValueError(f"Right boundary values must have shape ({self._shape[0]},), got {values.shape}")
            self._state[:, -1] = values
        elif boundary == 'top':
            if values.shape != (self._shape[1],):
                raise ValueError(f"Top boundary values must have shape ({self._shape[1]},), got {values.shape}")
            self._state[0, :] = values
        elif boundary == 'bottom':
            if values.shape != (self._shape[1],):
                raise ValueError(f"Bottom boundary values must have shape ({self._shape[1]},), got {values.shape}")
            self._state[-1, :] = values
    
    def __repr__(self) -> str:
        """String representation of the MeshBlock."""
        return f"MeshBlock(shape={self._shape}, dtype={self._dtype})"
    
    def __str__(self) -> str:
        """String representation showing the current state."""
        return f"MeshBlock(shape={self._shape})\nState:\n{self._state}" 
