import numpy as np
from typing import Union


class MeshBlock:
    """
    A class that stores a numpy array and provides methods for state management.
    
    This class encapsulates a 2D numpy array representing a mesh block and provides
    a safe interface for updating the state in-place.
    """
    
    def __init__(self, shape: Union[tuple, int], dtype: np.dtype = np.float64):
        """
        Initialize a MeshBlock with a numpy array of the specified shape.
        
        Args:
            shape: Shape of the array (height, width) or single dimension for square array
            dtype: Data type for the array (default: np.float64)
        """
        if isinstance(shape, int):
            shape = (shape, shape)
        
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
    
    def __repr__(self) -> str:
        """String representation of the MeshBlock."""
        return f"MeshBlock(shape={self._shape}, dtype={self._dtype})"
    
    def __str__(self) -> str:
        """String representation showing the current state."""
        return f"MeshBlock(shape={self._shape})\nState:\n{self._state}" 
