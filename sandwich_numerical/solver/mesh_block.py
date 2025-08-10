import numpy as np
from typing import Union, Callable
from enum import Enum


class BoundaryType(Enum):
    """Enumeration of valid boundary types for the mesh block."""
    LEFT = "left"
    RIGHT = "right"
    TOP = "top"
    BOTTOM = "bottom"


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
        self._shadow_state = np.zeros(shape, dtype=dtype)
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
    def shadow_state(self) -> np.ndarray:
        """
        Get the current shadow state array.
        
        Returns:
            The current shadow state array (read-only view)
        """
        return self._shadow_state.view()
    
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
    
    def set_shadow_state(self, new_state: np.ndarray) -> None:
        """
        Update the shadow state array in-place with new values.
        
        Args:
            new_state: New state array to copy into the shadow state
            
        Raises:
            ValueError: If the new state has incompatible shape
        """
        if not isinstance(new_state, np.ndarray):
            raise TypeError("new_state must be a numpy array")
        
        if new_state.shape != self._shape:
            raise ValueError(f"Shape mismatch: expected {self._shape}, got {new_state.shape}")
        
        # Copy the new state into the existing shadow array (in-place update)
        # This will automatically handle dtype conversion if needed
        np.copyto(self._shadow_state, new_state)
    
    def _preprocess_boundary_values(self, boundary: BoundaryType, values: Union[np.ndarray, float, int]) -> np.ndarray:
        """
        Preprocess boundary values to ensure they are in the correct format.
        
        Args:
            boundary: Boundary to update (must be BoundaryType enum)
            values: Values to set. Can be:
                - Single number (float/int) to set all boundary points to the same value
                - 1D numpy array with length matching the boundary dimension
                - 2D numpy array with shape matching the boundary
        
        Returns:
            Preprocessed numpy array with correct shape and dtype
            
        Raises:
            ValueError: If values have incompatible shape
            TypeError: If values are not a number or numpy array
        """
        if isinstance(values, (int, float)):
            # Convert single value to appropriate array
            if boundary in [BoundaryType.LEFT, BoundaryType.RIGHT]:
                values = np.full(self._shape[0], values, dtype=self._dtype)
            else:  # TOP, BOTTOM
                values = np.full(self._shape[1], values, dtype=self._dtype)
        
        if not isinstance(values, np.ndarray):
            raise TypeError("Values must be a number or numpy array")
        
        return values

    def set_boundary_values(self, boundary: BoundaryType, values: Union[np.ndarray, float, int]) -> None:
        """
        Update boundary values of the mesh block.
        
        Args:
            boundary: Boundary to update (must be BoundaryType enum)
            values: Values to set. Can be:
                - Single number (float/int) to set all boundary points to the same value
                - 1D numpy array with length matching the boundary dimension
                - 2D numpy array with shape matching the boundary
        
        Raises:
            ValueError: If values have incompatible shape
            TypeError: If boundary is not a BoundaryType enum
        """
        # Validate boundary type
        if not isinstance(boundary, BoundaryType):
            raise TypeError(f"boundary must be a BoundaryType enum, got {type(boundary).__name__}")
        
        # Preprocess the values to ensure correct format
        preprocessed_values = self._preprocess_boundary_values(boundary, values)
        
        # Validate and set boundary values
        if boundary == BoundaryType.LEFT:
            if preprocessed_values.shape != (self._shape[0],):
                raise ValueError(f"Left boundary values must have shape ({self._shape[0]},), got {preprocessed_values.shape}")
            self._state[:, 0] = preprocessed_values
        elif boundary == BoundaryType.RIGHT:
            if preprocessed_values.shape != (self._shape[0],):
                raise ValueError(f"Right boundary values must have shape ({self._shape[0]},), got {preprocessed_values.shape}")
            self._state[:, -1] = preprocessed_values
        elif boundary == BoundaryType.TOP:
            if preprocessed_values.shape != (self._shape[1],):
                raise ValueError(f"Top boundary values must have shape ({self._shape[1]},), got {preprocessed_values.shape}")
            self._state[0, :] = preprocessed_values
        elif boundary == BoundaryType.BOTTOM:
            if preprocessed_values.shape != (self._shape[1],):
                raise ValueError(f"Bottom boundary values must have shape ({self._shape[1]},), got {preprocessed_values.shape}")
            self._state[-1, :] = preprocessed_values
    
    def get_boundary_values(self, boundary: BoundaryType) -> np.ndarray:
        """
        Get boundary values from the mesh block.
        
        Args:
            boundary: Boundary to retrieve (must be BoundaryType enum)
            
        Returns:
            Numpy array containing the boundary values
            
        Raises:
            TypeError: If boundary is not a BoundaryType enum
        """
        # Validate boundary type
        if not isinstance(boundary, BoundaryType):
            raise TypeError(f"boundary must be a BoundaryType enum, got {type(boundary).__name__}")
        
        # Return the appropriate boundary values
        if boundary == BoundaryType.LEFT:
            return self._state[:, 0].copy()
        elif boundary == BoundaryType.RIGHT:
            return self._state[:, -1].copy()
        elif boundary == BoundaryType.TOP:
            return self._state[0, :].copy()
        elif boundary == BoundaryType.BOTTOM:
            return self._state[-1, :].copy()
        else:
            # This should never happen due to the enum validation above
            valid_boundaries = [b.value for b in BoundaryType]
            raise ValueError(f"Boundary must be one of: {valid_boundaries}")
    
    def get_boundary_gradients(self, boundary: BoundaryType) -> np.ndarray:
        """
        Get boundary gradients from the mesh block.
        
        The gradients are computed as:
        - Top gradient: "top line" - "adjacent line below"
        - Bottom gradient: "line adjacent to the bottom one" - "bottom line"
        - Left gradient: "line adjacent to the left line" - "left line"
        - Right gradient: "right line" - "line adjacent to the right line"
        
        Args:
            boundary: Boundary to retrieve gradients for (must be BoundaryType enum)
            
        Returns:
            Numpy array containing the boundary gradients
            
        Raises:
            TypeError: If boundary is not a BoundaryType enum
            ValueError: If the mesh block is too small to compute gradients
        """
        # Validate boundary type
        if not isinstance(boundary, BoundaryType):
            raise TypeError(f"boundary must be a BoundaryType enum, got {type(boundary).__name__}")
        
        # Check if the mesh block is large enough to compute gradients
        if self._shape[0] < 2 or self._shape[1] < 2:
            raise ValueError("Mesh block must be at least 2x2 to compute gradients")
        
        # Compute gradients based on boundary type
        if boundary == BoundaryType.TOP:
            # Top gradient: top line - adjacent line below
            return self._state[0, :] - self._state[1, :]
        elif boundary == BoundaryType.BOTTOM:
            # Bottom gradient: line adjacent to bottom - bottom line
            return self._state[-2, :] - self._state[-1, :]
        elif boundary == BoundaryType.LEFT:
            # Left gradient: line adjacent to left - left line
            return self._state[:, 1] - self._state[:, 0]
        elif boundary == BoundaryType.RIGHT:
            # Right gradient: right line - line adjacent to right
            return self._state[:, -1] - self._state[:, -2]
        else:
            # This should never happen due to the enum validation above
            valid_boundaries = [b.value for b in BoundaryType]
            raise ValueError(f"Boundary must be one of: {valid_boundaries}")
    
    def _preprocess_gradients(self, boundary: BoundaryType, gradients: Union[np.ndarray, float, int]) -> np.ndarray:
        """
        Preprocess gradients to ensure correct format and validate shape.
        
        Args:
            boundary: Boundary type to determine expected shape
            gradients: Input gradients (can be number or numpy array)
            
        Returns:
            Preprocessed numpy array with correct shape and dtype
            
        Raises:
            TypeError: If gradients is not a number or numpy array
            ValueError: If gradients have incompatible shape
        """
        gradients_array = gradients
        if isinstance(gradients, (int, float)):
            if boundary in [BoundaryType.LEFT, BoundaryType.RIGHT]:
                gradients_array = np.full(self._shape[0], gradients, dtype=self._dtype)
            else:  # TOP, BOTTOM
                gradients_array = np.full(self._shape[1], gradients, dtype=self._dtype)
        elif isinstance(gradients, np.ndarray):
            gradients_array = gradients
        else:
            raise TypeError("gradients must be a number or numpy array")
        
        # Validate gradient shape
        if boundary in [BoundaryType.LEFT, BoundaryType.RIGHT]:
            if gradients_array.shape != (self._shape[0],):
                raise ValueError(f"Left/Right boundary gradients must have shape ({self._shape[0]},), got {gradients_array.shape}")
        else:  # TOP, BOTTOM
            if gradients_array.shape != (self._shape[1],):
                raise ValueError(f"Top/Bottom boundary gradients must have shape ({self._shape[1]},), got {gradients_array.shape}")
        
        return gradients_array

    def set_boundary_gradients(self, boundary: BoundaryType, gradients: Union[np.ndarray, float, int]) -> None:
        """
        Set boundary gradients by modifying only the boundary values.
        
        This method works backwards from the desired gradient to determine what boundary
        values need to be set. It preserves the adjacent interior values and adjusts
        only the boundary values to achieve the specified gradients.
        
        The gradients are set as:
        - Top gradient: "top line" - "adjacent line below" → adjusts top line
        - Bottom gradient: "line adjacent to bottom" - "bottom line" → adjusts bottom line
        - Left gradient: "line adjacent to left" - "left line" → adjusts left line
        - Right gradient: "right line" - "line adjacent to right" → adjusts right line
        
        Args:
            boundary: Boundary to update (must be BoundaryType enum)
            gradients: Desired gradients. Can be:
                - Single number (float/int) to set all boundary gradients to the same value
                - 1D numpy array with length matching the boundary dimension
        
        Raises:
            TypeError: If boundary is not a BoundaryType enum
            ValueError: If the mesh block is too small or gradients have incompatible shape
        """
        # Validate boundary type
        if not isinstance(boundary, BoundaryType):
            raise TypeError(f"boundary must be a BoundaryType enum, got {type(boundary).__name__}")
        
        # Check if the mesh block is large enough to compute gradients
        if self._shape[0] < 2 or self._shape[1] < 2:
            raise ValueError("Mesh block must be at least 2x2 to set gradients")
        
        # Preprocess the gradients to ensure correct format
        gradients_array = self._preprocess_gradients(boundary, gradients)
        
        # Set boundary values to achieve the desired gradients
        if boundary == BoundaryType.TOP:
            # Top gradient: top line - adjacent line below
            # So: top line = adjacent line below + gradient
            self._state[0, :] = self._state[1, :] + gradients_array
        elif boundary == BoundaryType.BOTTOM:
            # Bottom gradient: line adjacent to bottom - bottom line
            # So: bottom line = line adjacent to bottom - gradient
            self._state[-1, :] = self._state[-2, :] - gradients_array
        elif boundary == BoundaryType.LEFT:
            # Left gradient: line adjacent to left - left line
            # So: left line = line adjacent to left - gradient
            self._state[:, 0] = self._state[:, 1] - gradients_array
        elif boundary == BoundaryType.RIGHT:
            # Right gradient: right line - line adjacent to right
            # So: right line = line adjacent to right + gradient
            self._state[:, -1] = self._state[:, -2] + gradients_array
        else:
            # This should never happen due to the enum validation above
            valid_boundaries = [b.value for b in BoundaryType]
            raise ValueError(f"Boundary must be one of: {valid_boundaries}")
    
    def apply(self, func: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> None:
        """
        Apply a callable function that uses both the main state and shadow state.
        
        Args:
            func: A callable that takes (state, shadow_state) as arguments and returns the new state
        """
        new_state = func(self._state, self._shadow_state)
        self._state = new_state
    
    def swap(self) -> None:
        """
        Swap the main state with the shadow state.
        """
        self._state, self._shadow_state = self._shadow_state, self._state
    
    def __repr__(self) -> str:
        """String representation of the MeshBlock."""
        return f"MeshBlock(shape={self._shape}, dtype={self._dtype})"
    
    def __str__(self) -> str:
        """String representation showing the current state and shadow state."""
        return f"MeshBlock(shape={self._shape})\nState:\n{self._state}\nShadow State:\n{self._shadow_state}" 
