"""
Tests for the MeshBlock class.
"""

import pytest
import numpy as np
from sandwich_numerical.solver.mesh_block import MeshBlock, BoundaryType


class TestMeshBlock:
    """Test cases for the MeshBlock class."""
    
    @pytest.fixture
    def sample_block(self):
        """Create a sample MeshBlock for testing."""
        return MeshBlock((3, 4), dtype=np.float64)
    
    @pytest.fixture
    def large_block(self):
        """Create a larger MeshBlock for testing."""
        return MeshBlock((5, 7), dtype=np.float32)
    
    def test_creation_with_valid_shapes(self):
        """Test that MeshBlock objects can be created with valid shapes."""
        # Test various valid shapes
        block1 = MeshBlock((2, 3))
        assert block1.shape == (2, 3)
        assert block1.dtype == np.float64
        
        block2 = MeshBlock((10, 5), dtype=np.float32)
        assert block2.shape == (10, 5)
        assert block2.dtype == np.float32
        
        block3 = MeshBlock((1, 100))
        assert block3.shape == (1, 100)
        
        block4 = MeshBlock((100, 1))
        assert block4.shape == (100, 1)
    
    def test_initial_state(self, sample_block):
        """Test that the initial state is properly initialized."""
        assert sample_block.state.shape == (3, 4)
        assert sample_block.state.dtype == np.float64
        assert np.all(sample_block.state == 0.0)
        assert sample_block.state is not sample_block._state  # Should be a view
    
    def test_properties(self, sample_block):
        """Test that all properties return correct values."""
        assert sample_block.shape == (3, 4)
        assert sample_block.dtype == np.float64
        assert sample_block.state.shape == (3, 4)
    
    def test_set_state_with_valid_array(self, sample_block):
        """Test setting state with a valid numpy array."""
        new_state = np.array([[1.0, 2.0, 3.0, 4.0],
                             [5.0, 6.0, 7.0, 8.0],
                             [9.0, 10.0, 11.0, 12.0]])
        
        sample_block.set_state(new_state)
        
        # Check that the state was updated
        assert np.array_equal(sample_block.state, new_state)
        # Check that it's the same array (in-place update)
        assert sample_block.state is not new_state
    
    def test_set_state_with_different_dtype(self, sample_block):
        """Test setting state with an array of different dtype."""
        new_state = np.array([[1, 2, 3, 4],
                             [5, 6, 7, 8],
                             [9, 10, 11, 12]], dtype=np.int32)
        
        sample_block.set_state(new_state)
        
        # Check that the state was updated and dtype was converted
        assert np.array_equal(sample_block.state, new_state)
        assert sample_block.state.dtype == np.float64  # Should be converted to block's dtype
    
    def test_set_boundary_values_with_single_values(self, sample_block):
        """Test setting boundary values with single numbers."""
        # Test setting all boundaries to the same value
        sample_block.set_boundary_values(BoundaryType.LEFT, 5.0)
        sample_block.set_boundary_values(BoundaryType.RIGHT, 10.0)
        sample_block.set_boundary_values(BoundaryType.TOP, 15.0)
        sample_block.set_boundary_values(BoundaryType.BOTTOM, 20.0)
        
        # Check that boundaries were set correctly
        # Note: TOP and BOTTOM overwrite the corners of LEFT and RIGHT
        assert sample_block.state[1, 0] == 5.0   # Left (middle row)
        assert sample_block.state[1, -1] == 10.0  # Right (middle row)
        assert np.all(sample_block.state[0, :] == 15.0)   # Top (overwrites left/right corners)
        assert np.all(sample_block.state[-1, :] == 20.0)  # Bottom (overwrites left/right corners)
    
    def test_set_boundary_values_with_arrays(self, sample_block):
        """Test setting boundary values with numpy arrays."""
        # Create arrays for each boundary
        left_values = np.array([1.0, 2.0, 3.0])
        right_values = np.array([4.0, 5.0, 6.0])
        top_values = np.array([7.0, 8.0, 9.0, 10.0])
        bottom_values = np.array([11.0, 12.0, 13.0, 14.0])
        
        # Set boundary values
        sample_block.set_boundary_values(BoundaryType.LEFT, left_values)
        sample_block.set_boundary_values(BoundaryType.RIGHT, right_values)
        sample_block.set_boundary_values(BoundaryType.TOP, top_values)
        sample_block.set_boundary_values(BoundaryType.BOTTOM, bottom_values)
        
        # Check that boundaries were set correctly
        # Note: TOP and BOTTOM overwrite the corners of LEFT and RIGHT
        # The final state should be:
        # Top row: [7, 8, 9, 10] (overwrites left/right corners)
        # Middle row: [1, 0, 0, 4] (left and right boundaries)
        # Bottom row: [11, 12, 13, 14] (overwrites left/right corners)
        
        # Check top row (completely overwritten by TOP)
        np.testing.assert_array_equal(sample_block.state[0, :], top_values)
        
        # Check middle row (LEFT and RIGHT boundaries, not overwritten by TOP/BOTTOM)
        assert sample_block.state[1, 0] == left_values[1]  # Left boundary
        assert sample_block.state[1, -1] == right_values[1]  # Right boundary
        
        # Check bottom row (completely overwritten by BOTTOM)
        np.testing.assert_array_equal(sample_block.state[-1, :], bottom_values)
    
    def test_boundary_values_with_different_dtypes(self, sample_block):
        """Test setting boundary values with different data types."""
        # Test with integer values
        sample_block.set_boundary_values(BoundaryType.LEFT, 5)
        assert sample_block.state[1, 0] == 5.0  # Should be converted to float (middle row)
        
        # Test with float values
        sample_block.set_boundary_values(BoundaryType.RIGHT, 10.5)
        assert sample_block.state[1, -1] == 10.5  # Middle row
        
        # Test with numpy arrays of different dtypes
        int_array = np.array([1, 2, 3, 4], dtype=np.int32)  # Must match width (4)
        sample_block.set_boundary_values(BoundaryType.TOP, int_array)
        assert np.all(sample_block.state[0, :] == [1.0, 2.0, 3.0, 4.0])  # Top row
    
    def test_get_boundary_values(self, sample_block):
        """Test getting boundary values."""
        # Set some boundary values first
        sample_block.set_boundary_values(BoundaryType.LEFT, 5.0)
        sample_block.set_boundary_values(BoundaryType.RIGHT, 10.0)
        sample_block.set_boundary_values(BoundaryType.TOP, 15.0)
        sample_block.set_boundary_values(BoundaryType.BOTTOM, 20.0)
        
        # Get boundary values
        left_values = sample_block.get_boundary_values(BoundaryType.LEFT)
        right_values = sample_block.get_boundary_values(BoundaryType.RIGHT)
        top_values = sample_block.get_boundary_values(BoundaryType.TOP)
        bottom_values = sample_block.get_boundary_values(BoundaryType.BOTTOM)
        
        # Check that returned values are correct
        # Note: TOP and BOTTOM overwrite the corners of LEFT and RIGHT
        assert left_values[1] == 5.0  # Middle row (not overwritten by TOP/BOTTOM)
        assert right_values[1] == 10.0  # Middle row (not overwritten by TOP/BOTTOM)
        assert np.all(top_values == 15.0)  # Top row
        assert np.all(bottom_values == 20.0)  # Bottom row
        
        # Check that returned arrays are copies, not views
        assert left_values is not sample_block.state[:, 0]
        assert right_values is not sample_block.state[:, -1]
        assert top_values is not sample_block.state[0, :]
        assert bottom_values is not sample_block.state[-1, :]
    
    def test_get_boundary_values_with_enum_values(self, sample_block):
        """Test getting boundary values using enum values."""
        # Set boundary values
        sample_block.set_boundary_values(BoundaryType.LEFT, 5.0)
        
        # Get using enum value
        left_values = sample_block.get_boundary_values(BoundaryType.LEFT)
        assert np.all(left_values == 5.0)
    
    def test_get_boundary_gradients(self):
        """Test getting boundary gradients."""
        # Create a mesh block with known values
        mesh = MeshBlock((3, 3))
        
        # Set up a simple state with known gradients
        state = np.array([[1.0, 2.0, 3.0],
                         [4.0, 5.0, 6.0],
                         [7.0, 8.0, 9.0]])
        mesh.set_state(state)
        
        # Test gradients
        top_grad = mesh.get_boundary_gradients(BoundaryType.TOP)
        np.testing.assert_array_equal(top_grad, np.array([-3.0, -3.0, -3.0]))
        
        bottom_grad = mesh.get_boundary_gradients(BoundaryType.BOTTOM)
        np.testing.assert_array_equal(bottom_grad, np.array([-3.0, -3.0, -3.0]))
        
        left_grad = mesh.get_boundary_gradients(BoundaryType.LEFT)
        np.testing.assert_array_equal(left_grad, np.array([1.0, 1.0, 1.0]))
        
        right_grad = mesh.get_boundary_gradients(BoundaryType.RIGHT)
        np.testing.assert_array_equal(right_grad, np.array([1.0, 1.0, 1.0]))
    
    def test_set_boundary_gradients(self):
        """Test setting boundary gradients."""
        # Create a mesh block
        mesh = MeshBlock((3, 3))
        
        # Set up initial state
        state = np.array([[1.0, 2.0, 3.0],
                         [4.0, 5.0, 6.0],
                         [7.0, 8.0, 9.0]])
        mesh.set_state(state)
        
        # Test setting top gradient
        mesh.set_boundary_gradients(BoundaryType.TOP, np.array([0.5, 1.0, 1.5]))
        top_grad = mesh.get_boundary_gradients(BoundaryType.TOP)
        np.testing.assert_array_equal(top_grad, np.array([0.5, 1.0, 1.5]))
        
        # Test setting right gradient
        mesh.set_boundary_gradients(BoundaryType.RIGHT, np.array([-1.0, -1.0, -1.0]))
        right_grad = mesh.get_boundary_gradients(BoundaryType.RIGHT)
        np.testing.assert_array_equal(right_grad, np.array([-1.0, -1.0, -1.0]))
        
        # Test setting bottom gradient
        mesh.set_boundary_gradients(BoundaryType.BOTTOM, np.array([0.5, 1.0, 1.5]))
        bottom_grad = mesh.get_boundary_gradients(BoundaryType.BOTTOM)
        np.testing.assert_array_equal(bottom_grad, np.array([0.5, 1.0, 1.5]))
    
    def test_repr_and_str(self, sample_block):
        """Test string representations."""
        # Test __repr__
        repr_str = repr(sample_block)
        assert "MeshBlock" in repr_str
        assert "shape=(3, 4)" in repr_str
        assert "dtype=" in repr_str
        
        # Test __str__
        str_str = str(sample_block)
        assert "MeshBlock" in str_str
        assert "State:" in str_str
        assert "shape=(3, 4)" in str_str

if __name__ == "__main__":
    pytest.main([__file__]) 
