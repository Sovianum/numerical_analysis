"""
Tests for the MeshBlock class.
"""

import pytest
import numpy as np
from sandwich_numerical.solver.mesh_block import MeshBlock


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
        """Test setting state with array of different dtype."""
        new_state = np.array([[1, 2, 3, 4],
                              [5, 6, 7, 8],
                              [9, 10, 11, 12]], dtype=np.int32)
        
        sample_block.set_state(new_state)
        
        # Check that the state was updated and dtype was converted
        assert np.array_equal(sample_block.state, new_state.astype(np.float64))
    
    def test_set_boundary_values_with_single_values(self, sample_block):
        """Test setting boundary values with single numbers."""
        # Set all boundaries to different values
        sample_block.set_boundary_values('left', 1.0)
        sample_block.set_boundary_values('right', 2.0)
        sample_block.set_boundary_values('top', 3.0)
        sample_block.set_boundary_values('bottom', 4.0)
        
        expected = np.array([[3.0, 3.0, 3.0, 3.0],
                            [1.0, 0.0, 0.0, 2.0],
                            [4.0, 4.0, 4.0, 4.0]])
        
        assert np.array_equal(sample_block.state, expected)
    
    def test_set_boundary_values_with_arrays(self, sample_block):
        """Test setting boundary values with arrays."""
        # Set left boundary with array
        left_values = np.array([10.0, 20.0, 30.0])
        sample_block.set_boundary_values('left', left_values)
        
        # Set top boundary with array
        top_values = np.array([100.0, 200.0, 300.0, 400.0])
        sample_block.set_boundary_values('top', top_values)
        
        expected = np.array([[100.0, 200.0, 300.0, 400.0],
                            [20.0, 0.0, 0.0, 0.0],
                            [30.0, 0.0, 0.0, 0.0]])
        
        assert np.array_equal(sample_block.state, expected)
    
    def test_set_boundary_values_with_invalid_boundary(self, sample_block):
        """Test that set_boundary_values fails with invalid boundary names."""
        with pytest.raises(ValueError, match="Boundary must be one of: 'left', 'right', 'top', 'bottom'"):
            sample_block.set_boundary_values('invalid', 1.0)
        
        with pytest.raises(ValueError, match="Boundary must be one of: 'left', 'right', 'top', 'bottom'"):
            sample_block.set_boundary_values('center', 1.0)
        
        with pytest.raises(ValueError, match="Boundary must be one of: 'left', 'right', 'top', 'bottom'"):
            sample_block.set_boundary_values('', 1.0)
    
    def test_boundary_values_with_different_dtypes(self, sample_block):
        """Test setting boundary values with different dtypes."""
        # Test with integer values
        sample_block.set_boundary_values('left', 5)
        sample_block.set_boundary_values('top', 10)
        
        # Test with float32 array
        right_values = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        sample_block.set_boundary_values('right', right_values)
        
        # Test with int32 array
        bottom_values = np.array([100, 200, 300, 400], dtype=np.int32)
        sample_block.set_boundary_values('bottom', bottom_values)
        
        expected = np.array([[10.0, 10.0, 10.0, 1.0],
                            [5.0, 0.0, 0.0, 2.0],
                            [100.0, 200.0, 300.0, 400.0]])
        
        assert np.array_equal(sample_block.state, expected)
    
    def test_multiple_boundary_updates(self, sample_block):
        """Test multiple updates to the same boundary."""
        # Update left boundary multiple times
        sample_block.set_boundary_values('left', 1.0)
        sample_block.set_boundary_values('left', 5.0)
        sample_block.set_boundary_values('left', np.array([10.0, 20.0, 30.0]))
        
        # Update top boundary multiple times
        sample_block.set_boundary_values('top', 100.0)
        sample_block.set_boundary_values('top', np.array([1.0, 2.0, 3.0, 4.0]))
        
        expected = np.array([[1.0, 2.0, 3.0, 4.0],
                            [20.0, 0.0, 0.0, 0.0],
                            [30.0, 0.0, 0.0, 0.0]])
        
        assert np.array_equal(sample_block.state, expected)


if __name__ == "__main__":
    pytest.main([__file__]) 
