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
        sample_block.set_boundary_values(BoundaryType.LEFT, 1.0)
        sample_block.set_boundary_values(BoundaryType.RIGHT, 2.0)
        sample_block.set_boundary_values(BoundaryType.TOP, 3.0)
        sample_block.set_boundary_values(BoundaryType.BOTTOM, 4.0)
        
        expected = np.array([[3.0, 3.0, 3.0, 3.0],
                            [1.0, 0.0, 0.0, 2.0],
                            [4.0, 4.0, 4.0, 4.0]])
        
        assert np.array_equal(sample_block.state, expected)
    
    def test_set_boundary_values_with_arrays(self, sample_block):
        """Test setting boundary values with arrays."""
        # Set left boundary with array
        left_values = np.array([10.0, 20.0, 30.0])
        sample_block.set_boundary_values(BoundaryType.LEFT, left_values)
        
        # Set top boundary with array
        top_values = np.array([100.0, 200.0, 300.0, 400.0])
        sample_block.set_boundary_values(BoundaryType.TOP, top_values)
        
        expected = np.array([[100.0, 200.0, 300.0, 400.0],
                            [20.0, 0.0, 0.0, 0.0],
                            [30.0, 0.0, 0.0, 0.0]])
        
        assert np.array_equal(sample_block.state, expected)
    
    def test_set_boundary_values_with_invalid_boundary(self, sample_block):
        """Test that set_boundary_values fails with invalid boundary names."""
        # Test with invalid enum values (this should raise TypeError)
        with pytest.raises(TypeError):
            sample_block.set_boundary_values('invalid', 1.0)
        
        with pytest.raises(TypeError):
            sample_block.set_boundary_values('center', 1.0)
        
        with pytest.raises(TypeError):
            sample_block.set_boundary_values('', 1.0)
    
    def test_boundary_values_with_different_dtypes(self, sample_block):
        """Test setting boundary values with different dtypes."""
        # Test with integer values
        sample_block.set_boundary_values(BoundaryType.LEFT, 5)
        sample_block.set_boundary_values(BoundaryType.TOP, 10)
        
        # Test with float32 array
        right_values = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        sample_block.set_boundary_values(BoundaryType.RIGHT, right_values)
        
        # Test with int32 array
        bottom_values = np.array([100, 200, 300, 400], dtype=np.int32)
        sample_block.set_boundary_values(BoundaryType.BOTTOM, bottom_values)
        
        expected = np.array([[10.0, 10.0, 10.0, 1.0],
                            [5.0, 0.0, 0.0, 2.0],
                            [100.0, 200.0, 300.0, 400.0]])
        
        assert np.array_equal(sample_block.state, expected)
    
    def test_multiple_boundary_updates(self, sample_block):
        """Test multiple updates to the same boundary."""
        # Update left boundary multiple times
        sample_block.set_boundary_values(BoundaryType.LEFT, 1.0)
        sample_block.set_boundary_values(BoundaryType.LEFT, 5.0)
        sample_block.set_boundary_values(BoundaryType.LEFT, np.array([10.0, 20.0, 30.0]))
        
        # Update top boundary multiple times
        sample_block.set_boundary_values(BoundaryType.TOP, 100.0)
        sample_block.set_boundary_values(BoundaryType.TOP, np.array([1.0, 2.0, 3.0, 4.0]))
        
        expected = np.array([[1.0, 2.0, 3.0, 4.0],
                            [20.0, 0.0, 0.0, 0.0],
                            [30.0, 0.0, 0.0, 0.0]])
        
        assert np.array_equal(sample_block.state, expected)
    
    def test_get_boundary_values(self, sample_block):
        """Test getting boundary values from the mesh block."""
        # Set some boundary values first
        sample_block.set_boundary_values(BoundaryType.LEFT, 1.0)
        sample_block.set_boundary_values(BoundaryType.RIGHT, 2.0)
        sample_block.set_boundary_values(BoundaryType.TOP, 3.0)
        sample_block.set_boundary_values(BoundaryType.BOTTOM, 4.0)
        
        # Test getting left boundary values (note: TOP overwrites first row)
        left_values = sample_block.get_boundary_values(BoundaryType.LEFT)
        assert left_values.shape == (3,)
        assert np.array_equal(left_values, np.array([3.0, 1.0, 4.0]))  # TOP, LEFT, BOTTOM
        
        # Test getting right boundary values (note: TOP and BOTTOM overwrite first/last rows)
        right_values = sample_block.get_boundary_values(BoundaryType.RIGHT)
        assert right_values.shape == (3,)
        assert np.array_equal(right_values, np.array([3.0, 2.0, 4.0]))  # TOP, RIGHT, BOTTOM
        
        # Test getting top boundary values
        top_values = sample_block.get_boundary_values(BoundaryType.TOP)
        assert top_values.shape == (4,)
        assert np.array_equal(top_values, np.array([3.0, 3.0, 3.0, 3.0]))
        
        # Test getting bottom boundary values
        bottom_values = sample_block.get_boundary_values(BoundaryType.BOTTOM)
        assert bottom_values.shape == (4,)
        assert np.array_equal(bottom_values, np.array([4.0, 4.0, 4.0, 4.0]))
        
        # Test that returned arrays are copies, not views
        left_values[0] = 999.0
        assert sample_block.get_boundary_values(BoundaryType.LEFT)[0] == 3.0  # Original unchanged
    
    def test_get_boundary_values_with_enum_values(self, sample_block):
        """Test getting boundary values using enum values."""
        # Set boundary values
        sample_block.set_boundary_values(BoundaryType.LEFT, 5.0)
        sample_block.set_boundary_values(BoundaryType.TOP, 10.0)
        
        # Test getting with enum values (note: TOP overwrites first row of LEFT)
        left_values = sample_block.get_boundary_values(BoundaryType.LEFT)
        assert np.array_equal(left_values, np.array([10.0, 5.0, 5.0]))  # TOP, LEFT, LEFT
        
        top_values = sample_block.get_boundary_values(BoundaryType.TOP)
        assert np.array_equal(top_values, np.array([10.0, 10.0, 10.0, 10.0]))
    
    def test_get_boundary_values_with_invalid_boundary(self, sample_block):
        """Test that get_boundary_values fails with invalid boundary names."""
        # Test with invalid enum values (this should raise TypeError)
        with pytest.raises(TypeError):
            sample_block.get_boundary_values('invalid')
        
        with pytest.raises(TypeError):
            sample_block.get_boundary_values('center')
    
    def test_get_boundary_gradients(self):
        """Test getting boundary gradients for all boundary types."""
        # Create a 3x3 mesh block with known values
        mesh = MeshBlock((3, 3))
        # Set up a simple test pattern
        test_state = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ])
        mesh.set_state(test_state)
        
        # Test top gradient: top line - adjacent line below
        top_grad = mesh.get_boundary_gradients(BoundaryType.TOP)
        expected_top = np.array([1.0, 2.0, 3.0]) - np.array([4.0, 5.0, 6.0])
        np.testing.assert_array_equal(top_grad, expected_top)
        
        # Test bottom gradient: line adjacent to bottom - bottom line
        bottom_grad = mesh.get_boundary_gradients(BoundaryType.BOTTOM)
        expected_bottom = np.array([4.0, 5.0, 6.0]) - np.array([7.0, 8.0, 9.0])
        np.testing.assert_array_equal(bottom_grad, expected_bottom)
        
        # Test left gradient: line adjacent to left - left line
        left_grad = mesh.get_boundary_gradients(BoundaryType.LEFT)
        expected_left = np.array([2.0, 5.0, 8.0]) - np.array([1.0, 4.0, 7.0])
        np.testing.assert_array_equal(left_grad, expected_left)
        
        # Test right gradient: right line - line adjacent to right
        right_grad = mesh.get_boundary_gradients(BoundaryType.RIGHT)
        expected_right = np.array([3.0, 6.0, 9.0]) - np.array([2.0, 5.0, 8.0])
        np.testing.assert_array_equal(right_grad, expected_right)

    def test_set_boundary_gradients(self):
        """Test setting boundary gradients by modifying boundary values."""
        # Create a 3x3 mesh block with known values
        mesh = MeshBlock((3, 3))
        test_state = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ])
        mesh.set_state(test_state)
        
        # Test setting top gradient
        mesh.set_boundary_gradients(BoundaryType.TOP, 2.0)
        top_grad = mesh.get_boundary_gradients(BoundaryType.TOP)
        np.testing.assert_array_equal(top_grad, np.array([2.0, 2.0, 2.0]))
        
        # Verify that only boundary values changed
        assert mesh.state[1, 0] == 4.0  # Adjacent line unchanged
        assert mesh.state[1, 1] == 5.0  # Adjacent line unchanged
        assert mesh.state[1, 2] == 6.0  # Adjacent line unchanged
        
        # Test setting left gradient with array
        mesh.set_boundary_gradients(BoundaryType.LEFT, np.array([1.0, 2.0, 3.0]))
        left_grad = mesh.get_boundary_gradients(BoundaryType.LEFT)
        np.testing.assert_array_equal(left_grad, np.array([1.0, 2.0, 3.0]))
        
        # Test setting right gradient
        mesh.set_boundary_gradients(BoundaryType.RIGHT, -1.0)
        right_grad = mesh.get_boundary_gradients(BoundaryType.RIGHT)
        np.testing.assert_array_equal(right_grad, np.array([-1.0, -1.0, -1.0]))
        
        # Test setting bottom gradient
        mesh.set_boundary_gradients(BoundaryType.BOTTOM, np.array([0.5, 1.0, 1.5]))
        bottom_grad = mesh.get_boundary_gradients(BoundaryType.BOTTOM)
        np.testing.assert_array_equal(bottom_grad, np.array([0.5, 1.0, 1.5]))
    
    def test_shadow_state_initialization(self, sample_block):
        """Test that shadow state is properly initialized."""
        assert sample_block.shadow_state.shape == (3, 4)
        assert sample_block.shadow_state.dtype == np.float64
        assert np.all(sample_block.shadow_state == 0.0)
        assert sample_block.shadow_state is not sample_block._shadow_state  # Should be a view
    
    def test_set_shadow_state(self, sample_block):
        """Test setting shadow state with a valid numpy array."""
        new_shadow_state = np.array([[10.0, 20.0, 30.0, 40.0],
                                     [50.0, 60.0, 70.0, 80.0],
                                     [90.0, 100.0, 110.0, 120.0]])
        
        sample_block.set_shadow_state(new_shadow_state)
        
        # Check that the shadow state was updated
        assert np.array_equal(sample_block.shadow_state, new_shadow_state)
        # Check that it's the same array (in-place update)
        assert sample_block.shadow_state is not new_shadow_state
    
    def test_apply_method(self, sample_block):
        """Test the apply method with a callable function."""
        # Set up initial states
        sample_block.set_state(np.array([[1.0, 2.0, 3.0, 4.0],
                                         [5.0, 6.0, 7.0, 8.0],
                                         [9.0, 10.0, 11.0, 12.0]]))
        sample_block.set_shadow_state(np.array([[100.0, 200.0, 300.0, 400.0],
                                               [500.0, 600.0, 700.0, 800.0],
                                               [900.0, 1000.0, 1100.0, 1200.0]]))
        
        # Define a function that returns a copy of shadow state
        def copy_shadow_to_main(state, shadow_state):
            return shadow_state.copy()
        
                # Apply the function
        sample_block.apply(copy_shadow_to_main)
        
        # Check that the main state was updated with shadow state values
        expected_state = np.array([[100.0, 200.0, 300.0, 400.0],
                                   [500.0, 600.0, 700.0, 800.0],
                                   [900.0, 1000.0, 1100.0, 1200.0]])
        assert np.array_equal(sample_block.state, expected_state)
        
        # Check that shadow state remains unchanged
        assert np.array_equal(sample_block.shadow_state, expected_state)
    
    def test_apply_method_with_modification(self, sample_block):
        """Test the apply method with a function that modifies both states."""
        # Set up initial states
        sample_block.set_state(np.array([[1.0, 2.0, 3.0, 4.0],
                                         [5.0, 6.0, 7.0, 8.0],
                                         [9.0, 10.0, 11.0, 12.0]]))
        sample_block.set_shadow_state(np.array([[10.0, 20.0, 30.0, 40.0],
                                               [50.0, 60.0, 70.0, 80.0],
                                               [90.0, 100.0, 110.0, 120.0]]))
        
        # Define a function that returns the sum of main state and shadow state
        def add_shadow_to_main(state, shadow_state):
            return state + shadow_state
        
                # Apply the function
        sample_block.apply(add_shadow_to_main)
        
        # Check that the main state was updated correctly
        expected_state = np.array([[11.0, 22.0, 33.0, 44.0],
                                   [55.0, 66.0, 77.0, 88.0],
                                   [99.0, 110.0, 121.0, 132.0]])
        assert np.array_equal(sample_block.state, expected_state)
    
    def test_swap_method(self, sample_block):
        """Test the swap method that exchanges main and shadow states."""
        # Set up different initial states
        main_state = np.array([[1.0, 2.0, 3.0, 4.0],
                               [5.0, 6.0, 7.0, 8.0],
                               [9.0, 10.0, 11.0, 12.0]])
        shadow_state = np.array([[100.0, 200.0, 300.0, 400.0],
                                [500.0, 600.0, 700.0, 800.0],
                                [900.0, 1000.0, 1100.0, 1200.0]])
        
        sample_block.set_state(main_state)
        sample_block.set_shadow_state(shadow_state)
        
        # Store references to verify they're actually swapped
        original_main = sample_block.state.copy()
        original_shadow = sample_block.shadow_state.copy()
        
        # Perform swap
        sample_block.swap()
        
        # Check that states are now swapped
        assert np.array_equal(sample_block.state, original_shadow)
        assert np.array_equal(sample_block.shadow_state, original_main)
    
    def test_swap_method_preserves_data(self, sample_block):
        """Test that swap method preserves data integrity."""
        # Set up states with different values
        sample_block.set_state(np.array([[1.0, 2.0, 3.0, 4.0],
                                         [5.0, 6.0, 7.0, 8.0],
                                         [9.0, 10.0, 11.0, 12.0]]))
        sample_block.set_shadow_state(np.array([[0.1, 0.2, 0.3, 0.4],
                                               [0.5, 0.6, 0.7, 0.8],
                                               [0.9, 1.0, 1.1, 1.2]]))
        
        # Perform multiple swaps
        sample_block.swap()
        sample_block.swap()
        
        # After two swaps, states should be back to original
        expected_main = np.array([[1.0, 2.0, 3.0, 4.0],
                                  [5.0, 6.0, 7.0, 8.0],
                                  [9.0, 10.0, 11.0, 12.0]])
        expected_shadow = np.array([[0.1, 0.2, 0.3, 0.4],
                                    [0.5, 0.6, 0.7, 0.8],
                                    [0.9, 1.0, 1.1, 1.2]])
        
        assert np.array_equal(sample_block.state, expected_main)
        assert np.array_equal(sample_block.shadow_state, expected_shadow)
    
if __name__ == "__main__":
    pytest.main([__file__]) 
