"""
Tests for the Sandwich numerical analysis package.
"""

import pytest
import numpy as np
from sandwich_numerical.sandwich import Sandwich


class TestSandwich:
    """Test cases for the Sandwich class."""
    
    @pytest.fixture
    def sample_mesh(self):
        """Create a sample Sandwich mesh for testing."""
        block_size = (10, 10)  # (height, width) tuple
        grid_step = 0.1
        grad_factor = 1.0
    
        # Create simple gradient vector - should be 1D with length 3*block_height + 1
        grad_vec = np.linspace(0, 1, 3 * block_size[0] + 1)
    
        return Sandwich(
            block_size=block_size,
            grad_vec=grad_vec,
            grid_step=grid_step,
            grad_factor=grad_factor
        )
    
    def test_creation(self, sample_mesh):
        """Test that Sandwich objects can be created."""
        assert sample_mesh.block_size == (10, 10)
        assert sample_mesh.grid_step == 0.1
        assert sample_mesh.grad_factor == 1.0
        assert sample_mesh.grad_vec.shape == (31,)  # 3*block_height + 1 = 3*10 + 1 = 31
    
    def test_state_arrays(self, sample_mesh):
        """Test that state arrays are properly initialized."""
        assert sample_mesh.bottom.state.shape == (10, 10)
        assert sample_mesh.mid.state.shape == (12, 10)  # +2 rows for boundary conditions
        assert sample_mesh.top.state.shape == (10, 10)
        assert sample_mesh.bottom.next_state.shape == (10, 10)
        assert sample_mesh.mid.next_state.shape == (12, 10)  # +2 rows for boundary conditions
        assert sample_mesh.top.next_state.shape == (10, 10)
    
    def test_step_method(self, sample_mesh):
        """Test that the step method executes without error."""
        initial_residual = sample_mesh.get_residual()
        sample_mesh.step()
        new_residual = sample_mesh.get_residual()
        
        # Initial residual should be 0.0 (all arrays start as zeros)
        assert initial_residual == 0.0
        # After one step, residual should be non-zero as the system evolves
        assert new_residual > 0.0
        # Step method should execute without error
        assert isinstance(new_residual, (int, float))
    
    def test_displacement_array(self, sample_mesh):
        """Test that displacement array has correct shape."""
        displacement = sample_mesh.get_displacement_array()
        assert displacement.shape == (30, 10)  # 3 blocks * 10 rows
    
    def test_plot_method(self, sample_mesh):
        """Test that plot method returns a plotly figure."""
        fig = sample_mesh.plot()
        assert fig is not None
        # Additional plot-specific assertions could be added here





if __name__ == "__main__":
    pytest.main([__file__]) 
