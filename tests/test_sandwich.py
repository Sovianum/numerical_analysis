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
        grad_factors = [1.0]  # Single mid block
    
        # Create simple gradient vector - should be 1D with length (2 + num_mid_blocks) * block_height + 1
        # For 1 mid block: (2 + 1) * 10 + 1 = 31
        grad_vec = np.linspace(0, 1, (2 + len(grad_factors)) * block_size[0] + 1)
    
        return Sandwich(
            block_size=block_size,
            grad_vec=grad_vec,
            grid_step=grid_step,
            grad_factors=grad_factors
        )
    
    @pytest.fixture
    def multi_mid_mesh(self):
        """Create a Sandwich mesh with multiple mid blocks for testing."""
        block_size = (8, 8)  # (height, width) tuple
        grid_step = 0.1
        grad_factors = [1.0, 1.5, 0.8]  # Three mid blocks
    
        # Create gradient vector for 5 total blocks: (2 + 3) * 8 + 1 = 41
        grad_vec = np.linspace(0, 1, (2 + len(grad_factors)) * block_size[0] + 1)
    
        return Sandwich(
            block_size=block_size,
            grad_vec=grad_vec,
            grid_step=grid_step,
            grad_factors=grad_factors
        )
    
    def test_creation(self, sample_mesh):
        """Test that Sandwich objects can be created."""
        assert sample_mesh.block_size == (10, 10)
        assert sample_mesh.grid_step == 0.1
        assert sample_mesh.grad_factors == [1.0]
        assert sample_mesh.num_mid_blocks == 1
        assert sample_mesh.grad_vec.shape == (31,)  # (2 + 1) * 10 + 1 = 31
    
    def test_multi_mid_creation(self, multi_mid_mesh):
        """Test that Sandwich objects with multiple mid blocks can be created."""
        assert multi_mid_mesh.block_size == (8, 8)
        assert multi_mid_mesh.grid_step == 0.1
        assert multi_mid_mesh.grad_factors == [1.0, 1.5, 0.8]
        assert multi_mid_mesh.num_mid_blocks == 3
        assert multi_mid_mesh.grad_vec.shape == (41,)  # (2 + 3) * 8 + 1 = 41
    
    def test_state_arrays(self, sample_mesh):
        """Test that state arrays are properly initialized."""
        assert sample_mesh.bottom.state.shape == (10, 10)
        assert sample_mesh.mid_blocks[0].state.shape == (12, 10)  # +2 rows for boundary conditions
        assert sample_mesh.top.state.shape == (10, 10)
    
    def test_multi_mid_state_arrays(self, multi_mid_mesh):
        """Test that state arrays are properly initialized for multiple mid blocks."""
        assert multi_mid_mesh.bottom.state.shape == (8, 8)
        assert multi_mid_mesh.top.state.shape == (8, 8)
        
        # All mid blocks should have padding
        for mid_block in multi_mid_mesh.mid_blocks:
            assert mid_block.state.shape == (10, 8)  # +2 rows for boundary conditions
        
        assert len(multi_mid_mesh.mid_blocks) == 3
    
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
    
    def test_multi_mid_step_method(self, multi_mid_mesh):
        """Test that the step method executes without error for multiple mid blocks."""
        initial_residual = multi_mid_mesh.get_residual()
        multi_mid_mesh.step()
        new_residual = multi_mid_mesh.get_residual()
        
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
    
    def test_multi_mid_displacement_array(self, multi_mid_mesh):
        """Test that displacement array has correct shape for multiple mid blocks."""
        displacement = multi_mid_mesh.get_displacement_array()
        # 5 blocks * 8 rows = 40 rows
        assert displacement.shape == (40, 8)
    
    def test_plot_method(self, sample_mesh):
        """Test that plot method returns a plotly figure."""
        fig = sample_mesh.plot()
        assert fig is not None
        # Additional plot-specific assertions could be added here
    
    def test_plot_method_multi_mid(self, multi_mid_mesh):
        """Test that plot method returns a plotly figure for multiple mid blocks."""
        fig = multi_mid_mesh.plot()
        assert fig is not None
    
    def test_invalid_even_mid_blocks(self):
        """Test that creating a Sandwich with even number of mid blocks raises an error."""
        block_size = (10, 10)
        grid_step = 0.1
        grad_factors = [1.0, 1.5]  # Even number of mid blocks
        
        # Create gradient vector for 4 total blocks: (2 + 2) * 10 + 1 = 41
        grad_vec = np.linspace(0, 1, (2 + len(grad_factors)) * block_size[0] + 1)
        
        with pytest.raises(ValueError, match="Number of mid blocks must be odd"):
            Sandwich(
                block_size=block_size,
                grad_vec=grad_vec,
                grid_step=grid_step,
                grad_factors=grad_factors
            )
    
    def test_invalid_grad_vec_length(self):
        """Test that creating a Sandwich with wrong grad_vec length raises an error."""
        block_size = (10, 10)
        grid_step = 0.1
        grad_factors = [1.0, 1.5, 0.8]  # Three mid blocks
        
        # Wrong length gradient vector
        grad_vec = np.linspace(0, 1, 30)  # Should be 51 for (2 + 3) * 10 + 1
        
        with pytest.raises(ValueError, match="grad_vec must have length 51, got 30"):
            Sandwich(
                block_size=block_size,
                grad_vec=grad_vec,
                grid_step=grid_step,
                grad_factors=grad_factors
            )


if __name__ == "__main__":
    pytest.main([__file__]) 
