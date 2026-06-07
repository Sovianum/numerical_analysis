"""
Tests for the Sandwich numerical analysis package.
"""

import numpy as np
import pytest

from sandwich_numerical.sandwich import Sandwich


class TestSandwich:
    """Test cases for the Sandwich class."""

    @pytest.fixture
    def sample_mesh(self):
        """Create a generalized Sandwich mesh for testing."""
        block_size = (10, 10)
        grid_step = 0.1
        num_mid_blocks = 1
        grad_factors = [1.0, 1.0, 1.0]
        grad_vec = np.linspace(0, 1, (2 + num_mid_blocks) * block_size[0])

        return Sandwich(
            num_mid_blocks=num_mid_blocks,
            block_size=block_size,
            grad_vec=grad_vec,
            grid_step=grid_step,
            grad_factors=grad_factors,
        )

    @pytest.fixture
    def multi_mid_mesh(self):
        """Create a Sandwich mesh with multiple mid blocks for testing."""
        block_size = (8, 8)
        grid_step = 0.1
        num_mid_blocks = 3
        grad_factors = [1.0, 1.0, 1.5, 0.8, 1.0]
        grad_vec = np.linspace(0, 1, (2 + num_mid_blocks) * block_size[0])

        return Sandwich(
            num_mid_blocks=num_mid_blocks,
            block_size=block_size,
            grad_vec=grad_vec,
            grid_step=grid_step,
            grad_factors=grad_factors,
        )

    def test_creation(self, sample_mesh):
        """Test that generalized Sandwich objects can be created."""
        assert sample_mesh.block_size == (10, 10)
        assert sample_mesh.grid_step == 0.1
        assert sample_mesh.grad_factors == [1.0, 1.0, 1.0]
        assert sample_mesh.num_mid_blocks == 1
        assert sample_mesh.grad_vec.shape == (30,)

    def test_multi_mid_creation(self, multi_mid_mesh):
        """Test that Sandwich objects with multiple mid blocks can be created."""
        assert multi_mid_mesh.block_size == (8, 8)
        assert multi_mid_mesh.grid_step == 0.1
        assert multi_mid_mesh.grad_factors == [1.0, 1.0, 1.5, 0.8, 1.0]
        assert multi_mid_mesh.num_mid_blocks == 3
        assert multi_mid_mesh.grad_vec.shape == (40,)

    def test_state_arrays(self, sample_mesh):
        """Test that state arrays are properly initialized."""
        assert sample_mesh.bottom.state.shape == (10, 10)
        assert sample_mesh.mid_blocks[0].state.shape == (12, 10)
        assert sample_mesh.mid.state.shape == (12, 10)
        assert sample_mesh.top.state.shape == (10, 10)

    def test_multi_mid_state_arrays(self, multi_mid_mesh):
        """Test that state arrays are properly initialized for multiple mid blocks."""
        assert multi_mid_mesh.bottom.state.shape == (8, 8)
        assert multi_mid_mesh.top.state.shape == (8, 8)

        for mid_block in multi_mid_mesh.mid_blocks:
            assert mid_block.state.shape == (10, 8)

        assert len(multi_mid_mesh.mid_blocks) == 3

    def test_step_method(self, sample_mesh):
        """Test that the step method executes without error."""
        initial_residual = sample_mesh.get_residual()
        sample_mesh.step()
        new_residual = sample_mesh.get_residual()

        assert initial_residual == 0.0
        assert new_residual > 0.0
        assert isinstance(new_residual, (int, float))

    def test_multi_mid_step_method(self, multi_mid_mesh):
        """Test that the step method executes without error for multiple mid blocks."""
        initial_residual = multi_mid_mesh.get_residual()
        multi_mid_mesh.step()
        new_residual = multi_mid_mesh.get_residual()

        assert initial_residual == 0.0
        assert new_residual > 0.0
        assert isinstance(new_residual, (int, float))

    def test_displacement_array(self, sample_mesh):
        """Test that displacement array has correct shape."""
        displacement = sample_mesh.get_displacement_array()
        assert displacement.shape == (30, 10)

    def test_multi_mid_displacement_array(self, multi_mid_mesh):
        """Test that displacement array has correct shape for multiple mid blocks."""
        displacement = multi_mid_mesh.get_displacement_array()
        assert displacement.shape == (40, 8)

    def test_plot_method(self, sample_mesh):
        """Test that plot method returns a plotly figure."""
        fig = sample_mesh.plot()
        assert fig is not None

    def test_plot_method_multi_mid(self, multi_mid_mesh):
        """Test that plot method returns a plotly figure for multiple mid blocks."""
        fig = multi_mid_mesh.plot()
        assert fig is not None

    def test_invalid_even_mid_blocks(self):
        """Test that creating a Sandwich with even number of mid blocks raises."""
        block_size = (10, 10)
        num_mid_blocks = 2
        grad_vec = np.linspace(0, 1, (2 + num_mid_blocks) * block_size[0])

        with pytest.raises(ValueError, match="Number of mid blocks must be odd"):
            Sandwich(
                num_mid_blocks=num_mid_blocks,
                block_size=block_size,
                grad_vec=grad_vec,
                grid_step=0.1,
                grad_factors=[1.0, 1.0, 1.0, 1.0],
            )

    def test_invalid_generalized_grad_vec_length(self):
        """Test that wrong generalized grad_vec length raises an error."""
        block_size = (10, 10)
        num_mid_blocks = 3
        grad_vec = np.linspace(0, 1, 30)

        with pytest.raises(ValueError, match="grad_vec must have length 50, got 30"):
            Sandwich(
                num_mid_blocks=num_mid_blocks,
                block_size=block_size,
                grad_vec=grad_vec,
                grid_step=0.1,
                grad_factors=[1.0, 1.5, 0.8, 1.2, 1.0],
            )

    @pytest.mark.parametrize("grad_vec_length", [29, 31])
    def test_invalid_grad_vec_length(self, grad_vec_length):
        """Test that invalid gradient vector lengths are rejected."""
        block_size = (10, 10)
        grad_vec = np.linspace(0, 1, grad_vec_length)

        with pytest.raises(ValueError, match="grad_vec"):
            Sandwich(
                num_mid_blocks=1,
                block_size=block_size,
                grad_vec=grad_vec,
                grid_step=0.1,
                grad_factors=[1.0, 1.0, 1.0],
            )

    @pytest.mark.parametrize("grad_factors", [[1.0, 0.0, 1.0], [1.0, -1.0, 1.0]])
    def test_invalid_grad_factors(self, grad_factors):
        """Test that invalid gradient factors are rejected."""
        block_size = (10, 10)
        grad_vec = np.linspace(0, 1, 3 * block_size[0])

        with pytest.raises(ValueError, match="grad_factors"):
            Sandwich(
                num_mid_blocks=1,
                block_size=block_size,
                grad_vec=grad_vec,
                grid_step=0.1,
                grad_factors=grad_factors,
            )

    def test_invalid_grad_factors_length(self):
        """Test that invalid generalized gradient factor lengths are rejected."""
        block_size = (10, 10)
        grad_vec = np.linspace(0, 1, 3 * block_size[0])

        with pytest.raises(ValueError, match="grad_factors"):
            Sandwich(
                num_mid_blocks=1,
                block_size=block_size,
                grad_vec=grad_vec,
                grid_step=0.1,
                grad_factors=[1.0, 1.0],
            )

    @pytest.mark.parametrize("grid_step", [0.0, -0.1])
    def test_invalid_grid_step(self, grid_step):
        """Test that invalid grid steps are rejected."""
        block_size = (10, 10)
        grad_vec = np.linspace(0, 1, 3 * block_size[0])

        with pytest.raises(ValueError, match="grid_step"):
            Sandwich(
                num_mid_blocks=1,
                block_size=block_size,
                grad_vec=grad_vec,
                grid_step=grid_step,
                grad_factors=[1.0, 1.0, 1.0],
            )

    @pytest.mark.parametrize("block_size", [(1, 10), (10, 1)])
    def test_invalid_block_size(self, block_size):
        """Test that block sizes too small for gradients are rejected."""
        grad_vec = np.linspace(0, 1, 3 * block_size[0])

        with pytest.raises(ValueError, match="block_size"):
            Sandwich(
                num_mid_blocks=1,
                block_size=block_size,
                grad_vec=grad_vec,
                grid_step=0.1,
                grad_factors=[1.0, 1.0, 1.0],
            )


if __name__ == "__main__":
    pytest.main([__file__])
