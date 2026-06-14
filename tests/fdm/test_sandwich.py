"""
Tests for the Sandwich numerical analysis package.
"""

import numpy as np
import pytest

from sandwich_numerical.fdm.sandwich import (
    Sandwich,
    set_boundary_conditions_bottom_block,
    set_boundary_conditions_middle_block,
    set_boundary_conditions_top_block,
)
from sandwich_numerical.fdm.solver.laplace import set_laplace_update
from sandwich_numerical.fdm.solver.mesh_block import BoundaryType, MeshBlock
from sandwich_numerical.fdm.solver.mesh_utils import (
    copy_boundary_gradients,
    copy_boundary_values,
)


class ReferenceThreeLayerSandwich:
    """Reference implementation of the pre-generalized three-layer step."""

    def __init__(
        self,
        block_size: tuple[int, int],
        grad_vec: np.ndarray,
        grid_step: float,
        grad_factor: float,
    ) -> None:
        self.block_size = block_size
        self.block_height = block_size[0]
        self.grad_vec = grad_vec
        self.grid_step = grid_step
        self.grad_factor = grad_factor

        self.bottom = MeshBlock(block_size)
        self.mid = MeshBlock((block_size[0] + 2, block_size[1]))
        self.top = MeshBlock(block_size)
        self.blocks = [self.bottom, self.mid, self.top]

    def step(self) -> None:
        set_boundary_conditions_bottom_block(
            self.bottom,
            self.grad_vec[: self.block_height],
            self.grid_step,
        )
        set_boundary_conditions_middle_block(
            self.mid,
            self.grad_vec[self.block_height - 1 : 2 * self.block_height + 1],
            self.grid_step,
        )
        set_boundary_conditions_top_block(
            self.top,
            self.grad_vec[2 * self.block_height : 3 * self.block_height],
            self.grid_step,
        )

        set_laplace_update(self.bottom._state)
        set_laplace_update(self.top._state)

        copy_boundary_values(
            self.bottom,
            BoundaryType.TOP,
            self.mid,
            BoundaryType.BOTTOM,
        )
        copy_boundary_values(
            self.top,
            BoundaryType.BOTTOM,
            self.mid,
            BoundaryType.TOP,
        )

        set_laplace_update(self.mid._state)

        grad_scale = 1 / self.grad_factor
        copy_boundary_gradients(
            self.mid,
            BoundaryType.BOTTOM,
            self.bottom,
            BoundaryType.TOP,
            grad_scale,
        )
        copy_boundary_gradients(
            self.mid,
            BoundaryType.TOP,
            self.top,
            BoundaryType.BOTTOM,
            grad_scale,
        )

    def get_displacement_array(self) -> np.ndarray:
        return np.concatenate(
            (
                self.bottom._state,
                self.mid._state[1:-1],
                self.top._state,
            )
        )

    def get_residual(self) -> float:
        results = []
        for block in self.blocks:
            res_block = np.copy(block._state)
            set_laplace_update(res_block)
            results.append(
                float(np.sum(np.abs(block._state[1:-1, 1:-1] - res_block[1:-1, 1:-1])))
            )
        return sum(results)


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
        assert sample_mesh.gradient_relaxation == 1.0
        assert sample_mesh.enforce_overlap_continuity is True
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

    @pytest.mark.parametrize("grad_factor", [1.0, 1000.0])
    def test_three_layer_solver_matches_reference(self, grad_factor):
        """Generalized one-middle-block mode must reproduce the old solver."""
        block_size = (7, 11)
        grid_step = 0.05
        mesh_height = 3 * block_size[0]
        grad_vec = np.sin((2 * np.pi) / (mesh_height - 1) * np.arange(mesh_height))

        reference = ReferenceThreeLayerSandwich(
            block_size=block_size,
            grad_vec=grad_vec,
            grid_step=grid_step,
            grad_factor=grad_factor,
        )
        generalized = Sandwich(
            num_mid_blocks=1,
            block_size=block_size,
            grad_vec=grad_vec,
            grid_step=grid_step,
            grad_factors=[1.0, grad_factor, 1.0],
            enforce_overlap_continuity=False,
        )

        for _ in range(25):
            reference.step()
            generalized.step()

        np.testing.assert_allclose(
            generalized.get_displacement_array(),
            reference.get_displacement_array(),
            rtol=0,
            atol=1e-15,
        )
        assert generalized.get_residual() == pytest.approx(
            reference.get_residual(),
            abs=1e-15,
        )

    def test_gradient_relaxation_blends_copied_gradients(self):
        """Relaxed transfer should blend new interface gradients with current ones."""
        block_size = (7, 11)
        grid_step = 0.05
        mesh_height = 3 * block_size[0]
        grad_vec = np.sin((2 * np.pi) / (mesh_height - 1) * np.arange(mesh_height))
        relaxation = 0.25
        mesh = Sandwich(
            num_mid_blocks=1,
            block_size=block_size,
            grad_vec=grad_vec,
            grid_step=grid_step,
            grad_factors=[1000.0, 1.0, 1000.0],
            gradient_relaxation=relaxation,
        )

        mesh._set_boundary_conditions()
        mesh._run_laplace_inward_with_value_transfer()

        target = mesh.blocks[0]
        current_gradients = target.get_boundary_gradients(BoundaryType.TOP)
        desired_gradients = mesh.blocks[1].get_boundary_gradients(
            BoundaryType.BOTTOM
        ) * mesh._get_grad_scale(1, 0)
        expected_gradients = current_gradients + relaxation * (
            desired_gradients - current_gradients
        )

        mesh._copy_boundary_gradients(
            source_index=1,
            source_boundary=BoundaryType.BOTTOM,
            target_index=0,
            target_boundary=BoundaryType.TOP,
        )

        np.testing.assert_allclose(
            target.get_boundary_gradients(BoundaryType.TOP),
            expected_gradients,
            rtol=0,
            atol=1e-15,
        )

    def test_overlap_continuity_averages_duplicate_coordinates(self):
        """Overlap projection must average duplicate x2 coordinate rows."""
        mesh = Sandwich(
            num_mid_blocks=3,
            block_size=(4, 5),
            grad_vec=np.zeros(20),
            grid_step=0.1,
            grad_factors=[1.0, 1.0, 1.0, 1.0, 1.0],
            enforce_overlap_continuity=True,
        )
        mesh.blocks[1]._state[-2, :] = 2.0
        mesh.blocks[2]._state[0, :] = 4.0
        mesh.blocks[1]._state[-1, :] = 6.0
        mesh.blocks[2]._state[1, :] = 10.0

        mesh._enforce_overlap_continuity()

        np.testing.assert_allclose(mesh.blocks[1]._state[-2, :], 3.0)
        np.testing.assert_allclose(mesh.blocks[2]._state[0, :], 3.0)
        np.testing.assert_allclose(mesh.blocks[1]._state[-1, :], 8.0)
        np.testing.assert_allclose(mesh.blocks[2]._state[1, :], 8.0)

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

    @pytest.mark.parametrize("gradient_relaxation", [0.0, -0.1, 1.1, np.inf])
    def test_invalid_gradient_relaxation(self, gradient_relaxation):
        """Test that invalid gradient relaxation values are rejected."""
        block_size = (10, 10)
        grad_vec = np.linspace(0, 1, 3 * block_size[0])

        with pytest.raises(ValueError, match="gradient_relaxation"):
            Sandwich(
                num_mid_blocks=1,
                block_size=block_size,
                grad_vec=grad_vec,
                grid_step=0.1,
                grad_factors=[1.0, 1.0, 1.0],
                gradient_relaxation=gradient_relaxation,
            )

    def test_invalid_enforce_overlap_continuity(self):
        """Test that overlap continuity flag must be boolean."""
        block_size = (10, 10)
        grad_vec = np.linspace(0, 1, 3 * block_size[0])

        with pytest.raises(TypeError, match="enforce_overlap_continuity"):
            Sandwich(
                num_mid_blocks=1,
                block_size=block_size,
                grad_vec=grad_vec,
                grid_step=0.1,
                grad_factors=[1.0, 1.0, 1.0],
                enforce_overlap_continuity=1,
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
