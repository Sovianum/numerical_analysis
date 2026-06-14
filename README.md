# Sandwich Numerical Analysis Method

This project contains Python implementations of numerical analysis methods, specifically the "Sandwich" method for solving differential equations using a multi-block approach.

## Project Structure

- `sandwich_numerical/` - Main package directory
  - `__init__.py` - Package initialization
  - `fdm/` - Finite difference method implementation
    - `sandwich.py` - Main FDM implementation of the Sandwich class and related functions
    - `integration.py` - Reusable FDM integration scenario runner
    - `solver/` - FDM mesh blocks, Laplace update, and mesh utilities
- `tests/` - Test suite
  - `__init__.py` - Tests package
  - `fdm/` - FDM pytest test suite
- `scripts/` - Reproducible command-line runners
  - `fdm/` - Generate FDM integration CSV/PNG artifacts
- `pyproject.toml` - Poetry configuration and dependencies
- `README.md` - This documentation file

## Features

The Sandwich class implements:

- **Multi-block numerical solver** with bottom, middle, and top blocks
- **Boundary condition handling** for each block
- **Laplace operator updates** for numerical integration
- **Data transfer mechanisms** between blocks
- **Visualization tools** using Plotly for 3D surface plots
- **Convergence monitoring** with residual calculations
- **Comprehensive test coverage** with pytest

## Installation

1. Clone or download this repository
2. Install Poetry if you haven't already:
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```
3. Install the package and dependencies:
   ```bash
   poetry install
   ```
4. Activate the virtual environment:
   ```bash
   poetry shell
   ```

## Usage

### Basic Usage

```python
from sandwich_numerical.fdm.sandwich import Sandwich
import numpy as np

# Create a gradient vector
block_size = (10, 10)  # (height, width)
grad_vec = np.linspace(0, 1, 3 * block_size[0])

# Create and use the Sandwich solver
mesh = Sandwich(
    num_mid_blocks=1,
    block_size=block_size,
    grad_vec=grad_vec,
    grid_step=0.1,
    grad_factors=[1.0, 1.0, 1.0]  # bottom, middle, top
)

# Run iterations
for i in range(100):
    mesh.step()
    if i % 10 == 0:
        print(f"Iteration {i}: Residual = {mesh.get_residual():.6f}")
```

### Running Tests

```bash
# Run all tests
poetry run pytest tests/ -v

# Run tests with coverage
poetry run pytest tests/ --cov=sandwich_numerical --cov-report=html

# Run tests with coverage (terminal output)
poetry run pytest tests/ --cov=sandwich_numerical --cov-report=term-missing
```

### Sandwich Integration Scenarios

The reproducible integration artifacts compare FDM and FEM solutions for a
rectangular sandwich domain with `grid_step=0.005`, `block_height=21`, and
`block_width=3000` unless overridden from the CLI.

Configured layer scenarios:

- `grad_factors_1_1_1` - 3 layers with gradient factors `(1.0, 1.0, 1.0)`.
- `grad_factors_1_1000_1` - 3 layers with a stiff middle layer, gradient factors `(1.0, 1000.0, 1.0)`.
- `grad_factors_1000_1_1000_1_1000` - 5 alternating layers with gradient factors `(1000.0, 1.0, 1000.0, 1.0, 1000.0)`.

Each scenario is run with two left-boundary load profiles:

- `sine` - `sin(2*pi*i/(height - 1))`, zero at the lower and upper boundaries, with max absolute value close to 1 on the discrete mesh.
- `parabolic_zero_mean` - `(x2^2 - L^2/12)` normalized by its max absolute value; this keeps the profile symmetric and zero-mean while making `max |load| = 1`.

Generate local artifacts with:

```bash
poetry run python scripts/run_sandwich_artifacts.py \
  --mode local \
  --all-cases \
  --output-dir artifacts/local_sandwich_integration
```

Each generated run directory contains a short `README.md` with the layer count,
gradient factors, load profile, and output layout for that run.

## Key Components

### Sandwich Class

The main class that implements the numerical solver:

- `__init__(num_mid_blocks, block_size, grad_vec, grid_step, grad_factors)` - Initialize with block dimensions, gradient vector, grid step, and gradient factors for each block
- `step()` - Perform one iteration of the numerical method
- `plot()` - Generate 3D surface plots of the current state
- `get_residual()` - Calculate the current residual (error)
- `get_displacement_array()` - Get the combined displacement data

### Core Functions

- `set_laplace_update()` - Apply Laplace operator for numerical integration
- `set_boundary_conditions_*_block()` - Set boundary conditions for each block type
- `copy_boundary_values()` / `copy_boundary_gradients()` - Handle data transfer between blocks

### Integration Utilities

- `solve_case()` - Run a configured FDM scenario and collect residual, sample, and displacement data
- `build_gradient_vector()` - Build the configured boundary-gradient profile
- `displacement_data_to_df()` - Convert displacement data to DataFrame format
- `write_solution_csvs()` - Write reproducible CSV artifacts for a scenario

## Testing

The project includes a comprehensive test suite:

### Test Coverage

- **TestSandwich**: Tests for basic object creation, state arrays, and step methods
- **TestSandwich**: Tests for displacement arrays and plotting functionality

### Running Tests Locally

```bash
# Basic test run
poetry run pytest tests/ -v

# With coverage report
poetry run pytest tests/ --cov=sandwich_numerical --cov-report=html

# Coverage report will be generated in htmlcov/ directory
```

## Continuous Integration

This project uses GitHub Actions for automated testing on every commit:

### GitHub Actions Workflows

- **`.github/workflows/test.yml`** - Full workflow with coverage reporting and Codecov integration

### CI Features

- **Python Testing**: Tests against Python 3.11
- **Automatic Triggering**: Runs on every push and pull request
- **Dependency Caching**: Fast builds with Poetry virtual environment caching
- **Coverage Reporting**: Generates coverage reports for quality monitoring

## Visualization

The project generates several types of plots:

1. **State Visualization** - 3D surface plots showing the state of each block
2. **Samples Data** - Bar charts showing statistical summaries
3. **Residuals Convergence** - Line plots showing error reduction over iterations

All plots are saved as interactive HTML files that can be opened in any web browser.

## Mathematical Background

The Sandwich method is a numerical technique for solving partial differential equations that:

- Divides the computational domain into multiple blocks
- Applies appropriate boundary conditions at block interfaces
- Uses the finite difference method (Laplace operator) for spatial discretization
- Transfers information between blocks to maintain solution continuity
- Iteratively refines the solution until convergence

## Dependencies

The package automatically manages all dependencies through Poetry. Main dependencies include:

- **NumPy** - Numerical computing and array operations
- **Pandas** - Data manipulation and analysis
- **Plotly** - Interactive plotting and visualization
- **pytest** - Testing framework
- **pytest-cov** - Coverage reporting

## Development

### Using Poetry

```bash
# Install development dependencies
poetry install

# Run tests
poetry run pytest tests/ -v

# Run tests with coverage
poetry run pytest tests/ --cov=sandwich_numerical --cov-report=html

# Format code
poetry run black sandwich_numerical/ tests/

# Lint code
poetry run flake8 sandwich_numerical/ tests/
poetry run mypy sandwich_numerical/

# Build package
poetry build
```

### Project Structure

```
numerical_analysis/
├── sandwich_numerical/           # Main package
│   ├── __init__.py              # Package initialization
│   ├── fdm/                     # Finite difference method implementation
│   │   ├── integration.py       # Reusable FDM integration scenario runner
│   │   ├── sandwich.py          # Main FDM Sandwich class implementation
│   │   └── solver/              # FDM solver primitives
├── tests/                       # Test suite
│   ├── __init__.py              # Tests package
│   └── fdm/                     # FDM pytest tests
├── scripts/                     # Command-line runners
│   └── fdm/
│       └── run_sandwich_integration.py
├── .github/                     # GitHub configuration
│   └── workflows/               # GitHub Actions workflows
│       └── test.yml             # CI workflow with coverage
├── pyproject.toml               # Poetry configuration
└── README.md                    # This file
```

## License

This project is provided as-is for educational and research purposes.
