# Sandwich Numerical Analysis Method

This project contains Python implementations of numerical analysis methods, specifically the "Sandwich" method for solving differential equations using a multi-block approach.

## Project Structure

- `sandwich_numerical/` - Main package directory
  - `__init__.py` - Package initialization
  - `sandwich.py` - Main implementation of the Sandwich class and related functions
- `tests/` - Test suite
  - `__init__.py` - Tests package
  - `test_sandwich.py` - Comprehensive pytest test suite
- `notebooks/` - Jupyter notebooks directory
  - `Sandwich.ipynb` - Original Jupyter notebook (for reference)
- `sandwich.py` - Root-level implementation with utility functions
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
from sandwich_numerical.sandwich import Sandwich
import numpy as np

# Create a gradient vector
block_size = (10, 10)  # (height, width)
grad_vec = np.linspace(0, 1, 3 * block_size[0] + 1)

# Create and use the Sandwich solver
mesh = Sandwich(
    block_size=block_size,
    grad_vec=grad_vec,
    grid_step=0.1,
    grad_factor=1.0
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

## Key Components

### Sandwich Class

The main class that implements the numerical solver:

- `__init__(block_size, grad_vec, grid_step, grad_factor)` - Initialize with block dimensions, gradient vector, grid step, and gradient factor
- `step()` - Perform one iteration of the numerical method
- `plot()` - Generate 3D surface plots of the current state
- `get_residual()` - Calculate the current residual (error)
- `get_displacement_array()` - Get the combined displacement data

### Core Functions

- `set_laplace_update()` - Apply Laplace operator for numerical integration
- `set_boundary_conditions_*_block()` - Set boundary conditions for each block type
- `transfer_data_inwards()` / `transfer_data_outwards()` - Handle data transfer between blocks

### Utility Functions (in root sandwich.py)

- `process_mesh()` - Run multiple iterations and track residuals
- `get_samples_df()` - Extract statistical data from the mesh
- `plot_samples_data()` - Create statistical visualizations
- `displacement_data_to_df()` - Convert displacement data to DataFrame format

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
- Uses finite difference methods (Laplace operator) for spatial discretization
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
│   └── sandwich.py              # Main Sandwich class implementation
├── tests/                       # Test suite
│   ├── __init__.py              # Tests package
│   └── test_sandwich.py         # Comprehensive pytest tests
├── notebooks/                   # Jupyter notebooks
│   └── Sandwich.ipynb           # Original Jupyter notebook
├── .github/                     # GitHub configuration
│   └── workflows/               # GitHub Actions workflows
│       └── test.yml             # CI workflow with coverage
├── sandwich.py                  # Root-level implementation with utilities
├── pyproject.toml               # Poetry configuration
└── README.md                    # This file
```

## License

This project is provided as-is for educational and research purposes.
