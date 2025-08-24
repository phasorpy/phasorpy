# PhasorPy Development Instructions

PhasorPy is an open-source Python library for the analysis of fluorescence lifetime and hyperspectral images using the phasor approach. It includes Cython extensions for performance-critical computations.

**ALWAYS reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.**

## Quick Validation Test
**Before making any changes, run this validation sequence to check environment:**
```bash
# Test import, CLI, and basic functionality
python -c "import sys; sys.path.insert(0, 'src'); import phasorpy; print('✓ Import OK')"
PYTHONPATH=src python -m phasorpy versions | head -3
PYTHONPATH=src python -m pytest tests/test__utils.py::test_phasor_to_polar_scalar -q
echo "✓ Environment validated"
```

## Working Effectively

### Bootstrap and Install Dependencies
```bash
# Install system dependencies (Ubuntu/Debian)
sudo apt update
sudo apt install -y python3-numpy python3-scipy python3-matplotlib python3-click python3-pytest python3-sphinx cython3

# Install development dependencies (pip may timeout due to network issues)
python -m pip install --user --timeout=300 --retries 5 -r requirements_dev.txt

# If pip fails due to network timeouts, use system packages:
sudo apt install -y python3-pip python3-setuptools python3-wheel python3-build
```

### Build the Package
```bash
# CRITICAL: Cython compilation may fail with system cython3 (version 3.0.8)
# Package requires Cython>=3.1.0 but system may have older version

# Method 1: Install in editable mode (PREFERRED - builds extensions automatically)
python -m pip install --user --timeout=600 -v --editable .
# NEVER CANCEL: Build takes 2-5 minutes. TIMEOUT: 10+ minutes.

# Method 2: Build Cython extensions manually (if pip install fails)
python setup.py build_ext --inplace
# NEVER CANCEL: Cython compilation takes 1-3 minutes. May fail with old Cython.

# Method 3: Use without Cython extensions (limited functionality)
export PYTHONPATH=src  # For testing basic imports only
```

### Run Tests
```bash
# Run full test suite
python -m pytest -v
# NEVER CANCEL: Test suite takes 3-8 minutes. TIMEOUT: 15+ minutes.

# Run tests with coverage
python -m pytest -v --cov=phasorpy --cov-report=html tests
# NEVER CANCEL: Coverage analysis takes 5-10 minutes. TIMEOUT: 20+ minutes.

# Run specific test file
python -m pytest tests/test_phasor.py -v

# Run single test
python -m pytest tests/test__utils.py::test_phasor_to_polar_scalar -v
```

### Build Documentation
```bash
cd docs
make clean
make dirhtml
# REQUIRES: sphinx-gallery, numpydoc, and other doc dependencies
# NEVER CANCEL: Documentation build takes 8-15 minutes. TIMEOUT: 30+ minutes.
# NOTE: Will fail without pip-installed doc dependencies (sphinx-gallery, etc.)

# Test basic sphinx availability
sphinx-build --help  # Check if Sphinx is installed
```

## Working with Limited Dependencies

### What Works with System Packages Only
- **Basic imports**: `import phasorpy` 
- **CLI interface**: `python -m phasorpy versions`, `python -m phasorpy --help`
- **Utility functions**: Many functions in `phasorpy._utils`
- **Some tests**: Basic tests in `tests/test__utils.py` (12/13 pass)

### What Requires Full pip Installation
- **Cython extensions**: Performance-critical functions in `phasorpy._phasorpy`
- **Full phasor analysis**: Functions in `phasorpy.phasor` module
- **File I/O**: Reading/writing various microscopy formats
- **Documentation building**: Requires sphinx-gallery, numpydoc, etc.
- **Complete test suite**: Many tests need missing dependencies
- **Code quality tools**: black, mypy, isort, codespell

## CLI Interface
```bash
# Test CLI (requires package installation or PYTHONPATH=src)
python -m phasorpy --help
python -m phasorpy versions
python -m phasorpy lifetime --help
python -m phasorpy fret --hide
```

## Validation

### CRITICAL Build Requirements
- **Python 3.11+** (tested with 3.12)
- **numpy>=1.26.0** (for Cython compilation)
- **Cython>=3.1.0** (system cython3=3.0.8 TOO OLD - causes compiler crashes)
- **Network connectivity** for pip installs (may timeout - use system packages as fallback)

### Known Issues and Workarounds
- **Pip timeouts**: Network connectivity issues prevent reliable pip installations. Use system packages when possible.
- **Cython version**: System cython3 (3.0.8) is too old and causes "Compiler crash in AnalyseDeclarationsTransform". Requires Cython>=3.1.0.
- **License configuration**: Fixed pyproject.toml license field format for newer setuptools.
- **Missing Cython extensions**: Many tests and full functionality require `phasorpy._phasorpy` Cython module. Basic imports and CLI work without it.
- **Documentation dependencies**: Building docs requires sphinx-gallery, numpydoc, and other pip-only dependencies.

### Manual Validation Steps
**ALWAYS run these validation steps after making changes:**

1. **Import test**: `python -c "import sys; sys.path.insert(0, 'src'); import phasorpy; print('OK')"`
2. **CLI test**: `PYTHONPATH=src python -m phasorpy versions`
3. **Single test** (without Cython dependency): `PYTHONPATH=src python -m pytest tests/test__utils.py::test_phasor_to_polar_scalar -v`
4. **Multiple tests** (some may fail): `PYTHONPATH=src python -m pytest tests/test__utils.py::test_phasor_to_polar_scalar tests/test__utils.py::test_parse_kwargs -v`
4. **Build test** (if extensions work): `python setup.py build_ext --inplace`

**Note**: Some tests require Cython extensions (`phasorpy._phasorpy`) and will fail if extensions are not built. Tests in `tests/test__utils.py` mostly work without extensions, but many other tests require missing pip dependencies (pooch, xarray, etc.).

### Pre-commit Quality Checks
**ALWAYS run these before committing changes (may require pip-installed tools):**
```bash
# Code formatting and style (if available)
python -m black --check src/phasorpy tutorials docs
python -m isort --check src/phasorpy tutorials

# Type checking (if mypy available)
python -m mypy

# Spell checking (if codespell available)
python -m codespell

# Documentation style (if blackdoc available)
python -m blackdoc --check src/phasorpy

# NOTE: These tools may not be available with system packages
# Install with: pip install black isort mypy codespell blackdoc
```

### CI/CD Integration
The GitHub Actions workflow (`.github/workflows/run-tests.yml`) includes:
- **Test matrix**: Ubuntu, Windows, macOS with Python 3.11-3.13
- **Build wheels**: Uses cibuildwheel for distribution
- **Static analysis**: black, mypy, isort, codespell
- **Documentation**: Builds and verifies docs

## Common Tasks

### Repository Structure
```
.
├── README.md                  # Project overview
├── pyproject.toml            # Project configuration and dependencies
├── setup.py                  # Build script for Cython extensions
├── requirements_dev.txt      # Development dependencies
├── src/phasorpy/            # Main package source
│   ├── __init__.py
│   ├── cli.py               # Command-line interface
│   ├── _phasorpy.pyx        # Cython extension (performance critical)
│   ├── phasor.py            # Core phasor analysis
│   ├── lifetime.py          # Lifetime analysis
│   └── io/                  # File I/O for various formats
├── tests/                   # Test suite
├── docs/                    # Sphinx documentation
├── tutorials/               # Example notebooks/scripts
└── .github/                 # CI/CD and project templates
```

### Key Files and Directories
- **src/phasorpy/_phasorpy.pyx**: Cython extension requiring compilation
- **requirements_dev.txt**: Complete list of development dependencies
- **pyproject.toml**: pytest, mypy, black, and build configuration
- **docs/conf.py**: Sphinx documentation configuration
- **.pre-commit-config.yaml**: Code quality hooks

### Development Workflow
1. **Fork and clone** the repository
2. **Create branch**: `git checkout -b feature-branch-name`
3. **Install dependencies** using methods above
4. **Make changes** with minimal modifications
5. **Run validation** steps listed above
6. **Run quality checks** before committing
7. **Test documentation** if docs changed: `cd docs && make dirhtml`
8. **Commit and push** for pull request

### Expected Timing (for timeout configuration)
- **Package install**: 2-5 minutes
- **Cython compilation**: 1-3 minutes  
- **Full test suite**: 3-8 minutes
- **Test with coverage**: 5-10 minutes
- **Documentation build**: 8-15 minutes
- **Single test**: <1 minute
- **Quality checks**: 1-2 minutes

### Troubleshooting
- **Import errors**: Check PYTHONPATH=src or reinstall in editable mode
- **Cython compilation fails**: Ensure Cython>=3.1.0 or use system packages
- **Network timeouts**: Use system packages: `sudo apt install python3-package-name`
- **Test failures**: Check if related to missing Cython extensions
- **Documentation build fails**: Install sphinx dependencies: `sudo apt install python3-sphinx`

## File Format Support
PhasorPy supports reading various fluorescence microscopy file formats through specialized libraries:
- **Becker & Hickl SPC**: via `sdtfile`
- **PicoQuant**: via `ptufile`  
- **Leica**: via `liffile`
- **OME-TIFF**: via `tifffile`
- **SimFCS**: via custom readers

Install format-specific dependencies as needed:
```bash
pip install --user sdtfile ptufile liffile lfdfiles pawflim
```