"""Tests for the phasorpy command line interface."""

from click.testing import CliRunner

from phasorpy import __version__, versions
from phasorpy.cli import main


def test_version():
    """Test ``python -m phasorpy --version``."""
    runner = CliRunner()
    result = runner.invoke(main, ['--version'])

    assert result.exit_code == 0
    assert __version__ in result.output


def test_versions():
    """Test ``python -m phasorpy --versions``."""
    runner = CliRunner()
    result = runner.invoke(main, ['--versions'])

    assert result.exit_code == 0
    assert result.output.strip() == versions()
