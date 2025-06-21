"""Tests for the phasorpy command line interface."""

from click.testing import CliRunner

from phasorpy import __version__
from phasorpy.cli import main
from phasorpy.utils import versions


def test_version():
    """Test ``python -m phasorpy --version``."""
    runner = CliRunner()
    result = runner.invoke(main, ['--version'])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_versions():
    """Test ``python -m phasorpy versions``."""
    runner = CliRunner()
    result = runner.invoke(main, ['versions', '--verbose'])
    assert result.exit_code == 0
    assert result.output.strip() == versions(verbose=True)


def test_fetch():
    """Test ``python -m phasorpy fetch``."""
    runner = CliRunner()
    result = runner.invoke(main, ['fetch', 'simfcs.r64'])
    assert result.exit_code == 0
    assert result.output.strip().endswith('simfcs.r64')


def test_fret():
    """Test ``python -m phasorpy fret``."""
    runner = CliRunner()
    result = runner.invoke(main, ['fret', '--hide'])
    assert result.exit_code == 0


def test_lifetime():
    """Test ``python -m phasorpy lifetime``."""
    runner = CliRunner()
    result = runner.invoke(main, ['lifetime', '--hide'])
    assert result.exit_code == 0
    result = runner.invoke(main, ['lifetime', '-f 60', '-l 4.2', '--hide'])
    assert result.exit_code == 0
    result = runner.invoke(
        main, ['lifetime', '-l 4.2', '-l 1.0', '-a 0.6', '-a 0.4', '--hide']
    )
    assert result.exit_code == 0


# mypy: allow-untyped-defs, allow-untyped-calls
# mypy: disable-error-code="arg-type"
