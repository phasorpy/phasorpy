"""Tests for the phasorpy.datasets module."""

import os

import pytest

from phasorpy.datasets import DATA_ON_GITHUB, fetch

# skip large downloads by default
SKIP_LARGE = bool(int(os.environ.get('SKIP_LARGE', 1)))


@pytest.mark.skipif(not DATA_ON_GITHUB, reason='not using GitHub Actions')
def test_data_on_github():
    """Test data files on GitHub."""
    assert DATA_ON_GITHUB


def test_fetch():
    """Test fetch file."""
    name = 'simfcs.r64'
    filename = fetch(name)
    assert filename.endswith(name)
    assert os.path.exists(filename)
    filename = fetch(name, return_scalar=False)
    assert filename[0].endswith(name)


def test_fetch_inzip():
    """Test fetch file from ZIP."""
    filename = fetch('simfcs.b&h')
    assert os.path.exists(filename)


def test_fetch_zip():
    """Test fetch ZIP file."""
    filename = fetch('flimage.int.bin.zip', extract_dir='unzipped')[0]
    assert os.path.exists(filename)
    assert 'unzipped' in filename


def test_fetch_compound():
    """Test fetch compound file in ZIP."""
    filename = fetch('simfcs_1000.int')
    assert os.path.exists(filename.replace('int', 'mod'))


def test_fetch_nonexistent():
    """Test fetch non-existent file."""
    with pytest.raises(ValueError):
        fetch('non-existent.file')


def test_fetch_multi():
    """Test fetch multiple files."""
    filenames = fetch('simfcs.r64', 'simfcs.ref')
    for filename in filenames:
        assert os.path.exists(filename)


def test_fetch_list():
    """Test fetch multiple files."""
    filenames = fetch(['simfcs.r64', 'simfcs.ref'])
    assert len(filenames) == 2


@pytest.mark.skipif(SKIP_LARGE, reason='large download')
def test_fetch_repo():
    """Test fetch repo."""
    filenames = fetch('tests')
    for filename in filenames:
        assert os.path.exists(filename)


@pytest.mark.skipif(SKIP_LARGE, reason='large download')
def test_fetch_all():
    """Test fetch all files."""
    filenames = fetch()
    for filename in filenames:
        assert os.path.exists(filename)


# mypy: allow-untyped-defs, allow-untyped-calls
# mypy: disable-error-code="arg-type"
