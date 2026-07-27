# Copyright (c) PhasorPy Contributors
# SPDX-License-Identifier: MIT
# See LICENSE.txt file in the project root for details.

"""Test the phasorpy.datasets module."""

import os

import pytest

from phasorpy.datasets import DATA_ON_GITHUB, fetch

if os.environ.get('SKIP_FETCH', ''):
    pytest.skip('fetch is disabled', allow_module_level=True)

# skip large downloads by default
SKIP_LARGE = bool(os.environ.get('SKIP_LARGE', '1'))


@pytest.mark.skipif(not DATA_ON_GITHUB, reason='not using GitHub Actions')
def test_data_on_github() -> None:
    """Test data files on GitHub."""
    assert DATA_ON_GITHUB


def test_fetch() -> None:
    """Test fetch file."""
    name = 'simfcs.r64'
    filename = fetch(name)
    assert filename.endswith(name)
    assert os.path.exists(filename)
    filename = fetch(name, return_scalar=False)
    assert filename[0].endswith(name)


def test_fetch_inzip() -> None:
    """Test fetch file from ZIP."""
    filename = fetch('simfcs.b&h')
    assert os.path.exists(filename)


def test_fetch_zip() -> None:
    """Test fetch ZIP file."""
    filename = fetch('flimage.int.bin.zip', extract_dir='unzipped')[0]
    assert os.path.exists(filename)
    assert 'unzipped' in filename


def test_fetch_compound() -> None:
    """Test fetch compound file in ZIP."""
    filename = fetch('simfcs_1000.int')
    assert os.path.exists(filename.replace('int', 'mod'))


def test_fetch_nonexistent() -> None:
    """Test fetch non-existent file."""
    with pytest.raises(ValueError):
        fetch('non-existent.file')


def test_fetch_multi() -> None:
    """Test fetch multiple files."""
    filenames = fetch('simfcs.r64', 'simfcs.ref')
    for filename in filenames:
        assert os.path.exists(filename)


def test_fetch_list() -> None:
    """Test fetch multiple files."""
    filenames = fetch(['simfcs.r64', 'simfcs.ref'])
    assert len(filenames) == 2


@pytest.mark.skipif(SKIP_LARGE, reason='large download')
def test_fetch_repo() -> None:
    """Test fetch repo."""
    filenames = fetch('tests')
    for filename in filenames:
        assert os.path.exists(filename)


@pytest.mark.skipif(SKIP_LARGE, reason='large download')
def test_fetch_all() -> None:
    """Test fetch all files."""
    filenames = fetch()
    for filename in filenames:
        assert os.path.exists(filename)
