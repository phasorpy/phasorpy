"""Tests for the phasorpy.datasets module."""

import os

import pytest

from phasorpy.datasets import fetch


def test_fetch():
    """Test fetch file."""
    filename = fetch('simfcs.r64')
    assert os.path.exists(filename)


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
