"""Tests for the phasorpy.phasor module."""

import os

import numpy
import pytest

from phasorpy.datasets import fetch
from phasorpy.io import read_z64
from phasorpy.phasor import phasor

SKIP_FETCH = os.environ.get('SKIP_FETCH', False)

def test_random_phasor():
    """Test calculating phasor transform"""
    data = numpy.random.randint(0, 255, size=(30, 512, 512))
    dc, g, s = phasor(data)
    assert dc.shape == (data.shape[1], data.shape[2])
    assert g.shape == (data.shape[1], data.shape[2])
    assert s.shape == (data.shape[1], data.shape[2])
    assert numpy.all(dc != 0)
    assert numpy.all((g >= -1) & (g <= 1))
    assert numpy.all((s >= -1) & (s <= 1))


def test_ones_phasor():
    """Test calculating phasor transform"""
    data = numpy.ones((30, 512, 512))
    dc, g, s = phasor(data)
    assert dc.shape == (data.shape[1], data.shape[2])
    assert g.shape == (data.shape[1], data.shape[2])
    assert s.shape == (data.shape[1], data.shape[2])
    assert numpy.all(dc != 0)
    assert numpy.all(g == 0)
    assert numpy.all(s == 0)


@pytest.mark.skipif(SKIP_FETCH, reason='fetch is disabled')
def test_phasor_img():
    """Test phasor calculation on example Z64 file"""
    filename = fetch('simfcs.z64')
    data = read_z64(filename)
    dc, g, s = phasor(data)
    assert dc.shape == (data.shape[1], data.shape[2])
    assert g.shape == (data.shape[1], data.shape[2])
    assert s.shape == (data.shape[1], data.shape[2])
    assert numpy.all(dc != 0)


def test_empty_phasor():
    """Test empty image phasor calculation error"""
    empty_data = numpy.zeros((30, 512, 512))
    with pytest.raises(ValueError):
        phasor(empty_data)

