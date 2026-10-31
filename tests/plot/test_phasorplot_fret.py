# Copyright (c) PhasorPy Contributors
# SPDX-License-Identifier: MIT
# See LICENSE.txt file in the project root for details.

"""Test PhasorPlotFret class."""

import numpy
from matplotlib import pyplot
from numpy.testing import assert_allclose

from phasorpy.lifetime import phasor_from_lifetime
from phasorpy.plot import PhasorPlotFret

INTERACTIVE = False  # enable for interactive plotting


def test_phasorplot_fret_positions() -> None:
    """Test initial positions of FRET donor markers."""
    # 0.29 * 100 truncates to 28
    plot = PhasorPlotFret(
        frequency=60.0, donor_lifetime=4.2, fret_efficiency=0.29
    )
    expected = numpy.asarray(
        phasor_from_lifetime(60.0, 4.2 * (1.0 - 0.29))
    ).reshape(-1)
    donor_fret = numpy.asarray(plot._donor_fret_line.get_xydata())
    donor = numpy.asarray(plot._donor_line.get_xydata())
    assert_allclose(donor_fret[0], expected)
    assert_allclose(donor[0], expected)
    pyplot.close()


def test_phasorplot_fret() -> None:
    """Test PhasorPlotFret."""
    plot = PhasorPlotFret(
        frequency=60.0,
        donor_lifetime=4.2,
        acceptor_lifetime=3.0,
        fret_efficiency=0.5,
        donor_fretting=0.9,
        donor_bleedthrough=0.1,
        title='PhasorPlotFret',
    )
    if INTERACTIVE:
        plot.show()
    pyplot.close()


def test_phasorplot_fret_interactive() -> None:
    """Test PhasorPlotFret interactive."""
    plot = PhasorPlotFret(
        frequency=60.0,
        donor_lifetime=4.2,
        acceptor_lifetime=3.0,
        fret_efficiency=0.5,
        donor_background=0.1,
        acceptor_background=0.1,
        interactive=True,
        title='PhasorPlotFret interactive',
    )
    plot._frequency_slider.set_val(80.0)
    plot._donor_fretting_slider.set_val(0.9)
    plot._donor_bleedthrough_slider.set_val(0.1)
    plot._donor_bleedthrough_slider.set_val(0.0)
    plot._donor_background_slider.set_val(0.1)
    plot._acceptor_background_slider.set_val(0.1)
    plot._donor_background_slider.set_val(0.0)
    plot._acceptor_background_slider.set_val(0.0)
    plot._donor_fretting_slider.set_val(0.0)
    if INTERACTIVE:
        plot.show()
    pyplot.close()
