"""Test PhasorPlotFret class."""

from matplotlib import pyplot

from phasorpy.plot import PhasorPlotFret

INTERACTIVE = False  # enable for interactive plotting


def test_phasorplot_fret():
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


def test_phasorplot_fret_interactive():
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


# mypy: allow-untyped-defs, allow-untyped-calls
# mypy: disable-error-code="arg-type"
