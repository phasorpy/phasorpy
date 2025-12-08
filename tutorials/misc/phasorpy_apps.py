"""
Interactive apps
================

Interactive educational applications.

The PhasorPy library includes interactive applications for educational
purposes. They open graphical user interfaces where parameters can be
adjusted via sliders and the resulting phasor and lifetime plots are
updated in real time. The apps can be started from the :doc:`/api/cli` or
programmatically. A matplotlib backend that supports interactive
windows is required.

"""

# %%
# Lifetime app
# ------------
#
# Interactively plot time-domain signals, phasor coordinates, and
# multi-frequency phase and modulation curves for a set of lifetime
# components and their mixture at given frequency and fractional intensities.
#
# Start the lifetime app from the command line::
#
#     $ python -m phasorpy lifetime
#
# or run the following code in a Python interpreter:

from phasorpy.plot import LifetimePlots

LifetimePlots(
    frequency=80.0,  # MHz
    lifetime=(0.4, 3.4, 8),  # ns
    fraction=(0.3, 0.5, 0.2),  # fractional intensities
    interactive=True,
).show()

# %%
# Change frequency, lifetimes, and fractional intensities to see how the
# phasor coordinates, phase and modulation, and decay curves respond.

# %%
# FRET app
# --------
#
# Interactively plot phasor coordinates for Förster resonance energy transfer
# (FRET) donor and acceptor channels as a function of frequency, donor and
# acceptor lifetimes, FRET efficiency, fractions of donors undergoing FRET
# (donor fretting), donor and acceptor bleedthrough, and background signal.
#
# Start the FRET app from the command line::
#
#     $ python -m phasorpy fret
#
# or run the following code in a Python interpreter:

from phasorpy.plot import PhasorPlotFret

PhasorPlotFret(
    frequency=60.0,  # MHz
    donor_lifetime=4.2,  # ns
    acceptor_lifetime=3.0,  # ns
    fret_efficiency=0.5,  # 50%
    donor_fretting=0.9,  # 90% of donors can undergo FRET
    donor_bleedthrough=0.1,  # 10% donor signal in acceptor channel
    acceptor_bleedthrough=0.1,  # 10% acceptor signal in donor channel
    interactive=True,
).show()

# %%
# Change frequency, FRET efficiency, lifetimes, and bleedthrough fractions
# to see how the donor and acceptor phasor positions respond.

# sphinx_gallery_start_ignore
# sphinx_gallery_thumbnail_number = 1
# sphinx_gallery_end_ignore
