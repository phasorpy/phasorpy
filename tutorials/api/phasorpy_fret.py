"""
Förster Resonance Energy Transfer
=================================

Calculate and plot phasor coordinates of FRET donor and acceptor channels.

The :py:func:`phasorpy.phasor.phasor_from_fret_donor`,
:py:func:`phasorpy.phasor.phasor_from_fret_acceptor`, and
:py:class:`phasorpy.plot.PhasorPlotFret` functions and classes
are used to calculate and plot phasor coordinates of
FRET (Förster Resonance Energy Transfer) donor and acceptor channels
as a function of:

- laser pulse or modulation frequency
- donor and acceptor lifetimes
- FRET efficiency
- fraction of donors undergoing FRET
- fraction of directly excited acceptors (acceptor bleedthrough)
- fraction of donor fluorescence in acceptor channel (donor bleedthrough)
- fraction of background fluorescence

"""

# %%
# Define FRET model settings used throughout this tutorial:

settings = {
    'frequency': 60.0,  # MHz
    'donor_lifetime': 4.2,  # ns
    'acceptor_lifetime': 3.0,  # ns
    'fret_efficiency': 0.5,  # 50%
}


# %%
# FRET efficiency trajectories
# ----------------------------
#
# The lifetime :math:`\tau_{DA}` and fluorescence intensity :math:`F_{DA}`
# of a FRET donor quenched by energy transfer of efficiency :math:`E` is given
# by :math:`\tau_{DA} = \tau_{D} (1 - E)` and :math:`F_{DA} = F_{D} (1 - E)`,
# where :math:`\tau_{D}` and :math:`F_{D}` are the donor lifetime and
# intensity in the absence of energy transfer.
#
# Hence, in the absence of background fluorescence and donors not undergoing
# energy transfer, the phasor coordinates of the donor channel at different
# FRET efficiencies lie on the universal semicircle.
# At 100% energy transfer, the donor lifetime and fluorescence intensity are
# zero.
#
# The sensitized emission of a FRET acceptor is phase-shifted and demodulated
# relative to the FRET donor because of the duration of, and dissipation
# during, relaxation and energy transfer processes.
# Hence, in the absence of directly excited acceptor, donor bleedthrough,
# and background fluorescence, the phasor coordinates of the acceptor channel
# at different FRET efficiencies lie outside the universal semicircle of
# the donor.

from phasorpy.plot import PhasorPlotFret

PhasorPlotFret(
    **settings,
    title='FRET efficiency trajectories',
).show()


# %%
# Fractions not fretting
# ----------------------
#
# Adding fractions of donors not participating in FRET and fractions
# of directly excited acceptors (acceptor bleedthrough) pulls the
# FRET trajectories of the donor and acceptor channels towards the
# phasor coordinates of the donor and acceptor without FRET:

PhasorPlotFret(
    **settings,
    donor_fretting=0.9,  # 90%
    acceptor_bleedthrough=0.1,  # 10%
    title='FRET efficiency trajectories with fractions not fretting',
).show()


# %%
# Donor bleedthrough
# ------------------
#
# When the acceptor channel contains fractions of donor fluorescence
# (donor bleedthrough), the FRET efficiency trajectory of the acceptor
# channel shifts towards the phasor coordinates of the donor channel:

PhasorPlotFret(
    **settings,
    donor_bleedthrough=0.1,  # 10%
    title='FRET efficiency trajectories with donor bleedthrough',
).show()


# %%
# Background fluorescence
# -----------------------
#
# In the presence of background fluorescence, the FRET efficiency trajectories
# are linear combinations with the background phasor coordinates.
# At 100% energy transfer, the donor channel only contains background
# fluorescence if all donors participate in FRET.
# At 0% energy transfer, in the absence of donor bleedthrough and directly
# excited acceptor, the acceptor channel only contains background fluorescence:

PhasorPlotFret(
    **settings,
    donor_fretting=1.0,
    acceptor_background=0.1,  # 10%
    donor_background=0.1,  # 10%
    background_real=0.5,
    background_imag=0.2,
    title='FRET efficiency trajectories with background',
).show()


# %%
# Many parameters
# ---------------
#
# The phasor coordinates of the donor channel may contain fractions of:
#
# - donor not undergoing energy transfer
# - donor quenched by energy transfer
# - background fluorescence
#
# The phasor coordinates of the acceptor channel may contain fractions of:
#
# - acceptor sensitized by energy transfer
# - directly excited acceptor
# - donor bleedthrough
# - background fluorescence

PhasorPlotFret(
    **settings,
    donor_fretting=0.9,
    donor_bleedthrough=0.1,
    acceptor_bleedthrough=0.1,
    acceptor_background=0.1,
    donor_background=0.1,
    background_real=0.5,
    background_imag=0.2,
    title='FRET efficiency trajectories with many parameters',
).show()


# %%
# Interactive plot
# ----------------
#
# Run the FRET phasor plot interactively::
#
#     $ python -m phasorpy fret
#
# or

PhasorPlotFret(
    **settings,
    donor_fretting=0.9,
    donor_bleedthrough=0.1,
    interactive=True,
    title='Interactive FRET phasor plot',
).show()


# %%
# Multi-frequency plot
# --------------------
#
# Since each component of the FRET model has a distinct frequency response,
# the multi-frequency plots of donor and acceptor channels reveal
# complex patterns.
# Background fluorescence is omitted from this example to model an
# in vitro experiment:

import numpy

from phasorpy.phasor import (
    phasor_from_fret_acceptor,
    phasor_from_fret_donor,
    phasor_to_polar,
)
from phasorpy.plot import plot_polar_frequency

frequency = numpy.logspace(0, 4, 64).reshape(-1, 1)  # 1-10000 MHz
fret_efficiency = numpy.array([0.05, 0.95]).reshape(1, -1)  # 5% and 95%
donor_lifetime = 4.2
acceptor_lifetime = 3.0
donor_fretting = 0.9
donor_bleedthrough = 0.1
acceptor_bleedthrough = 0.1

donor_real, donor_imag = phasor_from_fret_donor(
    frequency,
    donor_lifetime,
    fret_efficiency=fret_efficiency,
    donor_fretting=donor_fretting,
)

# phasor of acceptor channel
acceptor_real, acceptor_imag = phasor_from_fret_acceptor(
    frequency,
    donor_lifetime,
    acceptor_lifetime,
    fret_efficiency=fret_efficiency,
    donor_fretting=donor_fretting,
    donor_bleedthrough=donor_bleedthrough,
    acceptor_bleedthrough=acceptor_bleedthrough,
)

plot_polar_frequency(
    frequency,
    phasor_to_polar(donor_real, donor_imag)[0],
    phasor_to_polar(donor_real, donor_imag)[1],
    title='Donor channel',
)

# %%

plot_polar_frequency(
    frequency,
    *phasor_to_polar(acceptor_real, acceptor_imag),
    title='Acceptor channel',
)

# %%
# sphinx_gallery_thumbnail_number = 5
# mypy: allow-untyped-defs, allow-untyped-calls
# mypy: disable-error-code="arg-type"
