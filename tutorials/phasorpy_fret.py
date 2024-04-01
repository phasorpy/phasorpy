"""
FRET calculations
=================

Calculate phasor coordinates of FRET donor and acceptor channels.

The :py:func:`phasorpy.phasor.phasor_from_fret_donor`
and :py:func:`phasorpy.phasor.phasor_from_fret_acceptor`
functions are used to calculate phasor coordinates of
FRET (Förster Resonance Energy Transfer) donor and acceptor channels
as a function of:

- laser pulse or modulation frequency
- donor and acceptor lifetimes
- FRET efficiency
- fraction of donors undergoing FRET
- fraction of directly excited acceptors
- fraction of donor fluorescence in acceptor channel
- fraction of background fluorescence

"""

# %%
# Define a helper function to compute and plot phasor coordinates of
# FRET donor and acceptor channels over a range of FRET efficiencies:

import numpy

from phasorpy.phasor import (
    phasor_from_fret_acceptor,
    phasor_from_fret_donor,
    phasor_from_lifetime,
)
from phasorpy.plot import PhasorPlot


def plot_fret_trajectories(
    frequency=60.0,  # MHz
    donor_lifetime=4.2,  # ns
    acceptor_lifetime=3.0,  # ns
    fret_efficiency=numpy.linspace(0.0, 1.0, 101),  # 0%..100%, 1% steps
    *,
    donor_freting=1.0,  # all donors participating in FRET
    donor_bleedthrough=0.0,  # no donor fluorescence in acceptor channel
    acceptor_excitation=0.0,  # no directly excited acceptor
    acceptor_background=0.0,  # no background in acceptor channel
    donor_background=0.0,  # no background in donor channel
    background_real=0.0,
    background_imag=0.0,
    title=None,
):
    """Plot phasor coordinates of FRET donor and acceptor channels."""
    # phasor of donor channel
    donor_fret_real, donor_fret_imag = phasor_from_fret_donor(
        frequency,
        donor_lifetime,
        fret_efficiency=fret_efficiency,
        donor_freting=donor_freting,
        donor_background=donor_background,
        background_real=background_real,
        background_imag=background_imag,
    )

    # phasor of acceptor channel
    acceptor_fret_real, acceptor_fret_imag = phasor_from_fret_acceptor(
        frequency,
        donor_lifetime,
        acceptor_lifetime,
        fret_efficiency=fret_efficiency,
        donor_freting=donor_freting,
        donor_bleedthrough=donor_bleedthrough,
        acceptor_excitation=acceptor_excitation,
        acceptor_background=acceptor_background,
        background_real=background_real,
        background_imag=background_imag,
    )

    # phasor of donor lifetime
    donor_real, donor_imag = phasor_from_lifetime(frequency, donor_lifetime)

    # phasor of acceptor lifetime
    acceptor_real, acceptor_imag = phasor_from_lifetime(
        frequency, acceptor_lifetime
    )

    plot = PhasorPlot(
        title=title,
        frequency=frequency,
        xlim=[-0.2, 1.1],
    )
    plot.semicircle(phasor_reference=(acceptor_real, acceptor_imag))
    if donor_background > 0.0:
        plot.line(
            [donor_real, background_real],
            [donor_imag, background_imag],
        )
    if acceptor_background > 0.0:
        plot.line(
            [acceptor_real, background_real],
            [acceptor_imag, background_imag],
        )
    plot.plot(
        donor_fret_real,
        donor_fret_imag,
        fmt='-',
        color='tab:green',
    )
    plot.plot(
        acceptor_fret_real,
        acceptor_fret_imag,
        fmt='-',
        color='tab:red',
    )
    plot.plot(
        donor_real,
        donor_imag,
        fmt='o',
        color='tab:green',
        label='Donor',
    )
    plot.plot(
        acceptor_real,
        acceptor_imag,
        fmt='o',
        color='tab:red',
        label='Acceptor',
    )
    if donor_background > 0.0 or acceptor_background > 0.0:
        plot.plot(
            background_real,
            background_imag,
            fmt='o',
            color='black',
            label='Background',
        )
    plot.show()


# %%
# FRET efficiency trajectories
# ----------------------------
#
# The lifetime :math:`\tau_{DA}` and fluorescence intensity :math:`F_{DA}`
# of a FRET donor quenched by energy transfer of efficiency :math:`E` is given
# by :math:`\tau_{DA} = \tau_{D} (1 - E)` and :math:`F_{DA} = F_{D} (1 - E)`,
# where :math:`\tau_{D}` and :math:`F_{D}` are the donor lifetime and
# intensity in absence of energy transfer.
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

plot_fret_trajectories(title='FRET efficiency trajectories')


# %%
# Fractions not FRETing
# ----------------------
#
# Adding fractions of donors not participating in FRET and fractions
# of directly excited acceptors pulls the FRET trajectories of the donor
# and acceptor channels towards the phasor coordinates of the donor and
# acceptor without FRET:

plot_fret_trajectories(
    title='FRET efficiency trajectories with fractions not FRETing',
    donor_freting=0.9,  # 90%
    acceptor_excitation=0.1,  # 10%
)


# %%
# Donor bleedthrough
# ------------------
#
# If the acceptor channel contains fractions of donor fluorescence,
# the FRET efficiency trajectory of the acceptor channel is pulled towards
# the phasor coordinates of the donor channel:

plot_fret_trajectories(
    title='FRET efficiency trajectories with donor bleedthrough',
    donor_bleedthrough=0.1,  # 10%
)


# %%
# Background fluorescence
# -----------------------
#
# In the presence of background fluorescence, the FRET efficiency trajectories
# are linear combinations with the background phasor coordinates.
# At 100% energy transfer, the donor channel only contains background
# fluorescence.
# At 0% energy transfer, in the absence of donor bleedthrough and directly
# excited acceptor, the acceptor channel only contains background fluorescence:

plot_fret_trajectories(
    title='FRET efficiency trajectories with background',
    acceptor_background=0.1,  # 10%
    donor_background=0.1,  # 10%
    background_real=0.5,
    background_imag=0.2,
)


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

plot_fret_trajectories(
    title='FRET efficiency trajectories with many parameters',
    donor_freting=0.9,
    donor_bleedthrough=0.1,
    acceptor_excitation=0.1,
    acceptor_background=0.1,
    donor_background=0.1,
    background_real=0.5,
    background_imag=0.2,
)

# %%
# Multi-frequency plot
# --------------------
#
# Since the components of the FRET model have different frequency responses,
# the multi-frequency plots of donor and acceptor channels may show
# complex patterns.
# Background fluorescence is omitted from this example to model an
# in vitro experiment:

from phasorpy.phasor import phasor_to_polar
from phasorpy.plot import plot_polar_frequency

frequency = numpy.logspace(0, 4, 64).reshape(-1, 1)  # 1-1000 MHz
fret_efficiency = numpy.array([0.05, 0.95]).reshape(1, -1)  # 5% and 95%
donor_lifetime = 4.2  # ns
acceptor_lifetime = 3.0  # ns
donor_freting = 0.9  # 90%
donor_bleedthrough = 0.1  # 10%
acceptor_excitation = 0.1  # 10%

donor_real, donor_imag = phasor_from_fret_donor(
    frequency,
    donor_lifetime,
    fret_efficiency=fret_efficiency,
    donor_freting=donor_freting,
)

# phasor of acceptor channel
acceptor_real, acceptor_imag = phasor_from_fret_acceptor(
    frequency,
    donor_lifetime,
    acceptor_lifetime,
    fret_efficiency=fret_efficiency,
    donor_freting=donor_freting,
    donor_bleedthrough=donor_bleedthrough,
    acceptor_excitation=acceptor_excitation,
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
