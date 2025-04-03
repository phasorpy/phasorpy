"""Calculate and analyze phasor coordinates and related data.

The ``phasorpy`` module provides functions to:

- **calculate** phasor coordinates from time-resolved and spectral signals:

  - :py:func:`phasor_from_signal`

- **synthesize** signals from phasor coordinates or lifetimes:

  - :py:func:`phasor_to_signal`
  - :py:func:`lifetime_to_signal`

- **convert** between phasor coordinates and single- or multi-component
  fluorescence **lifetimes**:

  - :py:func:`phasor_from_lifetime`
  - :py:func:`phasor_from_apparent_lifetime`
  - :py:func:`phasor_to_apparent_lifetime`

- **convert** to and from **polar coordinates** (phase and modulation):

  - :py:func:`phasor_from_polar`
  - :py:func:`phasor_to_polar`
  - :py:func:`polar_from_apparent_lifetime`
  - :py:func:`polar_to_apparent_lifetime`

- **transform** phasor coordinates:

  - :py:func:`phasor_transform`
  - :py:func:`phasor_multiply`
  - :py:func:`phasor_divide`
  - :py:func:`phasor_normalize`

- **calibrate** phasor coordinates with a reference of known lifetimes:

  - :py:func:`phasor_calibrate`
  - :py:func:`polar_from_reference`
  - :py:func:`polar_from_reference_phasor`

- **reduce** dimensionality of arrays of phasor coordinates:

  - :py:func:`phasor_center`
  - :py:func:`phasor_to_principal_plane`

- **calculate** phasor coordinates for **FRET donor and acceptor** channels:

  - :py:func:`phasor_from_fret_donor`
  - :py:func:`phasor_from_fret_acceptor`

- **convert** between single component **lifetimes** and **optimal frequency**:

  - :py:func:`lifetime_to_frequency`
  - :py:func:`lifetime_from_frequency`

- **convert** between **fractional intensities** and
  **pre-exponential amplitudes**:

  - :py:func:`lifetime_fraction_from_amplitude`
  - :py:func:`lifetime_fraction_to_amplitude`

- **calculate** phasor coordinates on **universal semicircle**:

  - :py:func:`phasor_semicircle`
  - :py:func:`phasor_at_harmonic`

- **filter** phasor coordinates:

  - :py:func:`phasor_filter_median`
  - :py:func:`phasor_filter_pawflim`
  - :py:func:`phasor_threshold`

- **cluster** phasor coordinates using machine learning:

  - :py:func:`phasor_cluster_gmm`

- **analyze components** of phasor coordinates using geometric and
  linear algebra approaches:

  - :py:func:`phasor_component_fraction`
  - :py:func:`phasor_component_graphical_analysis`

- **mask** regions of interest in phasor space:

  - :py:func:`phasor_mask_circular`
  - :py:func:`phasor_mask_elliptic`
  - :py:func:`phasor_mask_polar`

"""

from __future__ import annotations

__all__ = ['__version__']

from ._cluster import *
from ._component import *
from ._cursor import *
from ._filter import *
from ._lifetime import *
from ._phasor import *
from ._utils import init_module

__version__ = '0.5.dev'
"""PhasorPy version string."""

init_module(globals())
del init_module

# flake8: noqa: F401, F403
