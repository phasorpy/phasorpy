"""Read and write time-resolved and hyperspectral image file formats.

The ``phasorpy.io`` module provides functions to:

- read time-resolved and hyperspectral signals, as well as metadata from
  many file formats used in bio-imaging:

  - :py:func:`signal_from_lif` - Leica LIF and XLEF
  - :py:func:`signal_from_lsm` - Zeiss LSM
  - :py:func:`signal_from_ptu` - PicoQuant PTU
  - :py:func:`signal_from_sdt` - Becker & Hickl SDT
  - :py:func:`signal_from_fbd` - FLIMbox FBD
  - :py:func:`signal_from_flimlabs_json` - FLIM LABS JSON
  - :py:func:`signal_from_imspector_tiff` - ImSpector FLIM TIFF
  - :py:func:`signal_from_flif` - FlimFast FLIF
  - :py:func:`signal_from_b64` - SimFCS B64
  - :py:func:`signal_from_z64` - SimFCS Z64
  - :py:func:`signal_from_bhz` - SimFCS BHZ
  - :py:func:`signal_from_bh` - SimFCS B&H

- read phasor coordinates, lifetime images, and metadata from
  specialized file formats:

  - :py:func:`phasor_from_ometiff` - PhasorPy OME-TIFF
  - :py:func:`phasor_from_ifli` - ISS IFLI
  - :py:func:`phasor_from_lif` - Leica LIF and XLEF
  - :py:func:`phasor_from_flimlabs_json` - FLIM LABS JSON
  - :py:func:`phasor_from_simfcs_referenced` - SimFCS REF and R64
  - :py:func:`lifetime_from_lif` - Leica LIF and XLEF

- write phasor coordinate images to OME-TIFF and SimFCS file formats:

  - :py:func:`phasor_to_ometiff`
  - :py:func:`phasor_to_simfcs_referenced`

  Support for other file formats is being considered:

  - OME-TIFF
  - Zeiss CZI
  - Nikon ND2
  - Olympus OIB/OIF
  - Olympus OIR

The functions are implemented as minimal wrappers around specialized
third-party file reader libraries, currently
`tifffile <https://github.com/cgohlke/tifffile>`_,
`ptufile <https://github.com/cgohlke/ptufile>`_,
`liffile <https://github.com/cgohlke/liffile>`_,
`sdtfile <https://github.com/cgohlke/sdtfile>`_, and
`lfdfiles <https://github.com/cgohlke/lfdfiles>`_.
For advanced or unsupported use cases, consider using these libraries directly.

The signal-reading functions typically have the following signature::

    signal_from_ext(
        filename: str | PathLike,
        /,
        **kwargs
    ): -> xarray.DataArray

where ``ext`` indicates the file format and ``kwargs`` are optional arguments
passed to the underlying file reader library or used to select which data is
returned. The returned `xarray.DataArray
<https://docs.xarray.dev/en/stable/user-guide/data-structures.html>`_
contains an N-dimensional array with labeled coordinates, dimensions, and
attributes:

- ``data`` or ``values`` (*array_like*)

  Numpy array or array-like holding the array's values.

- ``dims`` (*tuple of str*)

  :ref:`Axes character codes <axes>` for each dimension in ``data``.
  For example, ``('T', 'C', 'Y', 'X')`` defines the dimension order in a
  4-dimensional array of a time-series of multi-channel images.

- ``coords`` (*dict_like[str, array_like]*)

  Coordinate arrays labelling each point in the data array.
  The keys are :ref:`axes character codes <axes>`.
  Values are 1-dimensional arrays of numbers or strings.
  For example, ``coords['C']`` could be an array of emission wavelengths.

- ``attrs`` (*dict[str, Any]*)

  Arbitrary metadata such as measurement or calibration parameters required to
  interpret the data values.
  For example, the laser repetition frequency of a time-resolved measurement.

.. _axes:

Axes character codes from the OME model and tifffile library are used as
``dims`` items and ``coords`` keys:

- ``'X'`` : width (OME)
- ``'Y'`` : height (OME)
- ``'Z'`` : depth (OME)
- ``'S'`` : sample (color components or phasor coordinates)
- ``'I'`` : sequence (of images, frames, or planes)
- ``'T'`` : time (OME)
- ``'C'`` : channel (OME. Acquisition path or emission wavelength)
- ``'A'`` : angle (OME)
- ``'P'`` : phase (OME. In LSM, ``'P'`` maps to position)
- ``'R'`` : tile (OME. Region, position, or mosaic)
- ``'H'`` : lifetime histogram (OME)
- ``'E'`` : lambda (OME. Excitation wavelength)
- ``'F'`` : frequency (ISS)
- ``'Q'`` : other (OME. Harmonics in PhasorPy TIFF)
- ``'L'`` : exposure (FluoView)
- ``'V'`` : event (FluoView)
- ``'M'`` : mosaic (LSM 6)
- ``'J'`` : column (NDTiff)
- ``'K'`` : row (NDTiff)

"""

from __future__ import annotations

__all__: list[str] = []

from .._utils import init_module
from ._flimlabs import *
from ._leica import *
from ._ometiff import *
from ._other import *
from ._simfcs import *

# The `init_module()` function dynamically populates the `__all__` list with
# all public symbols imported from submodules or defined in this module.
# Any name not starting with an underscore will be automatically exported
# when using "from phasorpy.io import *"

init_module(globals())
del init_module

# flake8: noqa: F401, F403
