"""Plot phasor coordinates and related data.

The ``phasorpy.plot`` module provides functions and classes to visualize
phasor coordinates and related data using the
`matplotlib <https://matplotlib.org/>`_ library.

"""

from __future__ import annotations

__all__: list[str] = []

from .._utils import init_module
from ._functions import *
from ._lifetime_plots import *
from ._phasorplot import *
from ._phasorplot_fret import *

# The `init_module()` function dynamically populates the `__all__` list with
# all public symbols imported from submodules or defined in this module.
# Any name not starting with an underscore will be automatically exported
# when using "from phasorpy.plot import *"

init_module(globals())
del init_module

# flake8: noqa: F401, F403
