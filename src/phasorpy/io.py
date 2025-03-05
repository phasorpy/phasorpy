# phasorpy.io proxy module
# The VSCode debugger cannot to step into or break inside modules named io.py

from ._io import *
from ._io import __all__, __doc__
from ._utils import set_module

set_module(globals())
del set_module
