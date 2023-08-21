"""
Introduction
============

This tutorial provides an introduction to using the PhasorPy library.

"""

# %%
# Install Python
# --------------
#
# An installation of Python version 3.10 or higher is required to use the
# PhasorPy library.
# Python is an easy to learn, powerful programming language.
# Python installers can be obtained from, for example,
# `Python.org <https://www.python.org/downloads/>`_ or
# `Anaconda.com <https://www.anaconda.com/>`_.
# Refer to the `Python Tutorial <https://docs.python.org/3/tutorial/>`_
# for an introduction to Python.
#
# Install PhasorPy
# ----------------
#
# On a command prompt, shell or terminal, run ``pip`` to download and install
# the PhasorPy library and all its dependencies from the Python Package Index::
#
# $ python -m pip install -U phasorpy

# %%
# Import phasorpy
# ---------------
#
# Open the Python interpreter, import the ``phasorpy`` package, and check its
# version:

import phasorpy

print(phasorpy.__version__)

# %%
# Read data file
# --------------
#
# PhasorPy provides many functions to read image and metadata from file formats
# used in microscopy.
# For example, ...:

...

# %%
# Calculate phasor coordinates
# ----------------------------
#
# Calculate the phasor coordinates for the image stack using known reference
# coordinates:

...

# %%
# Display phasor plot
# -------------------

from matplotlib import pyplot

pyplot.imshow([[0, 1], [2, 3]])

# %%
# Segment intensity image
# -----------------------
# Show the intensity image, colored according to a selection of phasor
# coordinates:

...
