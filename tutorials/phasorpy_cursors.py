"""
Phasor Cursors selectors
========================

An introduction to cursors module.

"""

# %%
# Import required modules, functions, and classes:

import math

import numpy
import tifffile

from phasorpy.cursors import label_from_phasor_circular, label_from_ranges
from phasorpy.datasets import fetch
from phasorpy.phasor import phasor_from_signal, phasor_to_polar
from phasorpy.plot import PhasorPlot

# %%
# Using circular cursors
# ----------------------
#
# Use circular cursors to select regions of interest.

signal = tifffile.imread(fetch('paramecium.lsm'))
mean, real, imag = phasor_from_signal(signal, axis=0)

label = label_from_phasor_circular(
    real,
    imag,
    numpy.array(
        [[-0.48, -0.65], [-0.22, -0.75], [0.40, -0.80], [0.66, -0.68]]
    ),
    radius=[0.15, 0.15, 0.15, 0.15],
)


# %%
# Circular cursors
# ----------------
#

mask = mean > 1
real = real[mask]
imag = imag[mask]

import matplotlib.pyplot as plt


def cart2pol(x, y):
    rho = numpy.sqrt(x**2 + y**2)
    phi = numpy.arctan2(y, x)
    return (phi, rho)


# components centers in polars
t1, r1 = cart2pol(-0.48, -0.65)
t2, r2 = cart2pol(-0.22, -0.75)
t3, r3 = cart2pol(0.40, -0.80)
t4, r4 = cart2pol(0.66, -0.68)

plotcursors = True
if plotcursors:
    plot = PhasorPlot(allquadrants=True, title='Test cursors selection')
    plot.polar_cursor(t1, r1, radius=0.15)
    plot.polar_cursor(t2, r2, radius=0.15)
    plot.polar_cursor(t3, r3, radius=0.15)
    plot.polar_cursor(t4, r4, radius=0.15)
    plot.hist2d(real, imag, cmap='Blues')

plt.figure()
plt.imshow(label)
plt.show()


# %%
# Range cursors
# -------------
#
# Compute phasor from signal and get the phase values.

mean, real, imag = phasor_from_signal(signal, axis=0)
phase, _ = phasor_to_polar(real, imag)
rang = numpy.array(
    [(0, numpy.pi), (-numpy.pi / 2, 0), (-numpy.pi, -numpy.pi / 2)]
)
label = label_from_ranges(phase, ranges=rang)


plt.figure()
plt.imshow(label)

plot = PhasorPlot(allquadrants=True, title='Raw phasor')
plot.hist2d(real, imag, cmap='Blues')

plt.show()
