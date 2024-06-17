"""
Phasor Cursors
==============

An introduction to selecting phasor coordinates using cursors.

"""

# %%
# Import required modules, functions, and classes:

import math

import numpy
import tifffile
from matplotlib import pyplot

from phasorpy.cursors import *
from phasorpy.cursors import join_masks
from phasorpy.datasets import fetch
from phasorpy.phasor import phasor_from_signal, phasor_to_polar
from phasorpy.plot import PhasorPlot

# %%
# Circular cursors
# ----------------
#
# Use circular cursors in the phasor space to segment the images:

signal = tifffile.imread(fetch('paramecium.lsm'))
mean, real, imag = phasor_from_signal(signal, axis=0)

center = [(-0.48, -0.65), (-0.22, -0.75), (0.4, -0.8), (0.66, -0.68)]
radius = [0.15, 0.15, 0.15, 0.15]

label = label_from_phasor_circular(real, imag, center, radius)


# %%
# Show the circular cursors in the phasor plot:

mask = mean > 1

plot = PhasorPlot(allquadrants=True, title='Circular cursors')
plot.hist2d(real[mask], imag[mask])
plot.cursor(*center[0], radius=radius[0], color='tab:orange', linestyle='-')
plot.cursor(*center[1], radius=radius[1], color='tab:green', linestyle='-')
plot.cursor(*center[2], radius=radius[2], color='tab:red', linestyle='-')
plot.cursor(*center[3], radius=radius[3], color='tab:purple', linestyle='-')

# %%
# Show the label image:

fig, ax = pyplot.subplots()
ax.set_title('Labels from circular cursors')
plt = ax.imshow(label, vmin=0, vmax=10, cmap='tab10')
fig.colorbar(plt)
pyplot.show()


# %%
# Cursors with LUT without overlapping
# ------------------------------------
#
# Create labels from LUT of phase and modulation values computed from phasor coordinates:

mean, real, imag = phasor_from_signal(signal, axis=0)
phase, mod = phasor_to_polar(real, imag)


xrange1 = [-2.27, -1.57]
yrange1 = [0.7, 0.9]
mask1 = mask_from_cursor(phase, mod, xrange1, yrange1)
xrange2 = [-1.22, -0.7]
yrange2 = [0.8, 1]
# mask2 = mask_from_cursor(phase, mod, xrange2, yrange2)
# label = join_masks([mask1, mask2])
# label1 = numpy.where(label, 1, 0)
# compose masks


# %%
# Show the label image:

fig, ax = pyplot.subplots()
ax.set_title('Mask from cursor')
plt = ax.imshow(mask1, vmin=0, vmax=10, cmap='tab10')
fig.colorbar(plt)
pyplot.show()

# %%
# Show cursors in the phasor plot:

plot = PhasorPlot(allquadrants=True, title='Phase/Modulation cursors')
plot.hist2d(real, imag, cmap='Blues')
plot.polar_cursor(
    phase=xrange1[0],
    phase_limit=xrange1[1],
    modulation=yrange1[0],
    modulation_limit=yrange1[1],
    color='tab:orange',
    linestyle='-',
)
plot.polar_cursor(
    phase=xrange2[0],
    phase_limit=xrange2[1],
    modulation=yrange2[0],
    modulation_limit=yrange2[1],
    color='tab:purple',
    linestyle='-',
)
plot.show()
