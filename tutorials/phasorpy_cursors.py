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

mask_c1 = mask_from_circular_cursor(real, imag, (-0.33, -0.72), 0.2)
mask_c2 = mask_from_circular_cursor(real, imag, (0.55, -0.72), 0.2)

# %%
# Show the circular cursors in the phasor plot:

threshold = mean > 1

plot = PhasorPlot(allquadrants=True, title='Circular cursors')
plot.hist2d(real[threshold], imag[threshold])
plot.cursor(-0.33, -0.72, radius=0.2, color='tab:blue', linestyle='-')
plot.cursor(0.55, -0.72, radius=0.2, color='tab:red', linestyle='-')

# %%
# Show segmented image with circular cursors:
mask = join_arrays([mask_c1, mask_c2])
cursors_colors = [[0, 0, 255], [255, 0, 0]]
segmented = segmentate_with_cursors(mask, cursors_colors, mean)

fig, ax = pyplot.subplots()
ax.set_title('Segmented image with circular cursors')
plt = ax.imshow(segmented)
# %%
# Phase and modulation cursor
# ---------------------------
#
# Create a mask with phase and modulation values:

mean, real, imag = phasor_from_signal(signal, axis=0)
phase, mod = phasor_to_polar(real, imag)
threshold = mean > 1

xrange1 = [-2.27, -1.57]
yrange1 = [0.7, 0.9]
maskc1 = mask_from_cursor(phase, mod, xrange1, yrange1)

xrange2 = [-1.22, -0.70]
yrange2 = [0.8, 1.0]
maskc2 = mask_from_cursor(phase, mod, xrange2, yrange2)

# %%
# Show cursors in the phasor plot:

plot = PhasorPlot(allquadrants=True, title='Phase/Modulation cursors')
plot.hist2d(real[threshold], imag[threshold], cmap='Blues')
plot.polar_cursor(
    phase=xrange1[0],
    phase_limit=xrange1[1],
    modulation=yrange1[0],
    modulation_limit=yrange1[1],
    color='tab:blue',
    linestyle='-',
)
plot.polar_cursor(
    phase=xrange2[0],
    phase_limit=xrange2[1],
    modulation=yrange2[0],
    modulation_limit=yrange2[1],
    color='tab:red',
    linestyle='-',
)

# %%
# Segmented intensity image with cursors
mask = join_arrays([maskc1, maskc2])
cursors_colors = [[0, 0, 255], [255, 0, 0]]
segmented = segmentate_with_cursors(mask, cursors_colors, mean)

fig, ax = pyplot.subplots()
ax.set_title('Segmented image with cursors')
plt = ax.imshow(segmented)
plot.show()
