"""
Phasor Cursors
==============

An introduction to selecting phasor coordinates using cursors.

"""

# %%
# Import required modules, functions, and classes:

import math

import tifffile
from matplotlib import pyplot

from phasorpy.cursors import label_from_phasor_circular, label_from_ranges
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
# Range cursors
# -------------
#
# Create labels from ranges of phase values computed from phasor coordinates:

mean, real, imag = phasor_from_signal(signal, axis=0)
phase, _ = phasor_to_polar(real, imag)
ranges = [(0, math.pi), (-math.pi / 2, 0), (-math.pi, -math.pi / 2)]
label = label_from_ranges(phase, ranges=ranges)

# %%
# Show the label image:

fig, ax = pyplot.subplots()
ax.set_title('Labels from phase ranges')
plt = ax.imshow(label, vmin=0, vmax=10, cmap='tab10')
fig.colorbar(plt)
pyplot.show()

# %%
# Show the range cursors in the phasor plot:

plot = PhasorPlot(allquadrants=True, title='Phase range cursors')
plot.hist2d(real, imag, cmap='Blues')
plot.polar_cursor(
    phase=ranges[0][0],
    phase_limit=ranges[0][1],
    color='tab:orange',
    linestyle='-',
)
plot.polar_cursor(
    phase=ranges[1][0],
    phase_limit=ranges[1][1],
    color='tab:green',
    linestyle='-',
)
plot.polar_cursor(
    phase=ranges[2][0],
    phase_limit=ranges[2][1],
    color='tab:red',
    linestyle='-',
)
plot.show()
