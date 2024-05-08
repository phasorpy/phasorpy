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
min_vals1 = numpy.array([-2.20, -1.13])
max_vals1 = numpy.array([-1.75, -0.7])
min_vals2 = numpy.array([0.70, 0.85])
max_vals2 = numpy.array([0.90, 1.0])
lut = create_lut(min_vals1, max_vals1, min_vals2, max_vals2)
label = label_from_lut(phase, mod, lut)
# %%
# Show the label image:

fig, ax = pyplot.subplots()
ax.set_title('Labels from LUT of phase/modulation values')
plt = ax.imshow(label, vmin=0, vmax=10, cmap='tab10')
fig.colorbar(plt)
pyplot.show()

# %%
# Show cursors in the phasor plot:

plot = PhasorPlot(allquadrants=True, title='Phase/Modulation cursors')
plot.hist2d(real, imag, cmap='Blues')
plot.polar_cursor(
    phase=min_vals1[0],
    phase_limit=max_vals1[0],
    modulation=min_vals2[0],
    modulation_limit=max_vals2[0],
    color='tab:orange',
    linestyle='-',
)
plot.polar_cursor(
    phase=min_vals1[1],
    phase_limit=max_vals1[1],
    modulation=min_vals2[1],
    modulation_limit=max_vals2[1],
    color='tab:purple',
    linestyle='-',
)
plot.show()
# %%

# %%
# Cursors with LUT and overlapping
# --------------------------------
#
# Create labels from LUT of phase and modulation values computed from phasor coordinates:

mean, real, imag = phasor_from_signal(signal, axis=0)
phase, mod = phasor_to_polar(real, imag)
min_vals1 = numpy.array([-2.20, -2.0, -1.13, -0.95])
max_vals1 = numpy.array([-1.9, -1.7, -0.90, -0.70])
min_vals2 = numpy.array([0.70, 0.70, 0.85, 0.85])
max_vals2 = numpy.array([0.90, 0.90, 1.0, 1.0])
lut = create_lut(min_vals1, max_vals1, min_vals2, max_vals2)
label = label_from_lut(phase, mod, lut)
# %%
# Show the label image:

fig, ax = pyplot.subplots()
ax.set_title('Labels from LUT of phase/modulation values')
plt = ax.imshow(label, vmin=0, vmax=10, cmap='tab10')
fig.colorbar(plt)
pyplot.show()

# %%
# Show cursors in the phasor plot:

plot = PhasorPlot(allquadrants=True, title='Phase/Modulation cursors')
plot.hist2d(real, imag, cmap='Blues')
plot.polar_cursor(
    phase=min_vals1[0],
    phase_limit=max_vals1[0],
    modulation=min_vals2[0],
    modulation_limit=max_vals2[0],
    color='tab:purple',
    linestyle='-',
)
plot.polar_cursor(
    phase=min_vals1[1],
    phase_limit=max_vals1[1],
    modulation=min_vals2[1],
    modulation_limit=max_vals2[1],
    color='tab:green',
    linestyle='-',
)

plot.polar_cursor(
    phase=min_vals1[2],
    phase_limit=max_vals1[2],
    modulation=min_vals2[2],
    modulation_limit=max_vals2[2],
    color='tab:gray',
    linestyle='-',
)
plot.polar_cursor(
    phase=min_vals1[3],
    phase_limit=max_vals1[3],
    modulation=min_vals2[3],
    modulation_limit=max_vals2[3],
    color='tab:cyan',
    linestyle='-',
)
plot.show()
# %%
