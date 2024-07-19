"""
Cursors
=======

An introduction to selecting phasor coordinates using cursors.

"""

# %%
# Import required modules, functions, and classes:

import matplotlib.pyplot as plt

from phasorpy.color import CATEGORICAL
from phasorpy.cursors import (
    mask_from_circular_cursor,
    mask_from_polar_cursor,
    pseudo_color,
)
from phasorpy.datasets import fetch
from phasorpy.io import read_lsm
from phasorpy.phasor import phasor_from_signal
from phasorpy.plot import PhasorPlot

# %%
# Open a hyperspectral dataset used throughout this tutorial:

signal = read_lsm(fetch('paramecium.lsm'))
mean, real, imag = phasor_from_signal(signal, axis=0)

# %%
# Circular cursors
# ----------------
#
# Use circular cursors to mask regions of interest in the phasor space:

cursors_real = [-0.33, 0.55]
cursors_imag = [-0.72, -0.72]

circular_mask = mask_from_circular_cursor(
    real, imag, cursors_real, cursors_imag, radius=0.2
)

# %%
# Show the circular cursors in a phasor plot:

threshold = mean > 1

plot = PhasorPlot(allquadrants=True, title='Circular cursors')
plot.hist2d(real[threshold], imag[threshold])
for i in range(len(cursors_real)):
    plot.cursor(
        cursors_real[i],
        cursors_imag[i],
        radius=0.2,
        color=CATEGORICAL[i],
        linestyle='-',
    )
plot.show()

# %%
# Polar cursors
# -------------
#
# Create a mask with two ranges of phase and modulation values:

phase_min = [-2.27, -1.22]
phase_max = [-1.57, -0.70]
modulation_min = [0.7, 0.8]
modulation_max = [0.9, 1.0]

polar_mask = mask_from_polar_cursor(
    real, imag, phase_min, phase_max, modulation_min, modulation_max
)

# %%
# Show the polar cursors in a phasor plot:

plot = PhasorPlot(allquadrants=True, title='Polar cursors')
plot.hist2d(real[threshold], imag[threshold])
for i in range(len(phase_min)):
    plot.polar_cursor(
        phase=phase_min[i],
        phase_limit=phase_max[i],
        modulation=modulation_min[i],
        modulation_limit=modulation_max[i],
        color=CATEGORICAL[i + 2],
        linestyle='-',
    )
plot.show()

# %%
# Pseudo-color images
# -------------------
#
# The cursor masks and optionally the intensity image can be
# blended to produce pseudo-colored images.
# Blending the masks from the circular cursors:

pseudo_color_image = pseudo_color(*circular_mask)

fig, ax = plt.subplots()
ax.set_title('Pseudo-color image from circular cursors')
ax.imshow(pseudo_color_image)
plt.show()

# %%
# Using the mean intensity image as a base layer to overlay the masks from
# the polar cursors:

pseudo_color_image = pseudo_color(
    *polar_mask, intensity=mean, colors=CATEGORICAL[2:]
)

fig, ax = plt.subplots()
ax.set_title('Pseudo-color image from polar cursors and intensity')
ax.imshow(pseudo_color_image)
plt.show()

# %%
# sphinx_gallery_thumbnail_number = 1
