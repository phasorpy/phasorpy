"""
Cursors
=======

An introduction to selecting phasor coordinates using cursors.

"""

# %%
# Import required modules, functions, and classes:

from phasorpy.color import CATEGORICAL
from phasorpy.cursor import (
    mask_from_circular_cursor,
    mask_from_elliptic_cursor,
    mask_from_polar_cursor,
    pseudo_color,
)
from phasorpy.datasets import fetch
from phasorpy.filter import phasor_threshold
from phasorpy.io import signal_from_lsm
from phasorpy.phasor import phasor_from_signal
from phasorpy.plot import PhasorPlot, plot_image

# %%
# Load a hyperspectral dataset used throughout this tutorial:

signal = signal_from_lsm(fetch('paramecium.lsm'))
mean, real, imag = phasor_from_signal(signal, axis=0)

# remove coordinates with zero intensity
mean_thresholded, real, imag = phasor_threshold(mean, real, imag, mean_min=1)

# %%
# Circular cursors
# ----------------
#
# Use circular cursors to mask regions of interest in the phasor space.
# Define two cursors by specifying their real and imaginary coordinates
# and radii:

cursor_real = [-0.33, 0.54]
cursor_imag = [-0.72, -0.74]
radius = [0.2, 0.22]

circular_mask = mask_from_circular_cursor(
    real, imag, cursor_real, cursor_imag, radius=radius
)

# %%
# Show the circular cursors in a phasor plot:

plot = PhasorPlot(allquadrants=True, title='Circular cursors')
plot.hist2d(real, imag, cmap='Greys')
plot.cursor(
    cursor_real,
    cursor_imag,
    radius=radius,
    color=CATEGORICAL[:2],
    label=['cursor 0', 'cursor 1'],
)
plot.show()

# %%
#
# The cursor masks can be blended to produce a pseudo-colored image.
# Each cursor's region is assigned a different color:

pseudo_color_image = pseudo_color(*circular_mask)

plot_image(
    pseudo_color_image, title='Pseudo-color image from circular cursors'
)

# %%
# The pseudo-color image is numpy array with values between 0 and 1 (RGB)
# that can be further processed or saved as needed.

# %%
# Elliptical cursors
# ------------------
#
# Use elliptical cursors to mask better-defined regions of interest in the
# phasor space. Elliptical cursors allow independent control of the radii,
# which can better match elongated clusters in phasor space:

radius = [0.1, 0.06]  # major axis
radius_minor = [0.3, 0.25]  # minor axis

elliptic_mask = mask_from_elliptic_cursor(
    real,
    imag,
    cursor_real,
    cursor_imag,
    radius=radius,
    radius_minor=radius_minor,
)

# %%
# Show the elliptical cursors in a phasor plot:

plot = PhasorPlot(allquadrants=True, title='Elliptical cursors')
plot.hist2d(real, imag, cmap='Greys')
plot.cursor(
    cursor_real,
    cursor_imag,
    radius=radius,
    radius_minor=radius_minor,
    color=CATEGORICAL[:2],
    label=['cursor 0', 'cursor 1'],
)
plot.show()

# %%
#
# The mean intensity image can be used as a base layer to overlay
# the masks from the elliptical cursors:

pseudo_color_image = pseudo_color(*elliptic_mask, intensity=mean)

plot_image(
    pseudo_color_image,
    title='Pseudo-color image from elliptical cursors and intensity',
)

# %%
# Polar cursors
# -------------
#
# Use polar cursors to select regions of interest in the phasor space based
# on phase and modulation ranges:

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
plot.hist2d(real, imag, cmap='Greys')
plot.polar_cursor(
    phase=phase_min,
    phase_limit=phase_max,
    modulation=modulation_min,
    modulation_limit=modulation_max,
    color=CATEGORICAL[2:4],  # use different colors
    label=['cursor 0', 'cursor 1'],
)
plot.show()

# %%
# The thresholded mean intensity image can be used as a base layer to
# overlay the masks from the polar cursors. Values below the threshold are
# transparent (white):

pseudo_color_image = pseudo_color(
    *polar_mask, intensity=mean_thresholded, colors=CATEGORICAL[2:]
)

plot_image(
    pseudo_color_image,
    title='Pseudo-color image from\npolar cursors and thresholded intensity',
)

# sphinx_gallery_start_ignore
# sphinx_gallery_thumbnail_number = 1
# mypy: allow-untyped-defs, allow-untyped-calls
# mypy: disable-error-code="arg-type"
# sphinx_gallery_end_ignore
