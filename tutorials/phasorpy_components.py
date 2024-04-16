"""
Component analysis
===========

An introduction to component analysis in the phasor space.



"""

# %%
# Import required modules, functions, and classes:

import math

import numpy

from phasorpy.components import two_fractions_from_phasor
from phasorpy.plot import PhasorPlot, components_histogram

# %%
# Component mixtures
# ------------------
#
# Show linear combinations of phasor coordinates or ranges thereof:

real, imag, weights = numpy.array(
    [[0.1, 0.2, 0.5, 0.9], [0.3, 0.4, 0.5, 0.3], [2, 1, 2, 1]]
)

plot = PhasorPlot(frequency=80.0, title='Component mixtures')
plot.components(real, imag, linestyle='', fill=True, facecolor='lightyellow')
plot.components(real, imag, weights)
plot.show()

# %%
# 2D Histogram
# ------------
#
# Plot large number of phasor coordinates as a 2D histogram:

real, imag = numpy.random.multivariate_normal(
    (0.6, 0.4), [[3e-3, -1e-3], [-1e-3, 1e-3]], (256, 256)
).T
plot = PhasorPlot(frequency=80.0, title='2D Histogram')
plot.hist2d(real, imag)
plot.show()

# %%
# Contours
# --------
#
# Plot the contours of the density of phasor coordinates:

plot = PhasorPlot(frequency=80.0, title='Contours')
plot.contour(real, imag)
plot.show()


# %%
# Image
# -----
#
# Plot the image of a custom-colored 2D histogram:

plot = PhasorPlot(frequency=80.0, title='Image (not implemented yet)')
# plot.imshow(image)
plot.show()

# %%
# Combined plots
# --------------
#
# Multiple plots can be combined:

real2, imag2 = numpy.random.multivariate_normal(
    (0.9, 0.2), [[2e-4, -1e-4], [-1e-4, 2e-4]], 4096
).T

plot = PhasorPlot(
    title='Combined plots', xlim=(0.35, 1.03), ylim=(0.1, 0.59), grid=False
)
plot.hist2d(real, imag, bins=64, cmap='Blues')
plot.contour(real, imag, bins=48, levels=3, cmap='summer_r', norm='log')
plot.hist2d(real2, imag2, bins=64, cmap='Oranges')
plot.plot(0.6, 0.4, '.', color='tab:blue')
plot.plot(0.9, 0.2, '.', color='tab:orange')
plot.polar_cursor(math.atan(0.4 / 0.6), math.hypot(0.6, 0.4), color='tab:blue')
plot.polar_cursor(
    math.atan(0.2 / 0.9), math.hypot(0.9, 0.2), color='tab:orange'
)
plot.semicircle(frequency=80.0, color='tab:purple')
plot.show()

# %%
# All quadrants
# -------------
#
# Create an empty phasor plot showing all four quadrants:

plot = PhasorPlot(allquadrants=True, title='All quadrants')
plot.show()

# %%
# Matplotlib axes
# ---------------
#
# The PhasorPlot class can use an existing matlotlib axes.
# The `PhasorPlot.ax` attribute provides access to the underlying
# matplotlib axes, for example, to add annotations:

from matplotlib import pyplot

ax = pyplot.subplot(1, 1, 1)
plot = PhasorPlot(ax=ax, allquadrants=True, title='Matplotlib axes')
plot.hist2d(real, imag, cmap='Blues')
plot.ax.annotate(
    '0.6, 0.4',
    xy=(0.6, 0.4),
    xytext=(0.2, 0.2),
    arrowprops=dict(arrowstyle='->'),
)
pyplot.show()


# %%
# plot_phasor function
# --------------------
#
# The :py:func:`phasorpy.plot.plot_phasor` function provides a simpler
# alternative to plot phasor coordinates in a single statement:

from phasorpy.plot import plot_phasor

plot_phasor(real[0, :32], imag[0, :32], fmt='.', frequency=80.0)

# %%
# sphinx_gallery_thumbnail_number = 9
