"""
Clustering
==========

An introduction to clustering phasor points to be used with cursors.

"""

# %%
# Import required modules, functions, and classes:

import matplotlib.pyplot as plt
import numpy

from phasorpy.cluster import phasor_cluster_gmm
from phasorpy.color import CATEGORICAL
from phasorpy.cursors import (
    mask_from_circular_cursor,
    mask_from_elliptic_cursor,
    mask_from_polar_cursor,
    pseudo_color,
)
from phasorpy.datasets import fetch
from phasorpy.io import signal_from_imspector_tiff
from phasorpy.phasor import (
    phasor_calibrate,
    phasor_filter_median,
    phasor_from_signal,
    phasor_threshold,
)
from phasorpy.plot import PhasorPlot

signal = signal_from_imspector_tiff(fetch('Embryo.tif'))
frequency = signal.attrs['frequency']

mean, real, imag = phasor_from_signal(signal, axis='H')

# ====================================================
# ====================Calibration=====================
# ====================================================
reference_signal = signal_from_imspector_tiff(fetch('Fluorescein_Embryo.tif'))

reference_mean, reference_real, reference_imag = phasor_from_signal(
    reference_signal, axis=0
)

real, imag = phasor_calibrate(
    real,
    imag,
    reference_mean,
    reference_real,
    reference_imag,
    frequency=frequency,
    lifetime=4.2,
)

# ====================================================
# ====================Filtering=======================
# ====================================================

mean_filtered, real_filtered, imag_filtered = phasor_filter_median(
    mean, real, imag, size=3, repeat=2
)

mean_filtered, real_filtered, imag_filtered = phasor_threshold(
    mean_filtered, real_filtered, imag_filtered, mean_min=1
)

# ====================================================
# ====================clustering======================
# ====================================================

clusters = 2

centers_real, centers_imag, radio, radius_minor, angles = phasor_cluster_gmm(
    real_filtered,
    imag_filtered,
    clusters=clusters,
)

# ====================================================
# ====================Ellipses========================
# ====================================================

elliptic_mask = mask_from_elliptic_cursor(
    real,
    imag,
    centers_real,
    centers_imag,
    radius=radio,
    radius_minor=radius_minor,
    angle=angles,
)

plot = PhasorPlot(frequency=frequency, title='Elliptic cursors')
plot.hist2d(real_filtered, imag_filtered, cmap='Greys', bins=500)
for i in range(len(centers_real)):
    plot.cursor(
        centers_real[i],
        centers_imag[i],
        radius=radio[i],
        radius_minor=radius_minor[i],
        angle=angles[i],
        color=CATEGORICAL[i],
        linestyle='-',
    )
plot.show()

pseudo_color_image = pseudo_color(*elliptic_mask, intensity=mean_filtered)

fig, ax = plt.subplots()
ax.set_title('Pseudo-color image from elliptic cursors and intensity')
ax.imshow(pseudo_color_image)
plt.show()

# =============================================================================
# ============================Polar cursors=====================================
# =============================================================================
clusters = 1  # This is a test, to try and improve the results from GMM.

centers_real, centers_imag, radio, radius_minor, angles = phasor_cluster_gmm(
    real_filtered,
    imag_filtered,
    clusters=clusters,
)
modulation_center = []
phase_center = []
for i in range(len(centers_real)):
    modulation_center.append(
        numpy.sqrt(centers_real[i] ** 2 + centers_imag[i] ** 2)
    )
    phase_center.append(numpy.arctan2(centers_imag[i], centers_real[i]))

epsilon = 0.1
delta = 0.02

phase_min = [phase_center[0], (phase_center[0] - 2 * epsilon)]
phase_max = [(phase_center[0] + 2 * epsilon), phase_center[0]]
modulation_min = [
    (modulation_center[0] - delta),
    (modulation_center[0] - delta),
]
modulation_max = [
    (modulation_center[0] + delta),
    (modulation_center[0] + delta),
]

polar_mask = mask_from_polar_cursor(
    real,
    imag,
    phase_min,
    phase_max,
    modulation_min,
    modulation_max,
)

plot = PhasorPlot(allquadrants=False, title='Polar cursors')
plot.hist2d(real_filtered, imag_filtered, cmap='Greys')
for i in range(len(phase_min)):
    plot.polar_cursor(
        phase=phase_min[i],
        phase_limit=phase_max[i],
        modulation=modulation_min[i],
        modulation_limit=modulation_max[i],
        color=CATEGORICAL[i],
        linestyle='-',
    )
plot.show()

pseudo_color_image = pseudo_color(*polar_mask, intensity=mean)

fig, ax = plt.subplots()
ax.set_title('Pseudo-color image from elliptic cursors and intensity')
ax.imshow(pseudo_color_image)
plt.show()

# =============================================================================
# ==================Circular cursors (with separation)=========================
# =============================================================================
real_cf_1 = centers_real[0] + epsilon * numpy.sin(angles[0])
imag_cf_1 = centers_imag[0] + epsilon * numpy.cos(angles[0])

real_cf_2 = centers_real[0] - epsilon * numpy.sin(angles[0])
imag_cf_2 = centers_imag[0] - epsilon * numpy.cos(angles[0])

cursors_real = [real_cf_1, real_cf_2]
cursors_imag = [imag_cf_1, imag_cf_2]
radius = [epsilon, epsilon]
cursors_masks = mask_from_circular_cursor(
    real, imag, cursors_real, cursors_imag, radius=radius
)

phasorplot = PhasorPlot(frequency=frequency, title='Cursors')
phasorplot.hist2d(real_filtered, imag_filtered)
for i in range(len(cursors_real)):
    phasorplot.circle(
        cursors_real[i],
        cursors_imag[i],
        radius=radius[i],
        color=CATEGORICAL[i],
        linestyle='-',
    )
phasorplot.show()

pseudo_color_image = pseudo_color(*cursors_masks, intensity=mean)

fig, ax = plt.subplots()
ax.set_title('Pseudo-color image from circular cursors')
ax.imshow(pseudo_color_image)
plt.show()
