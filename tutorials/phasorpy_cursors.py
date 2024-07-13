"""
Cursors
=======

An introduction to selecting phasor coordinates using cursors.

"""

# %%
# Import required modules, functions, and classes:

import tifffile
from matplotlib import pyplot
from phasorpy.cursors import mask_from_circular_cursor, mask_from_polar_cursor, pseudo_color
from phasorpy.datasets import fetch
from phasorpy.phasor import phasor_from_signal, phasor_to_polar
from phasorpy.plot import PhasorPlot
from phasorpy.color import CATEGORICAL

# %%
# Circular cursors
# ----------------
#
# Use circular cursors in the phasor space to mask regions of interest:

signal = tifffile.imread(fetch('paramecium.lsm'))
mean, real, imag = phasor_from_signal(signal, axis=0)
cursors_real = [-0.33, 0.55]
cursors_imag = [-0.72, -0.72]
circular_mask = mask_from_circular_cursor(real, imag, cursors_real, cursors_imag, radius=0.2)

# %%
# Show the circular cursors in the phasor plot:

threshold = mean > 1

plot = PhasorPlot(allquadrants=True, title='Two circular cursors')
plot.hist2d(real[threshold], imag[threshold])
for i in range(len(cursors_real)):
    plot.cursor(
        cursors_real[i],
        cursors_imag[i],
        radius=0.2,
        color=CATEGORICAL[i],
        linestyle='-',
    )

# %%
# Polar cursor
# ------------
#
# Create a mask with phase and modulation values:

phase, mod = phasor_to_polar(real, imag)

phase_range = [[-2.27, -1.57], [-1.22, -0.70]]
modulation_range = [[0.7, 0.9], [0.8, 1.0]]
polar_mask = mask_from_polar_cursor(phase, mod, phase_range, modulation_range)

# %%
# Show cursors in the phasor plot:

plot = PhasorPlot(allquadrants=True, title='Two polar cursors')
plot.hist2d(real[threshold], imag[threshold], cmap='Blues')
for i in range(len(phase_range[0])):
    plot.polar_cursor(
        phase=phase_range[i][0],
        phase_limit=phase_range[i][1],
        modulation=modulation_range[i][0],
        modulation_limit=modulation_range[i][1],
        color=CATEGORICAL[i+2],
        linestyle='-',
    )

# %%
# Pseudo-color
# ------------
#
# Average images can be pseudo-colored (segmented) using cursor's masks.
# Segmented image with circular cursors:

segmented_image = pseudo_color(mean, circular_mask)

fig, ax = pyplot.subplots()
ax.set_title('Segmented image with circular cursors')
ax.imshow(segmented_image)

#%%
# Segmented image with polar cursors:

segmented_image = pseudo_color(mean, polar_mask, colors=CATEGORICAL[2:])

fig, ax = pyplot.subplots()
ax.set_title('Segmented image with polar cursors')
ax.imshow(segmented_image)

# %%
# sphinx_gallery_thumbnail_number = 1
