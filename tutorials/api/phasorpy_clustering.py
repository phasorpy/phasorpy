import numpy
# from matplotlib import pyplot
import matplotlib.pyplot as plt

from phasorpy.io import read_fbd

from phasorpy.phasor import (
    phasor_filter,
    phasor_threshold,
    phasor_from_signal,
    phasor_calibrate,
)
from phasorpy.plot import (
    PhasorPlot,
)

from phasorpy.color import CATEGORICAL
from phasorpy.cursors import (
    mask_from_circular_cursor,
    mask_from_elliptic_cursor,
    # mask_from_polar_cursor,
    pseudo_color,
)
from phasorpy import clustering

# import clustering # To be solved!!!
from clustering import gaussian_mixture_model
frequency = 80.0  # MHz

#=========================================================================
#Descargo los valores de calibracion.

calib_file = '../first-approach/2_40x_740_12_477_Coumarina000$EI0T.fbd'
reference_signal = read_fbd(calib_file, frame=-1, channel = 0, keepdims=False)

reference_mean, reference_real, reference_imag = phasor_from_signal(
   reference_signal
)

#=========================================================================
#Se descargan los datos en crudo.
raw_signal_file = '../first-approach/2_40x_740_10_477_MARTIN_CH5_PBP1000$EI0T.fbd'
signal = read_fbd(raw_signal_file, frame=-1, channel = 0, keepdims=False)

mean, real, imag = phasor_from_signal(signal)

#Se realiza la calibracion
real, imag = phasor_calibrate(
    real,
    imag,
    reference_real,
    reference_imag,
    frequency=frequency,
    lifetime=2.5,
)

# In order to improve visualization of the
# phasors a median filter is applied
real_filtered, imag_filtered = phasor_filter(real, imag, method='median', size=6, repeat=2)

n_components = 2
#The pixels with low intensities are excluded
mean_filtered, real_filtered, imag_filtered = phasor_threshold(mean, real_filtered, imag_filtered, mean_min=1)

centers_real, centers_imag, radio, radius_minor, angles = gaussian_mixture_model(
    real_filtered,
    imag_filtered,
    n_components = n_components,
)

elliptic_mask = mask_from_elliptic_cursor(
    real,
    imag,
    centers_real,
    centers_imag,
    radius=radio,
    radius_minor=radius_minor,
    angle=angles,
)

plot = PhasorPlot(allquadrants=True, title='Elliptic cursors')
plot.hist2d(real_filtered, imag_filtered, cmap='Greys')
for i in range(len(centers_real)):
    plot.cursor(
        centers_real[i],
        centers_imag[i],
        radius=radio[i],
        radius_minor=radius_minor[i],
        angle= 0,
        color=CATEGORICAL[i],
        linestyle='-',
    )
plot.show()

pseudo_color_image = pseudo_color(*elliptic_mask, intensity=mean_filtered)

fig, ax = plt.subplots()
ax.set_title('Pseudo-color image from elliptic cursors and intensity')
ax.imshow(pseudo_color_image)
plt.show()