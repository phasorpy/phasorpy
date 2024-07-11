#%%
from phasorpy.cursors import mask_from_polar_cursor
import numpy
phase = numpy.array([[337, 306, 227], [21, 231, 235], [244, 328, 116]])
mod = [[0.22, 0.40, 0.81], [0.33, 0.43, 0.36], [0.015, 0.82, 0.58]]
phase_range = numpy.array([[0, 270],[0, 270],[0, 270], [0, 270]])
# print(phase_range.shape)
mod_range = numpy.array([[0, 0.5],[0, 0.5], [0, 0.5], [0, 0.5]])
# print(phase.shape)
result = mask_from_polar_cursor(phase, mod, phase_range, mod_range)

print(result.shape)
print(result)

#%%
from phasorpy.cursors import mask_from_polar_cursor
mask_from_polar_cursor([5,100],[0.2, 0.4],[[50, 150], [0, 270]],[[0.2, 0.5], [0, 0.3]])
# import numpy
# phase = numpy.array([5,100])
# mod = numpy.array([0.2, 0.4])
# phase_range = numpy.asarray([50, 150])
# print(phase.shape)
# print(phase_range.shape)
# mod_range = numpy.asarray([0.2, 0.5])
# result = mask_from_polar_cursor(phase, mod, phase_range, mod_range)
# print(result.shape)
# print(result)
# %%
from phasorpy.cursors import mask_from_circular_cursor

result = mask_from_circular_cursor([0,1],[0,1], [0,0.9,2,3], [0,0.9,2,3], radius=[0.1,0.05,0.1,0.1])
print(result)
print(result.shape)

# %%
from phasorpy.cursors import pseudocolor, mask_from_circular_cursor
import numpy as np
mean = np.array([[0, 1], [2, 3]])
masks = [np.array([[True, False], [False, True]]), np.array([[False, True], [True, False]])]
CATEGORICAL = np.array([[255, 0, 0], [0, 255, 0]])
print(mean.shape)
print(mean)
colored_array = pseudocolor(mean, masks)
print(colored_array)
print(colored_array.shape)
# %%
from phasorpy.phasor import phasor_from_signal, phasor_calibrate
from phasorpy.cursors import mask_from_circular_cursor, pseudocolor, mask_from_polar_cursor
from phasorpy.io import read_fbd
from phasorpy.plot import PhasorPlot
from phasorpy.color import CATEGORICAL
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


data_path = 'test_data/FBDfiles-DIVER/BUENOS/convallaria_000$EI0S.fbd'
calibration_path = 'test_data/FBDfiles-DIVER/BUENOS/RH110CALIBRATION_000$EI0S.fbd'

data_signal = read_fbd(data_path, frame=-1,channel=0, keepdims=False)
calibration_signal = read_fbd(calibration_path, frame=-1,channel=0, keepdims=False)

mean, real, imag = phasor_from_signal(data_signal)
mean_calib, real_calib, imag_calib = phasor_from_signal(calibration_signal)

frequency = 80
from skimage.filters import median

for _ in range(5):
    real = median(real)
    imag = median(imag)

mask = mean <= 0.8

real = np.where(mask, np.nan, real)
imag = np.where(mask, np.nan, imag)
mean = np.where(mask, np.nan, mean)


real, imag = phasor_calibrate(real, imag, real_calib, imag_calib, frequency=frequency, lifetime=4)
fig, ax = plt.subplots()
plot = PhasorPlot(frequency=80, ax = ax)
plot.hist2d(real, imag, cmap = 'plasma', bins = 300)
radius = 0.07
x = [0.7, 0.5, 0.38]
y = [0.3, 0.35, 0.35]
for i in range(len(x)):
    circle = Circle((x[i], y[i]), radius, fill=False, edgecolor=CATEGORICAL[i], linewidth=3)
    ax.add_patch(circle)
plt.show()
mask = mask_from_circular_cursor(real, imag, x, y, radius=radius)
colored_array = pseudocolor(mean,mask)
plt.imshow(colored_array)
plt.axis('off')
plt.show()
# %%
from phasorpy.cursors import pseudo_color

result = pseudo_color([0, 1], [[True, False], [False, True]]) 
print(result.shape)
result
# %%
