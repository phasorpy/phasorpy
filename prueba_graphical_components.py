#%%
from phasorpy.phasor import phasor_from_lifetime
from phasorpy.plot import PhasorPlot
from phasorpy.components import graphical_component_analysis
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
frequency = 80.0
components_lifetimes = [15.0, 4.0, 1.5, 0.2]
real, imag = phasor_from_lifetime(
        frequency, components_lifetimes, [[0.4, 0.1, 0.1, 0.4], [0.5, 0.2, 0.2, 0.1]]
    )
components_lifetimes = [8.0, 4.0, 0.5]
real, imag = phasor_from_lifetime(
        frequency, components_lifetimes, [[0.1, 0.5, 0.4], [0.5, 0.4, 0.1]]
    )
components_real, components_imag = phasor_from_lifetime(frequency, components_lifetimes)
fig, ax = plt.subplots()
cursor_diameter = 0.05
fractions = graphical_component_analysis(
    real, imag, components_real, components_imag, cursor_diameter=cursor_diameter, number_of_steps=20
)
# for x, y in zip(centers_x, centers_y):
#     circle = Circle((x, y), cursor_diameter/2, fill=False, edgecolor='green')  # Create a circle patch
#     ax.add_patch(circle)
# # Loop over each nested list pair
# for x_list, y_list in zip(inner_centers_x, inner_centers_y):
#     # Iterate through the individual coordinates within each pair
#     for x, y in zip(x_list, y_list):
#         circle = Circle((x, y), cursor_diameter/2, fill=False, edgecolor='red')
#         ax.add_patch(circle)
plot = PhasorPlot(frequency = frequency, ax=ax)
plot.plot(components_real, components_imag, linestyle = '-')
plot.plot(real, imag)
print(fractions[0])
print(fractions[1])
print(fractions[2])
print(fractions[3])
# %%
from phasorpy.io import read_fbd
from phasorpy.phasor import phasor_from_signal, phasor_calibrate, phasor_center, phasor_from_lifetime
from phasorpy.plot import PhasorPlot
from phasorpy.components import graphical_component_analysis
import numpy as np

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


mask = mean <= 0.8  # Invert the condition to get values below the threshold

# Apply mask using NaN
real = np.where(mask, np.nan, real)
imag = np.where(mask, np.nan, imag)
mean = np.where(mask, np.nan, mean)


real, imag = phasor_calibrate(real, imag, real_calib, imag_calib, frequency=frequency, lifetime=4)

components_lifetimes = [8.0, 4.0, 2.0, 0.4]
# components_lifetimes = [4.5, 0.7]
components_real, components_imag = phasor_from_lifetime(frequency, components_lifetimes)

plot = PhasorPlot(frequency=80)
plot.hist2d(real, imag, cmap = 'plasma', bins = 300)
plot.plot(np.concatenate([components_real, [components_real[0]]]), np.concatenate([components_imag, [components_imag[0]]]),'-o')
cursor_diameter = 0.05
fractions = graphical_component_analysis(
    real, imag, components_real, components_imag, cursor_diameter=cursor_diameter
)
# %%
import matplotlib.pyplot as plt
plt.imshow(mean, cmap = 'Greys')
plt.colorbar()
plt.title("Mean intensity image")
plt.show()

plt.imshow(fractions[0], cmap = 'Reds')
plt.colorbar()
plt.title("Fractions of First component (tau =  3 ns)")
plt.show()
plt.imshow(fractions[1], cmap= 'Blues')
plt.colorbar()
plt.title("Fractions of Second component (tau = 0.5 ns)")
plt.show()
plt.imshow(fractions[2], cmap= 'Greens')
plt.colorbar()
plt.title("Fractions of Third component (tau = 0.4 ns)")
plt.show()

plt.hist(fractions[0].flatten(), bins = 30)
plt.title('First component')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()
plt.hist(fractions[1].flatten(), bins = 30)
plt.title('Second component')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()
plt.hist(fractions[2].flatten(), bins = 30)
plt.title('Third component')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()
# %%
from phasorpy.components import graphical_first_fraction
from phasorpy.plot import PhasorPlot
plot = PhasorPlot(frequency=80)
plot.plot([0.7, 0.55, 0.4], [0.35, 0.37, 0.39])
plot.plot([0.2, 0.9], [0.4, 0.3])

result = graphical_first_fraction([0.7, 0.55, 0.4], [0.35, 0.37, 0.39], [0.2, 0.9], [0.4, 0.3], cursor_diameter=0.05)
print(result)
# %%
from phasorpy._utils import move_cursor_along_line, line_from_components,mask_cursor
from phasorpy.components import graphical_component_analysis


result = mask_cursor([0.6, 0.5, 0.4], [0.4, 0.3, 0.2], 0.5, 0.3, 0.05)
print(result)
# %%
