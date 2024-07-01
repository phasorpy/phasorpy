#%%
from phasorpy.phasor import phasor_from_lifetime
from phasorpy.plot import PhasorPlot
from phasorpy.components import graphical_component_analysis
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
import numpy as np
frequency = 80.0
# components_lifetimes = [15.0, 4.0, 1.5, 0.2]
# real, imag = phasor_from_lifetime(
#         frequency, components_lifetimes, [[0.4, 0.1, 0.1, 0.4], [0.5, 0.2, 0.2, 0.1]]
#     )
components_lifetimes = [8.0, 4.0, 0.5]
real, imag = phasor_from_lifetime(
        frequency, components_lifetimes, [[0.1, 0.5, 0.4], [0.5, 0.4, 0.1]]
    )
components_real, components_imag = phasor_from_lifetime(frequency, components_lifetimes)
fig, ax = plt.subplots()
cursor_diameter = 0.05
number_of_steps = 100
fractions = graphical_component_analysis(
    real, imag, components_real, components_imag, cursor_diameter=cursor_diameter, number_of_steps=number_of_steps
)
plot = PhasorPlot(frequency = frequency, ax=ax)
plot.plot(components_real, components_imag, linestyle = '-')
plot.plot(real, imag)
plot.show()
print(fractions[0])
print(fractions[1])
print(fractions[2])

x_values = range(1, len(fractions[0]) + 1) 

# Plot the histogram
plt.bar(x_values, fractions[0], align='center')
# %%
from phasorpy.io import read_fbd
from phasorpy.phasor import phasor_from_signal, phasor_calibrate, phasor_center, phasor_from_lifetime
from phasorpy.plot import PhasorPlot
from phasorpy.components import graphical_component_analysis
import numpy as np
import matplotlib.pyplot as plt


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

components_lifetimes = [12.0, 2.0, 0.5]
# components_lifetimes = [4.5, 0.7]
components_real, components_imag = phasor_from_lifetime(frequency, components_lifetimes)
plot = PhasorPlot(frequency=80)
plot.hist2d(real, imag, cmap = 'plasma', bins = 300)
plot.plot(np.concatenate([components_real, [components_real[0]]]), np.concatenate([components_imag, [components_imag[0]]]),'-o')
plot.show()
cursor_diameter = 0.05
number_of_steps = 100

counts, fractions = graphical_component_analysis(
    real, imag, components_real, components_imag, cursor_diameter=cursor_diameter, number_of_steps=number_of_steps
)
for i in range(len(counts)):
    plt.plot(fractions*100, counts[i], linestyle='-')
    plt.show()

#%%
# from phasorpy.components import graphical_component_analysis
from phasorpy._utils import mask_cursor, mask_segment, line_from_components
from phasorpy.plot import PhasorPlot
import matplotlib.pyplot as plt

real = [0.6, 0.4]
imag = [0.35,0.38]
real_comp = [[0.9, 0.2],[0.3, 0.4]]
imag_comp = [[0.3, 0.4],[0.3, 0.4]]
plot = PhasorPlot(frequency=80)
# plot.plot(real, imag)
plot.plot(real_comp, imag_comp)
unit_vector, distance = line_from_components(real_comp, imag_comp)
start_point = (0.9, 0.3)
end_point = (0.2, 0.4)
plt.quiver(start_point[0], start_point[1], end_point[0] - start_point[0], end_point[1] - start_point[1], angles='xy', scale_units='xy', scale=1)
plot.show()
print(unit_vector)
print(distance)
# graphical_component_analysis(real, imag, real_comp, imag_comp, cursor_diameter=0.05, number_of_steps=10) 


# %%
