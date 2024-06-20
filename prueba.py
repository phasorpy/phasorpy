#%%
from phasorpy.io import read_fbd
from phasorpy.phasor import phasor_from_signal, phasor_calibrate, phasor_center, phasor_from_lifetime
from phasorpy.plot import PhasorPlot
from phasorpy.components import fractions_from_phasor, two_fractions_from_phasor, fractions_with_lstsq

data_path = '/Users/bruno/Downloads/drive-download-20240521T170940Z-001/2_40x_740_10_477_MARTIN_CH5_PAA1000$EI0T.fbd'
calibration_path = '/Users/bruno/Downloads/drive-download-20240521T170940Z-001/2_40x_740_12_477_Coumarina000$EI0T.fbd'

data_signal = read_fbd(data_path, frame=-1,channel=0, keepdims=False)
calibration_signal = read_fbd(calibration_path, frame=-1,channel=0, keepdims=False)

mean, real, imag = phasor_from_signal(data_signal)
mean_calib, real_calib, imag_calib = phasor_from_signal(calibration_signal)

from skimage.filters import median

for _ in range(3):
    real = median(real)
    imag = median(imag)

mask = mean > 1
real = real[mask]
imag = imag[mask]

real, imag = phasor_calibrate(real, imag, real_calib, imag_calib, frequency=80, lifetime=2.5)
plot = PhasorPlot(frequency=80)
plot.hist2d(real, imag, cmap = 'plasma', bins = 300)



#%%
fractions = fractions_from_phasor(real, imag, components_real, components_imag, axis = 0)
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.hist(
    fractions[0].flatten(),
    range=(0, 1),
    bins=100,
    alpha=0.75,
    label='8 ns',
)
ax.hist(
    fractions[1].flatten(),
    range=(0, 1),
    bins=100,
    alpha=0.75,
    label='0.5 ns',
)
ax.hist(
    fractions[2].flatten(),
    range=(0, 1),
    bins=100,
    alpha=0.75,
    label='2 ns',
)
ax.set_title('Histograms of fractions of 3 components')
ax.set_xlabel('Fraction')
ax.set_ylabel('Counts')
ax.legend()
plt.tight_layout()
plt.show()
print(fractions[0].shape)
print(fractions[0].min())
print(fractions[0].max())
#%%
fractions = fractions_with_lstsq(real, imag, components_real, components_imag, axis = 0)
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.hist(
    fractions[0].flatten(),
    range=(0, 1),
    bins=100,
    alpha=0.75,
    label='8 ns',
)
ax.hist(
    fractions[1].flatten(),
    range=(0, 1),
    bins=100,
    alpha=0.75,
    label='0.5 ns',
)
ax.hist(
    fractions[2].flatten(),
    range=(0, 1),
    bins=100,
    alpha=0.75,
    label='2 ns',
)
ax.set_title('Histograms of fractions of 3 components')
ax.set_xlabel('Fraction')
ax.set_ylabel('Counts')
ax.legend()
plt.tight_layout()
plt.show()
print(fractions.shape)
print(fractions.min())
print(fractions.max())

#%%
from phasorpy.phasor import phasor_from_lifetime
from phasorpy.plot import PhasorPlot
from phasorpy.components import fractions_from_phasor, two_fractions_from_phasor
import numpy as np
frequency = 80.0
components_lifetimes = [8.0, 1.0, 3.0]
real, imag = phasor_from_lifetime(
        frequency, components_lifetimes, [[0.10, 0.20, 0.70],[0.10, 0.20, 0.70]]
    )
components_real, components_imag = phasor_from_lifetime(frequency, components_lifetimes)
# plot = PhasorPlot(frequency=frequency, title = 'Phasor lying on the line between components')
# plot.plot(components_real, components_imag, fmt= 'o-')
# plot.plot(real, imag)
# plot.show()
# real  = np.expand_dims(real, axis=-1).repeat(3, axis=-1)
# imag  = np.expand_dims(imag, axis=-1).repeat(3, axis=-1)
print('Test 3 components')
# fractions = four_fractions_from_phasor(real, imag, components_real, components_imag)
fractions = fractions_from_phasor(real, imag, components_real, components_imag)
print(fractions)
print('Test 2 components')
fractions = fractions_from_phasor(real[0], imag[0], components_real[:2], components_imag[:2])
print ('Fraction from first component: ', fractions[0])
print ('Fraction from second component: ', fractions[1])
fractions = two_fractions_from_phasor(real, imag, components_real[:2], components_imag[:2])
print('two fractions phasor function: ')
print ('Fraction from first component: ', fractions[0])
print ('Fraction from second component: ', fractions[1])


#%%
#PRUEBA 4 COMPONENTES SINTETICO
from phasorpy.phasor import phasor_from_lifetime
from phasorpy.plot import PhasorPlot
from phasorpy.components import fractions_from_phasor, fractions_with_lstsq
import numpy as np

components_lifetimes = [0.3, 1.0, 3.0, 8.0]
frequency = [80.0, 160.0]
components_real, components_imag = phasor_from_lifetime(frequency, components_lifetimes)

print('Test 4 components, 1 coordinate 2 harm')
real, imag = phasor_from_lifetime(
        frequency, components_lifetimes, [0.10, 0.20, 0.45, 0.25],
    )
fractions = fractions_with_lstsq(real, imag, components_real, components_imag, axis = 0)
print('Fractions: ',fractions)
print('Fractions shape: ',fractions[0].shape)
print('First fraction', fractions[0])

print('Test 4 components, 2 coordinates 2 harm')
real, imag = phasor_from_lifetime(
        frequency, components_lifetimes, [[0.10, 0.20, 0.45, 0.25],[0.15, 0.20, 0.40, 0.25]],
    )
fractions = fractions_with_lstsq(real, imag, components_real, components_imag, axis = 0)
print('Fractions: ',fractions)
print('Fractions shape: ',fractions[0].shape)
print('First fraction', fractions[0])

print('Test 4 components 2D array 2 harm')
real, imag = phasor_from_lifetime(
        frequency, components_lifetimes, [[0.10, 0.20, 0.45, 0.25],[0.15, 0.20, 0.40, 0.25], [0.15, 0.20, 0.40, 0.25]],
    )

real  = np.expand_dims(real, axis=-1).repeat(3, axis=-1)
imag  = np.expand_dims(imag, axis=-1).repeat(3, axis=-1)
fractions = fractions_with_lstsq(real, imag, components_real, components_imag, axis = 0)
print('Fractions: ',fractions)
print('Fractions shape: ',fractions[0].shape)
print('First fraction', fractions[0])
# %%
#PRUEBA 3 COMPONENTES SINTETICO
from phasorpy.phasor import phasor_from_lifetime
from phasorpy.plot import PhasorPlot
from phasorpy.components import fractions_from_phasor, fractions_with_lstsq
import numpy as np

components_lifetimes = [0.3, 3.0, 8.0]
frequency = 80.0
components_real, components_imag = phasor_from_lifetime(frequency, components_lifetimes)

print('Test 3 components, 1 coordinate')
real, imag = phasor_from_lifetime(
        frequency, components_lifetimes, [0.35, 0.20, 0.45],
    )
fractions = fractions_with_lstsq(real, imag, components_real, components_imag, axis = 0)
print('Fractions: ',fractions)
print('Fractions shape: ',fractions[0].shape)
print('First fraction', fractions[0])

print('Test 3 components, 2 coordinates')
real, imag = phasor_from_lifetime(
        frequency, components_lifetimes, [[0.35, 0.20, 0.45],[0.15, 0.45, 0.40]],
    )
fractions = fractions_with_lstsq(real, imag, components_real, components_imag, axis = 0)
print('Fractions: ',fractions)
print('Fractions shape: ',fractions[0].shape)
print('First fraction', fractions[0])

print('Test 3 components, 2 coordinates')
real, imag = phasor_from_lifetime(
        frequency, components_lifetimes, [[0.35, 0.20, 0.45],[0.15, 0.45, 0.40], [0.15, 0.45, 0.40]],
    )

real  = np.expand_dims(real, axis=-1).repeat(4, axis=-1)
imag  = np.expand_dims(imag, axis=-1).repeat(4, axis=-1)
fractions = fractions_with_lstsq(real, imag, components_real, components_imag, axis = 0)
print('Fractions: ',fractions)
print('Fractions shape: ',fractions[0].shape)
print('First fraction', fractions[0])
# %%
# PRUEBA LSTSQ
from phasorpy.phasor import phasor_from_lifetime
from phasorpy._utils import project_phasor_to_line
from phasorpy.plot import PhasorPlot
from phasorpy.components import two_fractions_from_phasor, fractions_from_phasor, fractions_with_lstsq
import numpy as np
frequency = 80.0
components_lifetimes = [8.0, 1.0, 3.0]
real, imag = phasor_from_lifetime(
        frequency, components_lifetimes, [[0.10, 0.20, 0.70],[0.10, 0.20, 0.70]]
    )
# components_lifetimes = [8.0, 1.0]
# real, imag = phasor_from_lifetime(
#         frequency, components_lifetimes, [[0.30, 0.70],[0.40, 0.60]]
#     )
components_real, components_imag = phasor_from_lifetime(frequency, components_lifetimes)
# print('Test 3 components')
# fractions = fractions_with_lstsq(real, imag, components_real, components_imag)
# print(fractions)
print('Test 2 components')
fractions = fractions_with_lstsq(real[0], imag[0], components_real[:2], components_imag[:2])
print ('Fraction from first component: ', fractions)
print ('Fraction from second component: ', fractions)
fractions = two_fractions_from_phasor(real, imag, components_real[:2], components_imag[:2])
print('two fractions phasor function: ')
print ('Fraction from first component: ', fractions[0])
print ('Fraction from second component: ', fractions[1])

#%%
from phasorpy.io import read_fbd
from phasorpy.phasor import phasor_from_signal, phasor_calibrate, phasor_center

data_path = 'test_data/FBDfiles-DIVER/BUENOS/convallaria_000$EI0S.fbd'
calibration_path = 'test_data/FBDfiles-DIVER/BUENOS/RH110CALIBRATION_000$EI0S.fbd'
data_signal = read_fbd(data_path, frame=-1,channel=0, keepdims=False)
calibration_signal = read_fbd(calibration_path, frame=-1,channel=0, keepdims=False)
mean, real, imag = phasor_from_signal(data_signal, harmonic=[1,2])
mean_calib, real_calib, imag_calib = phasor_from_signal(calibration_signal, harmonic=[1,2])
reference_real, reference_imag = phasor_center(real_calib, imag_calib, skip_axes=0)
real, imag = phasor_calibrate(real, imag, reference_real, reference_imag, frequency=80, lifetime=4, skip_axes=0)
# %%
