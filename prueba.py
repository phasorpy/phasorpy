#%%
from phasorpy.io import read_fbd, read_ptu
from phasorpy.phasor import phasor_from_signal, phasor_calibrate, phasor_center
from phasorpy.plot import PhasorPlot


# data_path = 'test_data/FBDfiles-DIVER/BUENOS/convallaria_000$EI0S.fbd'
# calibration_path = 'test_data/FBDfiles-DIVER/BUENOS/RH110CALIBRATION_000$EI0S.fbd'

# data_signal = read_fbd(data_path, frame=-1,channel=0, keepdims=False)
# calibration_signal = read_fbd(calibration_path, frame=-1,channel=0, keepdims=False)

data_path = 'test_data/ptu_files/caprida_test01_001.ptu'
calibration_path = 'test_data/ptu_files/coumarin_calib_002.ptu'

data_signal = read_ptu(data_path, frame=-1,channel=0, keepdims=False)
calibration_signal = read_ptu(calibration_path, frame=-1,channel=0, keepdims=False)

mean, real, imag = phasor_from_signal(data_signal, harmonic=[1,2])
mean_calib, real_calib, imag_calib = phasor_from_signal(calibration_signal, harmonic=[1,2])
from skimage.filters import median
for _ in range(5):
    real = median(real)
    imag = median(imag)
    real_calib = median(real_calib)
    imag_calib = median(imag_calib)

mask = mean > 1
real = real[:,mask]
imag = imag[:,mask]
mask = mean > 1
real_calib = real_calib[:,mask]
imag_calib = imag_calib[:,mask]

real, imag = phasor_calibrate(real, imag, real_calib, imag_calib, frequency=[80,160], lifetime=2.5, skip_axes=0)
plot = PhasorPlot(frequency=80)
plot.hist2d(real[0], imag[0], cmap='plasma', bins = 300)
plot = PhasorPlot(frequency=80)
plot.hist2d(real[1], imag[1], cmap='plasma', bins = 300)
# %%
import numpy
from phasorpy.phasor import phasor_center

SYNTH_DATA_ARRAY = numpy.array([[50, 1], [1, 1]])
SYNTH_DATA_LIST = [1, 2, 4]
SYNTH_PHI = numpy.array([[0.5, 0.5], [0.5, 0.5]])
SYNTH_MOD = numpy.array([[2, 2], [2, 2]])

# stacked_array = numpy.stack((SYNTH_DATA_ARRAY, SYNTH_DATA_ARRAY/2, SYNTH_DATA_ARRAY/3), axis=0)
# reference_real = [0.5,0.25, 0.1]
# reference_imag = [0.3,0.2, 0.1]


real, imag = phasor_center(SYNTH_DATA_ARRAY, SYNTH_DATA_ARRAY, keepdims=True)
print(real)
print(imag)
# %%
