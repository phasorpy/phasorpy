# %%
from phasorpy.phasor import phasor_from_signal, phasor_calibrate, phasor_filter, phasor_threshold
from phasorpy.plot import PhasorPlot, plot_signal_image
from phasorpy.io import read_ptu
import tifffile

# data = 'test_data/ptu_files/caprida_test01_001.ptu'
# calibration_data = 'test_data/ptu_files/coumarin_calib_001.ptu'

# data_signal = read_ptu(data, frame=-1, channel=0, keepdims=False)
# calibration_signal = read_ptu(calibration_data, frame=-1, channel=0, keepdims=False)

data = 'test_data/embryoFLUTE/Embryo.tif'
calibration_data = 'test_data/embryoFLUTE/Fluorescein_Embryo.tif'

data_signal = tifffile.imread(data)
calibration_signal = tifffile.imread(calibration_data)

plot_signal_image(data_signal, axis = 0)
mean, real, imag = phasor_from_signal(data_signal, axis = 0, harmonic=[1,2])
mean_calib, real_calib, imag_calib = phasor_from_signal(calibration_signal,axis=0, harmonic=[1,2])

real, imag = phasor_calibrate(real, imag, real_calib, imag_calib, frequency=80, harmonic=[1,2], lifetime=4.2, skip_axis=0)

real, imag = phasor_filter(real, imag, repeat=3, axes=(1,2))

mean, real, imag = phasor_threshold(mean, real, imag, 1)

plot_harm_1 = PhasorPlot(frequency=80, title='H1')
plot_harm_1.hist2d(real[0], imag[0], cmap='plasma', bins=300)

plot_harm_2 = PhasorPlot(frequency=80*2, title='H2')
plot_harm_2.hist2d(real[1], imag[1], cmap='plasma', bins=300)

# %%


# %%
