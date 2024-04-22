#%%
import numpy
import matplotlib.pyplot as plt
from phasorpy.components import two_fractions_from_phasor
from phasorpy.plot import PhasorPlot
from phasorpy.phasor import phasor_from_lifetime

frequency = 80.0
components_lifetimes = [[8.0, 1.0],[4.0, 0.5]]
real, imag = phasor_from_lifetime(
        frequency, components_lifetimes[0], [0.25, 0.75]
    )
components_real, components_imag = phasor_from_lifetime(frequency, components_lifetimes[0])
plot = PhasorPlot(frequency=frequency, title = 'Phasor lying on the line between components')
plot.plot(components_real, components_imag, fmt= 'o-')
plot.plot(real, imag)
plot.show()
fraction_from_first_component, fraction_from_second_component = two_fractions_from_phasor(real, imag, components_real, components_imag)
print ('Fraction from first component: ', fraction_from_first_component)
print ('Fraction from second component: ', fraction_from_second_component) 
# %%
real1, imag1 = numpy.random.multivariate_normal(
    (0.6, 0.35), [[8e-3, 1e-3], [1e-3, 1e-3]], (100, 100)
).T
real2, imag2 = numpy.random.multivariate_normal(
    (0.4, 0.3), [[8e-3, 1e-3], [1e-3, 1e-3]], (100, 100)
).T
real = numpy.stack((real1, real2), axis=0)
imag = numpy.stack((imag1, imag2), axis=0)
components_real2, components_imag2 = phasor_from_lifetime(frequency, components_lifetimes[1])
components_real3 = numpy.stack((components_real, components_real2), axis=0)
components_imag3 = numpy.stack((components_imag, components_imag2), axis=0)
plot = PhasorPlot(frequency=frequency, title = 'Phasor lying on the line between components')
plot.plot(components_real3, components_imag3, fmt= 'o-')
plot.plot(real[0], imag[0], c='blue')
plot.plot(real[1], imag[1], c='orange')
plot.show()
fraction_from_first_component, fraction_from_second_component = two_fractions_from_phasor(real, imag, components_real3, components_imag3)
plt.figure()
plt.hist(fraction_from_first_component[0].flatten(), range=(0,1), bins=100)
plt.title('Histogram of fractions of first component ch1')
plt.xlabel('Fraction of first component')
plt.ylabel('Counts')
plt.show()

plt.figure()
plt.hist(fraction_from_first_component[1].flatten(), range=(0,1), bins=100)
plt.title('Histogram of fractions of first component ch2')
plt.xlabel('Fraction of first component')
plt.ylabel('Counts')
plt.show()

plt.figure()
plt.hist(fraction_from_second_component[0].flatten(), range=(0,1), bins=100)
plt.title('Histogram of fractions of second component ch1')
plt.xlabel('Fraction of second component')
plt.ylabel('Counts')
plt.show()
plt.figure()
plt.hist(fraction_from_second_component[1].flatten(), range=(0,1), bins=100)
plt.title('Histogram of fractions of second component ch2')
plt.xlabel('Fraction of second component')
plt.ylabel('Counts')
plt.show()

#%%
real, imag = numpy.random.multivariate_normal(
    (0.6, 0.35), [[8e-3, 1e-3], [1e-3, 1e-3]], (100, 100)
).T
plot = PhasorPlot(frequency=frequency, title = 'Point lying on the line between components')
plot.hist2d(real, imag)
plot.plot(*phasor_from_lifetime(frequency, components_lifetimes), fmt= 'o-')
plot.show()
fraction_from_first_component, fraction_from_second_component = two_fractions_from_phasor(real, imag, components_real, components_imag)

plt.figure()
plt.hist(fraction_from_first_component.flatten(), range=(0,1), bins=100)
plt.title('Histogram of fractions of first component')
plt.xlabel('Fraction of first component')
plt.ylabel('Counts')
plt.show()

plt.figure()
plt.hist(fraction_from_second_component.flatten(), range=(0,1), bins=100)
plt.title('Histogram of fractions of second component')
plt.xlabel('Fraction of second component')
plt.ylabel('Counts')
plt.show()

# %%
