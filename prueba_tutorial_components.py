#%%
import numpy
from phasorpy.components import fractional_intensities_from_phasor, fractional_intensities_from_phasor_old
from phasorpy.plot import PhasorPlot, components_histogram
from phasorpy.phasor import phasor_from_lifetime
import time

fractions = numpy.array(
    [[1, 0], [0.25, 0.75], [0, 1]]
)
frequency = 80.0
components_lifetimes = [8.0, 1.0]
coordinates = phasor_from_lifetime(
        frequency, components_lifetimes, fractions
    )
#%%
plot = PhasorPlot(frequency=frequency, title = 'Point lying on the line between components')
plot.plot(*phasor_from_lifetime(frequency, components_lifetimes), fmt= 'o-')
plot.plot(coordinates[0][1], coordinates[1][1])
fractions = fractional_intensities_from_phasor(coordinates[0][1], coordinates[1][1], *phasor_from_lifetime(frequency, components_lifetimes))
print (fractions) 
# %%
real, imag = numpy.random.multivariate_normal(
    (0.6, 0.35), [[8e-3, 1e-3], [1e-3, 1e-3]], (10, 10)
).T
# plot = PhasorPlot(frequency=frequency, title = 'Point lying on the line between components')
# plot.plot(*phasor_from_lifetime(frequency, components_lifetimes), fmt= 'o-')
# plot.plot(real, imag, c='orange', fmt='.')
start = time.time()
fractions = fractional_intensities_from_phasor(real, imag, *phasor_from_lifetime(frequency, components_lifetimes))
end = time.time()
print(end-start)
#%%
print (fractions) 
import matplotlib.pyplot as plt
plt.figure()
plt.hist(fractions[1].flatten(), range=(0,1), bins=100)
plt.title('Histogram of 1D array')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
# %%
#OLD IMPLEMENTATION
fractions = numpy.array(
    [[1, 0], [0.25, 0.75], [0, 1]]
)
frequency = 80.0
components_lifetimes = [8.0, 1.0]
coordinates = phasor_from_lifetime(
        frequency, components_lifetimes, fractions
    )

plot = PhasorPlot(frequency=frequency, title = 'Point lying on the line between components')
plot.plot(*phasor_from_lifetime(frequency, components_lifetimes), fmt= 'o-')
plot.plot(coordinates[0][1], coordinates[1][1])
start = time.time()
fractions = fractional_intensities_from_phasor_old(coordinates[0][1], coordinates[1][1], *phasor_from_lifetime(frequency, components_lifetimes))
end = time.time()
print(end-start)
print (fractions) 
# %%
# real, imag = numpy.random.multivariate_normal(
#     (0.6, 0.35), [[8e-3, 1e-3], [1e-3, 1e-3]], (1000, 1000)
# ).T
# plot = PhasorPlot(frequency=frequency, title = 'Point lying on the line between components')
# plot.plot(*phasor_from_lifetime(frequency, components_lifetimes), fmt= 'o-')
# plot.plot(real, imag, c='orange')
start = time.time()
fractions = fractional_intensities_from_phasor_old(real, imag, *phasor_from_lifetime(frequency, components_lifetimes))
end = time.time()
print(end-start)
#%%
print (fractions) 
import matplotlib.pyplot as plt
plt.figure()
plt.hist(fractions[1].flatten(), range=(0,1), bins=100)
plt.title('Histogram of 1D array')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
# %%
