#%%
from phasorpy.cursors import mask_from_polar_cursor
import numpy
phase = numpy.array([[337, 306, 227], [21, 231, 235], [244, 328, 116]])
mod = [[0.22, 0.40, 0.81], [0.33, 0.43, 0.36], [0.015, 0.82, 0.58]]
phase_range = numpy.array([[0, 270],[0, 270],[0, 270], [0, 270]])
mod_range = numpy.array([[0, 0.5],[0, 0.5], [0, 0.5], [0, 0.5]])
print(phase.shape)
print(phase_range.shape)
result = mask_from_polar_cursor(phase, mod, phase_range, mod_range)

print(result.shape)
print(result)

# %%
