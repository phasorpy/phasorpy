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
