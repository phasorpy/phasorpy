#%%
from phasorpy.cursors import mask_from_circular_cursor
import numpy as np
centers = np.array([[1,1,1],[0,0,0],[2,2,2]])

result = mask_from_circular_cursor([0,1], [0,1], centers , 0.5)
print(centers.shape)
print(result.shape)
print(result)

# %%
