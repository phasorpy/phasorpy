#%%
def func(a,b):
    return a

import inspect

func_c_params = inspect.signature(func).parameters
print(func_c_params)
# %%
if 'a' in inspect.signature(func).parameters:
    print('a')
# %%
