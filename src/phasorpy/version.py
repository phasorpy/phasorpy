"""Version information."""

from __future__ import annotations

__version__ = '0.4'


def versions(
    *, sep: str = '\n', dash: str = '-', verbose: bool = False
) -> str:
    """Return versions of installed packages that phasorpy depends on.

    Parameters
    ----------
    sep : str, optional
        Separator between version items. The default is a newline character.
    dash : str, optional
        Separator between module name and version.
    verbose : bool, optional
        Include paths to Python interpreter and modules.

    Example
    -------
    >>> print(versions())
    Python-3...
    phasorpy-0...
    numpy-...
    ...

    """
    import os
    import sys

    if verbose:
        version_strings = [f'Python{dash}{sys.version}  ({sys.executable})']
    else:
        version_strings = [f'Python{dash}{sys.version.split()[0]}']

    for module in (
        'phasorpy',
        'numpy',
        'tifffile',
        'imagecodecs',
        'lfdfiles',
        'sdtfile',
        'ptufile',
        'liffile',
        'matplotlib',
        'scipy',
        'skimage',
        'sklearn',
        'pandas',
        'xarray',
        'click',
        'pooch',
    ):
        try:
            __import__(module)
        except ModuleNotFoundError:
            version_strings.append(f'{module.lower()}{dash}n/a')
            continue
        lib = sys.modules[module]
        ver = f"{module.lower()}{dash}{getattr(lib, '__version__', 'unknown')}"
        if verbose:
            try:
                path = getattr(lib, '__file__')
            except NameError:
                pass
            else:
                ver += f'  ({os.path.dirname(path)})'
        version_strings.append(ver)
    return sep.join(version_strings)
