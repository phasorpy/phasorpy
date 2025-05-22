"""Version information for PhasorPy and dependencies."""

from __future__ import annotations

__all__ = ['__version__', 'versions']

__version__ = '0.6.dev'


def versions(
    *, sep: str = '\n', dash: str = '-', verbose: bool = False
) -> str:
    """Return version information for PhasorPy and its dependencies.

    Parameters
    ----------
    sep : str, optional
        Separator between version items. Defaults to newline.
    dash : str, optional
        Separator between module name and version. Defaults to dash.
    verbose : bool, optional
        Include paths to Python interpreter and modules.

    Returns
    -------
    str
        Formatted string containing version information.
        Format: "<package><dash><version>[<space>(<path>)]<sep>"

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
