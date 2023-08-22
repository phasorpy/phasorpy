"""Version information."""

from __future__ import annotations

__version__ = '0.1.dev'


def versions(*, sep: str = '\n') -> str:
    """Return versions of installed packages that phasorpy depends on.

    Parameters
    ----------
    sep : str, optional
        Separator between version items. The default is a newline character.

    Example
    -------
    >>> print(versions())
    Python 3...
    phasorpy 0...
    numpy 1...
    ...

    """
    import os
    import sys

    try:
        path = os.path.dirname(__file__)
    except NameError:
        path = ''

    version_strings = [
        f'Python {sys.version} ({sys.executable})',
        f'phasorpy {__version__} ({path})',
    ]

    for module in (
        'numpy',
        'matplotlib',
        'click',
        'tifffile',
        # 'scipy',
        # 'skimage',
        # 'sklearn',
        # 'aicsimageio',
        # 'lfdfiles',
        # 'sdtfile',
    ):
        try:
            __import__(module)
        except ModuleNotFoundError:
            version_strings.append(f'{module.lower()} N/A')
            continue
        lib = sys.modules[module]
        version_strings.append(
            f"{module.lower()} {getattr(lib, '__version__', 'unknown')}"
        )
    return sep.join(version_strings)
