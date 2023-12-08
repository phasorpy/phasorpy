"""
============================================
Phasor functions, module: "phasorpy.phasor"
============================================

This module provides a set of function to 
compute phasor transform based on the fast 
fourier transform algorithm from numpy.fft. 
There are general tools common to FLIM and 
HSI data, that helps when doing phasor 
analysis. 

"""

from __future__ import annotations

__all__ = ['phasor']

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._typing import Any, NDArray

import numpy as np


def phasor(
    imstack: NDArray[Any], harmonic: int = 1
) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
    """
    Computes the phasor transform using the
    fast fourier transform algorithm from numpy.fft

    Parameters
    ----------
    imstack : array_like
        description: Input array, which can be complex.
            dims = ZXY. Where z is along axis 0.
            Generally an image stack.
    harmonic : int, optional
        description:, by default 1, correspond to the nth
            harmonic.
    Returns
    -------
        dc: nd array or tuple.
            Contains the average of the 0-axis
        g: nd array or tuple.
            Contains the real part of the fft transform.
        s: nd array or tuple.
            Contains the imaginary part of the fft transform.

    """
    if imstack.any():
        data = np.fft.fft(imstack, axis=0, norm='ortho')
        dc = data[0].real
        dc = np.where(
            dc != 0, dc, np.NaN
        )  # Set NaN where there is a 0 division
        g = data[harmonic].real
        g /= dc
        s = data[harmonic].imag
        s /= dc
    else:
        raise ValueError('imstack value is empty')
    return dc, g, s
