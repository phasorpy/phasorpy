"""cursors.

The ``phasorpy.cursors`` module provides functions to:

- use cursors to select region of interest in the phasor:

  - :py:func:`circle`
  - :py:func:`rectangular`


"""

# TODO add:
#   - phase cursors
#   - rectangular cursos

from __future__ import annotations

__all__ = [
    'circular_cursor',
    'rectangular_cursor',
    'phase_cursor',
]

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._typing import ArrayLike

import numpy

def circular_cursor(
        real: ArrayLike,
        imag: ArrayLike,
        center: ArrayLike, 
        *,
        radius: ArrayLike=[0.1, 0.1],
        components: int = 2):
    """Return labeled mask with components for each circle.

    Parameters
    ----------
    real : ndarray
        Real component of phasor coordinates along axis.
    imag : ndarray
        Imaginary component of phasor coordinates along axis.
    center : float
        Circle center.
    radius : ndarray
        Radius for each circle.
    components : int
        Amount of components, default 2. 

    Returns
    -------
    label : ndarray
        Labeled matrix for all components. 


    Raises
    ------
    ValueError
        real and imag must have the same dimensions.
    ValueError
        radius must be greater than zero.
    ValueError
        components must be at leat 1.

    Examples
    --------
    Calculate phasor coordinates of a phase-shifted sinusoidal waveform:

    >>> real = numpy.array([])

    """
    if real.shape != imag.shape:
        raise ValueError(f'{real.shape=} != {imag.shape=}')
    for i in range(len(radius)):
        if radius[i] < 0:
            raise ValueError(f'radius is < 0 for index {i}')
    if components < 1:
        raise ValueError(f'components is {components}, must be at least 1')
    if len(center) == components:
        label = numpy.zeros(real.shape)
        for i in range (components):
            condition = (real - center[i][0]) ** 2 + (imag - center[i][1]) ** 2 - radius[i] ** 2
            label = numpy.where(condition > 0, label, numpy.ones(label.shape) * (i + 1))
        return label
    else: 
        raise ValueError(f'center lenth array and components must be equal')


def rectangular(
        real: ArrayLike,
        imag: ArrayLike,
        
)
    




tests = False # Prueba gratis
if tests:
    
    caso_base = False
    if caso_base:
        real = numpy.array([-0.5, -0.5, 0.5, 0.5])
        imag = numpy.array([-0.5, 0.5, -0.5, 0.5])
        center = numpy.array([[-0.5, -0.5], [-0.5, 0.5], [0.5, -0.5], [0.5, 0.5]])
        radius = [0.1, 0.1, 0.1, 0.1]
        mask = circular_cursor(real, imag, center, radius=radius, components=4)

    prueba_img = False
    if prueba_img:
        real = numpy.random.rand(100, 100)
        imag = numpy.random.rand(100, 100)
        center = numpy.transpose([numpy.random.rand(10), numpy.random.rand(10)])
        radius = numpy.ones(10) * 0.1
        mask = circular_cursor(real, imag, center, radius=radius, components=10)

        import matplotlib.pyplot as plt
        plt.imshow(mask, cmap="gray")
        plt.show()

    testim = False
    if testim:
        import tifffile
        from phasorpy.phasor import phasor_from_signal

        im = tifffile.imread("SP_paramecium.lsm")
        _, real, imag = phasor_from_signal(im, axis=0)
        center = numpy.array([[0.5, -0.65], [0.17, -0.77], [0.44, -0.8], [0.7, -0.67]])
        radius = numpy.ones(4) * 0.15
        mask = circular_cursor(real, imag, center, radius=radius, components=4)

        plotty = True
        if plotty:
            import matplotlib.pyplot as plt
            plt.imshow(mask, cmap="nipy_spectral")
            plt.show()

        from phasorpy.plot import plot_phasor, PhasorPlot
        import matplotlib.pyplot as plt

        ax = plt.subplot(1, 1, 1)
        plot = PhasorPlot(ax=ax, allquadrants=True, title='Matplotlib axes')
        plot.hist2d(real, imag, cmap="RdYlGn_r")
        plt.show()
