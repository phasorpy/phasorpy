"""Color palettes and manipulation."""

from __future__ import annotations

__all__ = ['float2int', 'wavelength2rgb', 'CATEGORICAL', 'SRGB_SPECTRUM']

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._typing import Any, ArrayLike, DTypeLike, NDArray

import numpy


def wavelength2rgb(
    wavelength: ArrayLike,
    /,
    dtype: DTypeLike | None = None,
) -> tuple[float, float, float] | NDArray[Any]:
    """Return approximate sRGB color components of visible wavelength(s).

    Wavelength values are clipped to 360..750, rounded, and used to index
    the :py:attr:`SRGB_SPECTRUM` palette.

    Parameters
    ----------
    wavelength : array_like
        Scalar or array of wavelength(s) to convert.
    dtype : data-type, optional
        Data-type of return value. The default is ``float32``.

    Returns
    -------
    ndarray or tuple
        Approximate sRGB color components of visible wavelength.
        Floating-point types are in range 0.0 to 1.0.
        Integer types are scaled to the dtype's maximum value.

    Examples
    --------
    >>> wavelength2rgb(517.2, 'uint8')
    (0, 191, 0)
    >>> wavelength2rgb([517, 566], 'uint8')
    array([[  0, 191,   0],
           [133, 190,   0]], dtype=uint8)

    """
    astuple = isinstance(wavelength, (float, int))
    indices = numpy.asarray(wavelength)
    indices = numpy.clip(indices, 360, 750)
    indices -= 360
    if indices.dtype.kind not in {'u', 'i'}:
        indices = numpy.round(indices).astype(numpy.uint32)
    rgb = SRGB_SPECTRUM.take(indices, axis=0)
    if dtype is not None:
        dtype = numpy.dtype(dtype)
        if dtype.kind in {'u', 'i'}:
            rgb = float2int(rgb)
        else:
            rgb = rgb.astype(dtype)
    if astuple:
        return tuple(rgb.tolist()[:3])  # type: ignore[index]
    return rgb


def float2int(
    rgb: ArrayLike,
    /,
    dtype: DTypeLike = numpy.uint8,
) -> NDArray[Any]:
    """Return normalized color components as integer type.

    Parameters
    ----------
    rgb : array_like
        Scalar or array of normalized floating-point color components.
    dtype : data-type, optional
        Data type of return value. The default is ``uint8``.

    Examples
    --------
    >>> float2int([0.0, 0.5, 1.0])
    array([  0, 128, 255], dtype=uint8)

    """
    dtype = numpy.dtype(dtype)
    if dtype.kind not in {'u', 'i'}:
        raise ValueError('not an integer dtype')
    arr: NDArray[Any] = numpy.asarray(rgb)
    if not arr.dtype.kind == 'f':
        raise ValueError('not a floating-point array')
    iinfo = numpy.iinfo(dtype)
    arr = numpy.round(arr * iinfo.max)
    numpy.clip(arr, iinfo.min, iinfo.max, out=arr)
    return arr.astype(dtype)


# fmt: off
CATEGORICAL: NDArray[numpy.float32] = numpy.array([
    [0.825397, 0.095238, 0.126984],
    [0.095239, 0.412698, 1.0],
    [0.000001, 0.539682, 0.0],
    [0.952381, 0.428571, 1.0],
    [0.444444, 0.0, 0.476191],
    [0.666667, 0.984127, 0.0],
    [0.0, 0.746032, 0.761905],
    [1.0, 0.634921, 0.206349],
    [0.365079, 0.238095, 0.015873],
    [0.031746, 0.0, 0.539683],
    [0.0, 0.365079, 0.365079],
    [0.603175, 0.492064, 0.507937],
    [0.634921, 0.68254, 1.0],
    [0.587302, 0.714286, 0.460317],
    [0.619048, 0.15873, 1.0],
    [0.301587, 0.0, 0.079365],
    [1.0, 0.68254, 0.746032],
    [0.809524, 0.0, 0.571429],
    [0.000001, 1.0, 0.714286],
    [0.0, 0.174603, 0.0],
    [0.619048, 0.460317, 0.0],
    [0.238095, 0.206349, 0.253968],
    [0.952381, 0.920635, 0.571429],
    [0.396825, 0.380952, 0.539683],
    [0.539683, 0.238095, 0.301587],
    [0.349206, 0.015873, 0.730159],
    [0.333333, 0.539683, 0.444444],
    [0.698413, 0.746032, 0.761905],
    [1.0, 0.365079, 0.507936],
    [0.111113, 0.777778, 0.0],
    [0.571429, 0.968254, 1.0],
    [0.174603, 0.52381, 0.650794],
    [0.222222, 0.365079, 0.15873],
    [0.920635, 0.809524, 1.0],
    [1.0, 0.365079, 0.0],
    [0.650794, 0.380952, 0.666667],
    [0.52381, 0.0, 0.0],
    [0.206349, 0.0, 0.349206],
    [0.0, 0.31746, 0.555556],
    [0.619048, 0.285714, 0.063492],
    [0.809524, 0.746032, 0.0],
    [0.0, 0.15873, 0.15873],
    [0.000001, 0.698413, 1.0],
    [0.793651, 0.650794, 0.52381],
    [0.746032, 0.603175, 0.761905],
    [0.174603, 0.126984, 0.047619],
    [0.460317, 0.396825, 0.269841],
    [0.507937, 0.47619, 0.873016],
    [0.0, 0.761905, 0.539682],
    [0.730159, 0.904762, 0.761905],
    [0.52381, 0.555556, 0.650794],
    [0.793651, 0.444444, 0.349206],
    [0.507937, 0.603175, 0.0],
    [0.174603, 0.0, 1.0],
    [0.825397, 0.015873, 0.968254],
    [1.0, 0.84127, 0.746032],
    [0.571429, 0.809524, 0.968254],
    [0.730159, 0.365079, 0.492063],
    [1.0, 0.253968, 0.761905],
    [0.746032, 0.523809, 1.0],
    [0.571429, 0.555556, 0.396825],
    [0.650794, 0.015874, 0.666667],
    [0.523809, 0.888889, 0.460317],
    [0.285714, 0.0, 0.238095],
], dtype=numpy.float32)
"""Categorical sRGB color palette inspired by C Glasbey.

The palette contains 64 maximally distinct colours.

Generated with the `glasbey <https://glasbey.readthedocs.io>`_ package::

    import glasbey; numpy.array(glasbey.create_palette(64, as_hex=False))

"""
# numpy.set_printoptions(6, suppress=True, threshold=512)
# fmt: on

# fmt: off
SRGB_SPECTRUM: NDArray[numpy.float32] = numpy.array([
    [0.000637, 0.0, 0.003852],
    [0.000715, 0.0, 0.004328],
    [0.000802, 0.0, 0.004863],
    [0.000899, 0.0, 0.005466],
    [0.001009, 0.0, 0.006144],
    [0.001131, 0.0, 0.006903],
    [0.001269, 0.0, 0.007758],
    [0.001425, 0.0, 0.008725],
    [0.0016, 0.0, 0.009811],
    [0.001794, 0.0, 0.011023],
    [0.00201, 0.0, 0.012369],
    [0.002247, 0.0, 0.013842],
    [0.00251, 0.0, 0.015481],
    [0.002811, 0.0, 0.017364],
    [0.003162, 0.0, 0.019563],
    [0.003575, 0.0, 0.022156],
    [0.004067, 0.0, 0.025266],
    [0.004636, 0.0, 0.028861],
    [0.005257, 0.0, 0.032785],
    [0.005906, 0.0, 0.036882],
    [0.006558, 0.0, 0.040984],
    [0.007197, 0.0, 0.044803],
    [0.007867, 0.0, 0.04859],
    [0.00863, 0.0, 0.052687],
    [0.009552, 0.0, 0.057365],
    [0.010697, 0.0, 0.062824],
    [0.012123, 0.0, 0.069172],
    [0.013823, 0.0, 0.076199],
    [0.015764, 0.0, 0.083636],
    [0.017911, 0.0, 0.091277],
    [0.020232, 0.0, 0.098967],
    [0.022687, 0.0, 0.106577],
    [0.025366, 0.0, 0.114362],
    [0.028425, 0.0, 0.122688],
    [0.032022, 0.0, 0.131827],
    [0.036313, 0.0, 0.141965],
    [0.04149, 0.0, 0.153349],
    [0.047101, 0.0, 0.165553],
    [0.052759, 0.0, 0.177868],
    [0.058222, 0.0, 0.189771],
    [0.063308, 0.0, 0.200873],
    [0.067943, 0.0, 0.211017],
    [0.07248, 0.0, 0.220985],
    [0.077342, 0.0, 0.231705],
    [0.082871, 0.0, 0.243915],
    [0.089323, 0.0, 0.258172],
    [0.096763, 0.0, 0.274629],
    [0.104948, 0.0, 0.292759],
    [0.113649, 0.0, 0.312047],
    [0.122685, 0.0, 0.332089],
    [0.131916, 0.0, 0.352572],
    [0.141263, 0.0, 0.373353],
    [0.150829, 0.0, 0.394689],
    [0.160741, 0.0, 0.416865],
    [0.171093, 0.0, 0.440085],
    [0.181952, 0.0, 0.464491],
    [0.193196, 0.0, 0.4898],
    [0.20469, 0.0, 0.515787],
    [0.216461, 0.0, 0.542603],
    [0.228524, 0.0, 0.570357],
    [0.240891, 0.0, 0.599121],
    [0.253574, 0.0, 0.628923],
    [0.266254, 0.0, 0.659006],
    [0.278515, 0.0, 0.688434],
    [0.290023, 0.0, 0.716465],
    [0.300501, 0.0, 0.742501],
    [0.309873, 0.0, 0.766385],
    [0.318252, 0.0, 0.788388],
    [0.325666, 0.0, 0.808609],
    [0.332132, 0.0, 0.827124],
    [0.337655, 0.0, 0.843989],
    [0.34221, 0.0, 0.859194],
    [0.345826, 0.0, 0.872791],
    [0.348574, 0.0, 0.884896],
    [0.350514, 0.0, 0.895609],
    [0.351698, 0.0, 0.905015],
    [0.35214, 0.0, 0.913161],
    [0.351844, 0.0, 0.920111],
    [0.350835, 0.0, 0.925957],
    [0.349134, 0.0, 0.930785],
    [0.346758, 0.0, 0.934672],
    [0.343711, 0.0, 0.937667],
    [0.339997, 0.0, 0.939848],
    [0.335622, 0.0, 0.941331],
    [0.330587, 0.0, 0.94223],
    [0.324891, 0.0, 0.942656],
    [0.318511, 0.0, 0.942657],
    [0.311425, 0.0, 0.942271],
    [0.303629, 0.0, 0.941593],
    [0.29511, 0.0, 0.940715],
    [0.285853, 0.0, 0.939732],
    [0.275831, 0.0, 0.938721],
    [0.264929, 0.0, 0.937617],
    [0.252956, 0.0, 0.936302],
    [0.239668, 0.0, 0.934652],
    [0.224736, 0.0, 0.932547],
    [0.207739, 0.0, 0.929913],
    [0.188181, 0.0, 0.926748],
    [0.16528, 0.0, 0.923032],
    [0.137584, 0.0, 0.918741],
    [0.101829, 0.0, 0.913852],
    [0.046511, 0.0, 0.908403],
    [0.0, 0.0, 0.902301],
    [0.0, 0.0, 0.895323],
    [0.0, 0.0, 0.88723],
    [0.0, 0.0, 0.877767],
    [0.0, 0.0, 0.866713],
    [0.0, 0.0, 0.854242],
    [0.0, 0.014235, 0.840692],
    [0.0, 0.086469, 0.826424],
    [0.0, 0.128801, 0.811832],
    [0.0, 0.16077, 0.797136],
    [0.0, 0.187393, 0.782253],
    [0.0, 0.210643, 0.767146],
    [0.0, 0.231538, 0.751774],
    [0.0, 0.250685, 0.736089],
    [0.0, 0.268464, 0.720127],
    [0.0, 0.285094, 0.703973],
    [0.0, 0.300717, 0.687653],
    [0.0, 0.315437, 0.671194],
    [0.0, 0.329335, 0.654629],
    [0.0, 0.342496, 0.637981],
    [0.0, 0.355056, 0.621294],
    [0.0, 0.367136, 0.604641],
    [0.0, 0.378836, 0.588105],
    [0.0, 0.390243, 0.571782],
    [0.0, 0.401404, 0.555762],
    [0.0, 0.412371, 0.540046],
    [0.0, 0.423219, 0.524612],
    [0.0, 0.434017, 0.509429],
    [0.0, 0.444821, 0.494462],
    [0.0, 0.455682, 0.47971],
    [0.0, 0.466602, 0.465195],
    [0.0, 0.477569, 0.450907],
    [0.0, 0.48857, 0.436835],
    [0.0, 0.499595, 0.422966],
    [0.0, 0.510616, 0.409277],
    [0.0, 0.52168, 0.395743],
    [0.0, 0.532886, 0.382342],
    [0.0, 0.544321, 0.369046],
    [0.0, 0.556066, 0.355823],
    [0.0, 0.568157, 0.342666],
    [0.0, 0.580484, 0.329491],
    [0.0, 0.592906, 0.316117],
    [0.0, 0.605293, 0.302332],
    [0.0, 0.617534, 0.287868],
    [0.0, 0.629559, 0.272478],
    [0.0, 0.641397, 0.256047],
    [0.0, 0.653081, 0.238435],
    [0.0, 0.664643, 0.219442],
    [0.0, 0.676109, 0.198778],
    [0.0, 0.687475, 0.17586],
    [0.0, 0.698664, 0.149763],
    [0.0, 0.709604, 0.118887],
    [0.0, 0.720225, 0.07922],
    [0.0, 0.730466, 0.016354],
    [0.0, 0.740314, 0.0],
    [0.0, 0.749744, 0.0],
    [0.0, 0.758674, 0.0],
    [0.0, 0.767026, 0.0],
    [0.0, 0.774726, 0.0],
    [0.0, 0.781724, 0.0],
    [0.0, 0.788079, 0.0],
    [0.0, 0.793878, 0.0],
    [0.0, 0.799205, 0.0],
    [0.0, 0.804139, 0.0],
    [0.0, 0.808714, 0.0],
    [0.0, 0.812915, 0.0],
    [0.0, 0.816742, 0.0],
    [0.0, 0.820196, 0.0],
    [0.0, 0.823277, 0.0],
    [0.0, 0.82599, 0.0],
    [0.0, 0.828356, 0.0],
    [0.0, 0.830391, 0.0],
    [0.0, 0.832112, 0.0],
    [0.0, 0.833534, 0.0],
    [0.0, 0.83467, 0.0],
    [0.0, 0.835521, 0.0],
    [0.0, 0.836088, 0.0],
    [0.0, 0.836369, 0.0],
    [0.0, 0.836364, 0.0],
    [0.0, 0.836076, 0.0],
    [0.0, 0.835512, 0.0],
    [0.0, 0.834675, 0.0],
    [0.0, 0.833569, 0.0],
    [0.0, 0.832197, 0.0],
    [0.0, 0.830559, 0.0],
    [0.0, 0.828659, 0.0],
    [0.0, 0.82651, 0.0],
    [0.0, 0.824121, 0.0],
    [0.0, 0.821506, 0.0],
    [0.0, 0.818651, 0.0],
    [0.0, 0.815559, 0.0],
    [0.0, 0.812229, 0.0],
    [0.0, 0.808655, 0.0],
    [0.0, 0.804833, 0.0],
    [0.0, 0.800754, 0.0],
    [0.117805, 0.79641, 0.0],
    [0.209152, 0.791793, 0.0],
    [0.270534, 0.786894, 0.0],
    [0.319691, 0.781701, 0.0],
    [0.361821, 0.776197, 0.0],
    [0.399231, 0.770399, 0.0],
    [0.433215, 0.764307, 0.0],
    [0.464567, 0.757925, 0.0],
    [0.493817, 0.751251, 0.0],
    [0.52133, 0.744288, 0.0],
    [0.547363, 0.737029, 0.0],
    [0.572108, 0.729468, 0.0],
    [0.595715, 0.721599, 0.0],
    [0.618302, 0.713415, 0.0],
    [0.639961, 0.704908, 0.0],
    [0.660766, 0.69608, 0.0],
    [0.680781, 0.686929, 0.0],
    [0.700054, 0.677456, 0.0],
    [0.718629, 0.667662, 0.0],
    [0.736543, 0.657544, 0.0],
    [0.753817, 0.647097, 0.0],
    [0.770461, 0.636315, 0.0],
    [0.786485, 0.625189, 0.0],
    [0.801895, 0.613711, 0.0],
    [0.816702, 0.60187, 0.0],
    [0.830921, 0.589667, 0.0],
    [0.844563, 0.577104, 0.0],
    [0.857636, 0.564184, 0.0],
    [0.870146, 0.550908, 0.0],
    [0.882092, 0.537277, 0.0],
    [0.893477, 0.523283, 0.0],
    [0.904318, 0.508919, 0.0],
    [0.914626, 0.494177, 0.0],
    [0.924413, 0.479048, 0.0],
    [0.9337, 0.463509, 0.0],
    [0.942459, 0.447555, 0.0],
    [0.950635, 0.431203, 0.0],
    [0.958169, 0.414472, 0.0],
    [0.965008, 0.397387, 0.0],
    [0.971112, 0.379967, 0.0],
    [0.976523, 0.362161, 0.0],
    [0.981305, 0.343869, 0.0],
    [0.985522, 0.32497, 0.0],
    [0.989232, 0.305304, 0.0],
    [0.992472, 0.284689, 0.0],
    [0.995205, 0.262981, 0.0],
    [0.997376, 0.239996, 0.0],
    [0.998932, 0.215465, 0.0],
    [0.999817, 0.188965, 0.0],
    [1.0, 0.159702, 0.0],
    [0.99952, 0.125912, 0.0],
    [0.998423, 0.083139, 0.0],
    [0.996755, 0.014479, 0.0],
    [0.99456, 0.0, 0.0],
    [0.991856, 0.0, 0.0],
    [0.988629, 0.0, 0.0],
    [0.984875, 0.0, 0.0],
    [0.980588, 0.0, 0.0],
    [0.97576, 0.0, 0.0],
    [0.970407, 0.0, 0.0],
    [0.964527, 0.0, 0.0],
    [0.95809, 0.0, 0.0],
    [0.951063, 0.0, 0.0],
    [0.94341, 0.0, 0.0],
    [0.935112, 0.0, 0.0],
    [0.926214, 0.0, 0.0],
    [0.916774, 0.0, 0.0],
    [0.906853, 0.0, 0.0],
    [0.896515, 0.0, 0.0],
    [0.885776, 0.0, 0.0],
    [0.874671, 0.0, 0.0],
    [0.863295, 0.0, 0.0],
    [0.851747, 0.0, 0.0],
    [0.840136, 0.0, 0.0],
    [0.828552, 0.0, 0.0],
    [0.816975, 0.0, 0.0],
    [0.805353, 0.0, 0.0],
    [0.793632, 0.0, 0.0],
    [0.781751, 0.0, 0.0],
    [0.769698, 0.0, 0.0],
    [0.757501, 0.0, 0.0],
    [0.745159, 0.0, 0.0],
    [0.732668, 0.0, 0.0],
    [0.720026, 0.0, 0.0],
    [0.707228, 0.0, 0.0],
    [0.694283, 0.0, 0.0],
    [0.681211, 0.0, 0.0],
    [0.668035, 0.0, 0.0],
    [0.654777, 0.0, 0.0],
    [0.641448, 0.0, 0.0],
    [0.628062, 0.0, 0.0],
    [0.614654, 0.0, 0.0],
    [0.601267, 0.0, 0.0],
    [0.587945, 0.0, 0.0],
    [0.574731, 0.0, 0.0],
    [0.561623, 0.0, 0.0],
    [0.548602, 0.0, 0.0],
    [0.53565, 0.0, 0.0],
    [0.522744, 0.0, 0.0],
    [0.509883, 0.0, 0.0],
    [0.497083, 0.0, 0.0],
    [0.484347, 0.0, 0.0],
    [0.471676, 0.0, 0.0],
    [0.459075, 0.0, 0.0],
    [0.446545, 0.0, 0.0],
    [0.434099, 0.0, 0.0],
    [0.421754, 0.0, 0.0],
    [0.409528, 0.0, 0.0],
    [0.397442, 0.0, 0.0],
    [0.385505, 0.0, 0.0],
    [0.373738, 0.0, 0.0],
    [0.362192, 0.0, 0.0],
    [0.350926, 0.0, 0.0],
    [0.340006, 0.0, 0.0],
    [0.329485, 0.0, 0.0],
    [0.319357, 0.0, 0.0],
    [0.309606, 0.0, 0.0],
    [0.300207, 0.0, 0.0],
    [0.291134, 0.0, 0.0],
    [0.282402, 0.0, 0.0],
    [0.273993, 0.0, 0.0],
    [0.265809, 0.0, 0.0],
    [0.257734, 0.0, 0.0],
    [0.249634, 0.0, 0.0],
    [0.241395, 0.0, 0.0],
    [0.233045, 0.0, 0.0],
    [0.224664, 0.0, 0.0],
    [0.216342, 0.0, 0.0],
    [0.208187, 0.0, 0.0],
    [0.200246, 0.0, 0.0],
    [0.192494, 0.0, 0.0],
    [0.184944, 0.0, 0.0],
    [0.177614, 0.0, 0.0],
    [0.170521, 0.0, 0.0],
    [0.163694, 0.0, 0.0],
    [0.157146, 0.0, 0.0],
    [0.150876, 0.0, 0.0],
    [0.144878, 0.0, 0.0],
    [0.139142, 0.0, 0.0],
    [0.133678, 0.0, 0.0],
    [0.128482, 0.0, 0.0],
    [0.123512, 0.0, 0.0],
    [0.118716, 0.0, 0.0],
    [0.114033, 0.0, 0.0],
    [0.10942, 0.0, 0.0],
    [0.104882, 0.0, 0.0],
    [0.100436, 0.0, 0.0],
    [0.096101, 0.0, 0.0],
    [0.0919, 0.0, 0.0],
    [0.087833, 0.0, 0.0],
    [0.083882, 0.0, 0.0],
    [0.080041, 0.0, 0.0],
    [0.076302, 0.0, 0.0],
    [0.072655, 0.0, 0.0],
    [0.06909, 0.0, 0.0],
    [0.065607, 0.0, 0.0],
    [0.062208, 0.0, 0.0],
    [0.058892, 0.0, 0.0],
    [0.055659, 0.0, 0.0],
    [0.052505, 0.0, 0.0],
    [0.049427, 0.0, 0.0],
    [0.046429, 0.0, 0.0],
    [0.043516, 0.0, 0.0],
    [0.04069, 0.0, 0.0],
    [0.037958, 0.0, 0.0],
    [0.035413, 0.0, 0.0],
    [0.033043, 0.0, 0.0],
    [0.030831, 0.0, 0.0],
    [0.028762, 0.0, 0.0],
    [0.026822, 0.0, 0.0],
    [0.025004, 0.0, 0.0],
    [0.023301, 0.0, 0.0],
    [0.021706, 0.0, 0.0],
    [0.020211, 0.0, 0.0],
    [0.018809, 0.0, 0.0],
    [0.017493, 0.0, 0.0],
    [0.01626, 0.0, 0.0],
    [0.015109, 0.0, 0.0],
    [0.014035, 0.0, 0.0],
    [0.013036, 0.0, 0.0],
    [0.012105, 0.0, 0.0],
    [0.011239, 0.0, 0.0],
    [0.010434, 0.0, 0.0],
    [0.009686, 0.0, 0.0],
    [0.00899, 0.0, 0.0],
    [0.008344, 0.0, 0.0],
    [0.007746, 0.0, 0.0],
    [0.007192, 0.0, 0.0],
    [0.006681, 0.0, 0.0],
    [0.00621, 0.0, 0.0],
    [0.005776, 0.0, 0.0],
    [0.005375, 0.0, 0.0],
    [0.005006, 0.0, 0.0],
    [0.004664, 0.0, 0.0],
], dtype=numpy.float32)
"""sRGB color components for visible light wavelengths 360-750 nm.

Based on the CIE 1931 2° Standard Observer.

Generated with the `colour <https://colour.readthedocs.io>`_ package::

    import colour; colour.plotting.plot_visible_spectrum()

"""
# fmt: on
