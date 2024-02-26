# distutils: language = c
# cython: language_level = 3
# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: nonecheck = False

"""Cython implementation of low-level functions for the PhasorPy package."""

from cython.parallel import parallel, prange

from libc.math cimport M_PI, NAN, fabs
from libc.stdint cimport (
    int8_t,
    int16_t,
    int32_t,
    int64_t,
    uint8_t,
    uint16_t,
    uint32_t,
    uint64_t,
)

ctypedef fused float_t:
    float
    double

ctypedef fused signal_t:
    uint8_t
    uint16_t
    uint32_t
    uint64_t
    int8_t
    int16_t
    int32_t
    int64_t
    float
    double


def _phasor_from_signal(
    float_t[:, :, ::1] phasor,
    const signal_t[:, :, ::1] signal,
    const double[:, :, ::1] sincos,
    const int num_threads
):
    """Return phasor coordinates from signal along middle axis.

    This implementation requires contiguous input arrays.

    TODO: use Numpy iterator API?
    https://numpy.org/devdocs/reference/c-api/iterator.html

    Parameters
    ----------
    phasor:
        Writable buffer of three-dimensions where calculated phasor
        coordinates are stored:
            0. mean, real, and imaginary components
            1. lower dimensions flat
            2. upper dimensions flat
    signal:
        Buffer of three-dimensions containing signal:
            0. lower dimensions flat
            1. dimension over which to compute FFT, number samples
            2. upper dimensions flat
    sincos:
        Buffer of two dimensions containing sin and cos terms to be multiplied
        with signal:
            0. number harmonics
            1. number samples
            2. cos and sin
    num_threads:
        Number of OpenMP threads to use for parallelization.

    """
    cdef:
        float_t[:, ::1] mean
        float_t[:, :, ::1] real, imag
        ssize_t samples = signal.shape[1]
        ssize_t harmonics = sincos.shape[0]
        ssize_t i, j, k, h
        double dc, re, im, sample

    if (
        samples < 3
        or harmonics > samples // 2
        or phasor.shape[0] != harmonics * 2 + 1
        or phasor.shape[1] != signal.shape[0]
        or phasor.shape[2] != signal.shape[2]
    ):
        raise ValueError('invalid shape of phasor or signal')
    if sincos.shape[1] != samples or sincos.shape[2] != 2:
        raise ValueError('invalid shape of sincos')

    mean = phasor[0]
    real = phasor[1 : 1 + harmonics]
    imag = phasor[1 + harmonics : 1 + harmonics * 2]

    if num_threads > 1 and signal.shape[0] >= num_threads:
        # parallelize outer dimensions
        with nogil, parallel(num_threads=num_threads):
            for i in prange(signal.shape[0]):
                for h in range(harmonics):
                    for j in range(signal.shape[2]):
                        dc = 0.0
                        re = 0.0
                        im = 0.0
                        for k in range(samples):
                            sample = <double> signal[i, k, j]
                            dc = dc + sample
                            re = re + sample * sincos[h, k, 0]
                            im = im + sample * sincos[h, k, 1]
                        if dc > 1e-16:
                            re = re / dc
                            im = im / dc
                            dc = dc /samples
                        else:
                            dc = 0.0
                            re = 0.0  # inf?
                            im = 0.0  # inf?
                        if h == 0:
                            mean[i, j] = <float_t> dc
                        real[h, i, j] = <float_t> re
                        imag[h, i, j] = <float_t> im

    elif num_threads > 1 and signal.shape[2] >= num_threads:
        # parallelize inner dimensions
        # TODO: do not use when not built with OpenMP
        with nogil, parallel(num_threads=num_threads):
            for j in prange(signal.shape[2]):
                for h in range(harmonics):
                    for i in range(signal.shape[0]):
                        dc = 0.0
                        re = 0.0
                        im = 0.0
                        for k in range(samples):
                            sample = <double> signal[i, k, j]
                            dc = dc + sample
                            re = re + sample * sincos[h, k, 0]
                            im = im + sample * sincos[h, k, 1]
                        if dc > 1e-16:
                            re = re / dc
                            im = im / dc
                            dc = dc /samples
                        else:
                            dc = 0.0
                            re = 0.0  # inf?
                            im = 0.0  # inf?
                        if h == 0:
                            mean[i, j] = <float_t> dc
                        real[h, i, j] = <float_t> re
                        imag[h, i, j] = <float_t> im

    else:
        # do not parallelize
        with nogil:
            for h in range(harmonics):
                # TODO: move harmonics to an inner loop?
                for i in range(signal.shape[0]):
                    for j in range(signal.shape[2]):
                            dc = 0.0
                            re = 0.0
                            im = 0.0
                            for k in range(samples):
                                sample = <double> signal[i, k, j]
                                dc += sample
                                re += sample * sincos[h, k, 0]
                                im += sample * sincos[h, k, 1]
                            if dc > 1e-16:
                                re /= dc
                                im /= dc
                                dc /= samples
                            else:
                                dc = 0.0
                                re = 0.0  # inf?
                                im = 0.0  # inf?
                            if h == 0:
                                mean[i, j] = <float_t> dc
                            real[h, i, j] = <float_t> re
                            imag[h, i, j] = <float_t> im


def _phasor_from_lifetime(
    double[:, :, ::1] phasor,
    const double[::1] frequency,
    const double[:, ::1] lifetime,
    const double[:, ::1] fraction,
    const double unit_conversion,
    const bint preexponential,
):
    """Calculate phasor coordinates from lifetime components.

    Parameters
    ----------
    phasor:
        Writable buffer of three-dimensions where calculated phasor
        coordinates are stored:
            0. real and imaginary components
            1. frequencies
            2. lifetimes or fractions
    frequency:
        One-dimensional sequence of laser-pulse or modulation frequencies.
    lifetime:
        Buffer of two dimensions:
            0. lifetimes
            1. components of lifetimes
    fraction:
        Buffer of two dimensions:
            0. fractions
            1. fractions of lifetime components
    unit_conversion:
        Product of `frequency` and `lifetime` units' prefix factors.
        1e-3 for MHz and ns. 1.0 for Hz and s.
    preexponential:
        If true, fractions are pre-exponential amplitudes, else fractional
        intensities.

    """
    cdef:
        ssize_t nfreq = frequency.shape[0]  # number frequencies
        ssize_t ncomp = lifetime.shape[1]  # number lifetime components
        ssize_t ntau = lifetime.shape[0]  # number lifetimes
        ssize_t nfrac = fraction.shape[0]  # number fractions
        double twopi = 2 * M_PI * unit_conversion
        double nan = NAN  # float('NaN')
        double freq, tau, frac, sum, re, im, gs
        ssize_t f, t, s

    if phasor.shape[0] != 2 or phasor.shape[1] != nfreq:
        raise ValueError(
            f'invalid {phasor.shape=!r} != (2, {nfreq}, -1))'
        )
    if fraction.shape[1] != ncomp:
        raise ValueError(f'{lifetime.shape[1]=} != {fraction.shape[1]=}')

    if nfreq == 1 and ntau == 1 and nfrac == 1 and ncomp == 1:
        # scalar
        tau = lifetime[0, 0] * frequency[0] * twopi  # omega_tau
        gs = 1.0 / (1.0 + tau * tau)
        phasor[0, 0, 0] = gs
        phasor[1, 0, 0] = gs * tau
        return

    if ntau == nfrac:
        # fractions specified for all lifetime components
        if phasor.shape[2] != ntau:
            raise ValueError(f'{phasor.shape[2]=} != {ntau}')
        with nogil:
            for f in range(nfreq):
                freq = frequency[f] * twopi  # omega
                for t in range(ntau):
                    re = 0.0
                    im = 0.0
                    sum = 0.0
                    if preexponential:
                        for s in range(ncomp):
                            sum += fraction[t, s] * lifetime[t, s]  # Fdc
                    else:
                        for s in range(ncomp):
                            sum += fraction[t, s]
                    if fabs(sum) < 1e-16:
                        phasor[0, f, t] = nan
                        phasor[1, f, t] = nan
                        continue
                    for s in range(ncomp):
                        tau = lifetime[t, s]
                        frac = fraction[t, s] / sum
                        if preexponential:
                            frac *= tau
                        tau *= freq  # omega_tau
                        gs = frac / (1.0 + tau * tau)
                        re += gs
                        im += gs * tau
                    phasor[0, f, t] = re
                    phasor[1, f, t] = im
        return

    if ntau > 1 and nfrac == 1:
        # varying lifetime components, same fractions
        if phasor.shape[2] != ntau:
            raise ValueError(f'{phasor.shape[2]=} != {ntau}')
        with nogil:
            for f in range(nfreq):
                freq = frequency[f] * twopi  # omega
                if not preexponential:
                    sum = 0.0
                    for s in range(ncomp):
                        sum += fraction[0, s]
                for t in range(ntau):
                    if preexponential:
                        sum = 0.0
                        for s in range(ncomp):
                            sum += fraction[0, s] * lifetime[t, s]  # Fdc
                    if fabs(sum) < 1e-16:
                        phasor[0, f, t] = nan
                        phasor[1, f, t] = nan
                        continue
                    re = 0.0
                    im = 0.0
                    for s in range(ncomp):
                        tau = lifetime[t, s]
                        frac = fraction[0, s] / sum
                        if preexponential:
                            frac *= tau
                        tau *= freq  # omega_tau
                        gs = frac / (1.0 + tau * tau)
                        re += gs
                        im += gs * tau
                    phasor[0, f, t] = re
                    phasor[1, f, t] = im
        return

    if ntau == 1 and nfrac > 1:
        # same lifetime components, varying fractions
        if phasor.shape[2] != nfrac:
            raise ValueError(f'{phasor.shape[2]=} != {nfrac}')
        with nogil:
            for f in range(nfreq):
                freq = frequency[f] * twopi  # omega
                for t in range(nfrac):
                    re = 0.0
                    im = 0.0
                    sum = 0.0
                    if preexponential:
                        for s in range(ncomp):
                            sum += fraction[t, s] * lifetime[0, s]  # Fdc
                    else:
                        for s in range(ncomp):
                            sum += fraction[t, s]
                    if fabs(sum) < 1e-16:
                        phasor[0, f, t] = nan
                        phasor[1, f, t] = nan
                        continue
                    for s in range(ncomp):
                        tau = lifetime[0, s]
                        frac = fraction[t, s] / sum
                        if preexponential:
                            frac *= tau
                        tau *= freq  # omega_tau
                        gs = frac / (1.0 + tau * tau)
                        re += gs
                        im += gs * tau
                    phasor[0, f, t] = re
                    phasor[1, f, t] = im
        return

    raise ValueError(
        f'{lifetime.shape[0]=} and {fraction.shape[0]=} do not match'
    )
