# distutils: language = c
# cython: language_level = 3
# cython: boundscheck = False
# cython: wraparound = True
# cython: cdivision = True
# cython: nonecheck = False

"""Cython implementation of low-level functions for the PhasorPy package."""

from libc.math cimport M_PI, NAN, fabs

ctypedef fused float_t:
    float
    double


def _phasor_from_lifetime(
    double[:, :, ::1] phasor,
    const double[::1] frequency,
    const double[:, ::1] lifetime,
    const double[:, ::1] fraction,
    const bint is_preexp,
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
    is_preexp:
        If true, fractions are pre-exponential amplitudes, else fractional
        intensities.

    """
    cdef:
        ssize_t nfreq = frequency.shape[0]  # number frequencies
        ssize_t ncomp = lifetime.shape[1]  # number lifetime components
        ssize_t ntau = lifetime.shape[0]  # number lifetimes
        ssize_t nfrac = fraction.shape[0]  # number fractions
        double twopi = 2e-3 * M_PI
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
                    if is_preexp:
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
                        if is_preexp:
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
                if not is_preexp:
                    sum = 0.0
                    for s in range(ncomp):
                        sum += fraction[0, s]
                for t in range(ntau):
                    if is_preexp:
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
                        if is_preexp:
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
                    if is_preexp:
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
                        if is_preexp:
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
