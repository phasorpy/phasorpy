# distutils: language = c
# cython: language_level = 3
# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: nonecheck = False
# cython: freethreading_compatible = True

"""Cython implementation of low-level functions for the PhasorPy library."""

cimport cython

from cython.parallel import parallel, prange

from libc.math cimport (
    INFINITY,
    M_PI,
    NAN,
    atan,
    atan2,
    copysign,
    cos,
    exp,
    fabs,
    floor,
    hypot,
    isnan,
    sin,
    sqrt,
    tan,
)
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

ctypedef fused uint_t:
    uint8_t
    uint16_t
    uint32_t
    uint64_t

ctypedef fused int_t:
    int8_t
    int16_t
    int32_t
    int64_t

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

from libc.stdlib cimport free, malloc


def _phasor_from_signal(
    float_t[:, :, ::1] phasor,
    const signal_t[:, :, ::1] signal,
    const double[:, :, ::1] sincos,
    const bint normalize,
    const int num_threads,
):
    """Return phasor coordinates from signal along middle axis.

    Parameters
    ----------
    phasor : 3D memoryview of float32 or float64
        Writable buffer of three dimensions where calculated phasor
        coordinates are stored:

        0. mean, real, and imaginary components
        1. lower dimensions flat
        2. upper dimensions flat

    signal : 3D memoryview of float32 or float64
        Buffer of three dimensions containing signal:

        0. lower dimensions flat
        1. dimension over which to compute FFT, number samples
        2. upper dimensions flat

    sincos : 3D memoryview of float64
        Buffer of three dimensions containing sine and cosine terms to be
        multiplied with signal:

        0. number harmonics
        1. number samples
        2. cos and sin

    normalize : bool
        Normalize phasor coordinates.
    num_threads : int
        Number of OpenMP threads to use for parallelization.

    Notes
    -----
    This implementation requires contiguous input arrays.

    """
    cdef:
        float_t[:, ::1] mean
        float_t[:, :, ::1] real, imag
        ssize_t samples = signal.shape[1]
        ssize_t harmonics = sincos.shape[0]
        ssize_t i, j, k, h
        double dc, re, im, sample

    # TODO: use Numpy iterator API?
    # https://numpy.org/devdocs/reference/c-api/iterator.html

    if (
        samples < 2
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
                        if normalize:
                            if dc != 0.0:
                                # includes isnan(dc)
                                re = re / dc
                                im = im / dc
                                dc = dc / samples
                            else:
                                # dc = 0.0
                                re = NAN if re == 0.0 else re * INFINITY
                                im = NAN if im == 0.0 else im * INFINITY
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
                        if normalize:
                            if dc != 0.0:
                                # includes isnan(dc)
                                re = re / dc
                                im = im / dc
                                dc = dc / samples
                            else:
                                # dc = 0.0
                                re = NAN if re == 0.0 else re * INFINITY
                                im = NAN if im == 0.0 else im * INFINITY
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
                        if normalize:
                            if dc != 0.0:
                                # includes isnan(dc)
                                re /= dc
                                im /= dc
                                dc = dc / samples
                            else:
                                # dc = 0.0
                                re = NAN if re == 0.0 else re * INFINITY
                                im = NAN if im == 0.0 else im * INFINITY
                        if h == 0:
                            mean[i, j] = <float_t> dc
                        real[h, i, j] = <float_t> re
                        imag[h, i, j] = <float_t> im


def _phasor_from_lifetime(
    float_t[:, :, ::1] phasor,
    const double[::1] frequency,
    const double[:, ::1] lifetime,
    const double[:, ::1] fraction,
    const double unit_conversion,
    const bint preexponential,
):
    """Calculate phasor coordinates from lifetime components.

    Parameters
    ----------
    phasor : 3D memoryview of float32 or float64
        Writable buffer of three dimensions where calculated phasor
        coordinates are stored:

        0. real and imaginary components
        1. frequencies
        2. lifetimes or fractions

    frequency : 2D memoryview of float64
        One-dimensional sequence of laser-pulse or modulation frequencies.
    lifetime : 2D memoryview of float64
        Buffer of two dimensions:

        0. lifetimes
        1. components of lifetimes

    fraction : 2D memoryview of float64
        Buffer of two dimensions:

        0. fractions
        1. fractions of lifetime components

    unit_conversion : float
        Product of `frequency` and `lifetime` units' prefix factors.
        1e-3 for MHz and ns. 1.0 for Hz and s.
    preexponential : bool
        If true, fractions are pre-exponential amplitudes, else fractional
        intensities.

    """
    cdef:
        ssize_t nfreq = frequency.shape[0]  # number frequencies
        ssize_t ncomp = lifetime.shape[1]  # number lifetime components
        ssize_t ntau = lifetime.shape[0]  # number lifetimes
        ssize_t nfrac = fraction.shape[0]  # number fractions
        double twopi = 2.0 * M_PI * unit_conversion
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
        phasor[0, 0, 0] = <float_t> gs
        phasor[1, 0, 0] = <float_t> (gs * tau)
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
                    if fabs(sum) < 1e-15:
                        phasor[0, f, t] = <float_t> NAN
                        phasor[1, f, t] = <float_t> NAN
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
                    phasor[0, f, t] = <float_t> re
                    phasor[1, f, t] = <float_t> im
        return

    if ntau > 1 and nfrac == 1:
        # varying lifetime components, same fractions
        if phasor.shape[2] != ntau:
            raise ValueError(f'{phasor.shape[2]=} != {ntau}')
        with nogil:
            for f in range(nfreq):
                freq = frequency[f] * twopi  # omega
                sum = 0.0
                if not preexponential:
                    for s in range(ncomp):
                        sum += fraction[0, s]
                for t in range(ntau):
                    if preexponential:
                        sum = 0.0
                        for s in range(ncomp):
                            sum += fraction[0, s] * lifetime[t, s]  # Fdc
                    if fabs(sum) < 1e-15:
                        phasor[0, f, t] = <float_t> NAN
                        phasor[1, f, t] = <float_t> NAN
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
                    phasor[0, f, t] = <float_t> re
                    phasor[1, f, t] = <float_t> im
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
                    if fabs(sum) < 1e-15:
                        phasor[0, f, t] = <float_t> NAN
                        phasor[1, f, t] = <float_t> NAN
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
                    phasor[0, f, t] = <float_t> re
                    phasor[1, f, t] = <float_t> im
        return

    raise ValueError(
        f'{lifetime.shape[0]=} and {fraction.shape[0]=} do not match'
    )


def _gaussian_signal(
    float_t[::1] signal,
    const double mean,
    const double stdev,
):
    """Return normal distribution, wrapped around at borders.

    Parameters
    ----------
    signal : memoryview of float32 or float64
        Writable buffer where calculated signal samples are stored.
    mean : float
        Mean of normal distribution.
    stdev : float
        Standard deviation of normal distribution.

    """
    cdef:
        ssize_t samples = signal.shape[0]
        ssize_t folds = 1  # TODO: calculate from stddev and samples
        ssize_t i
        double t, c

    if stdev <= 0.0 or samples < 1:
        return

    with nogil:
        c = 1.0 / sqrt(2.0 * M_PI) * stdev

        for i in range(-folds * samples, (folds + 1) * samples):
            t = (<double> i - mean) / stdev
            t *= t
            t = c * exp(-t / 2.0)
            # i %= samples
            i -= samples * <ssize_t> floor(<double> i / samples)
            signal[i] += <float_t> t


###############################################################################
# FRET model


@cython.ufunc
cdef (double, double) _phasor_from_fret_donor(
    double omega,
    double donor_lifetime,
    double fret_efficiency,
    double donor_fretting,
    double donor_background,
    double background_real,
    double background_imag,
) noexcept nogil:
    """Return phasor coordinates of FRET donor channel.

    See :py:func:`phasor_from_fret_donor` for parameter definitions.

    """
    cdef:
        double real, imag
        double quenched_real, quenched_imag  # quenched donor
        double f_pure, f_quenched, sum

    if fret_efficiency < 0.0:
        fret_efficiency = 0.0
    elif fret_efficiency > 1.0:
        fret_efficiency = 1.0

    if donor_fretting < 0.0:
        donor_fretting = 0.0
    elif donor_fretting > 1.0:
        donor_fretting = 1.0

    if donor_background < 0.0:
        donor_background = 0.0

    f_pure = 1.0 - donor_fretting
    f_quenched = (1.0 - fret_efficiency) * donor_fretting
    sum = f_pure + f_quenched + donor_background
    if sum < 1e-9:
        # no signal in donor channel
        return 1.0, 0.0

    # phasor of pure donor at frequency
    real, imag = phasor_from_lifetime(donor_lifetime, omega)

    # phasor of quenched donor
    quenched_real, quenched_imag = phasor_from_lifetime(
        donor_lifetime * (1.0 - fret_efficiency), omega
    )

    # weighted average
    real = (
        real * f_pure
        + quenched_real * f_quenched
        + donor_background * background_real
    ) / sum

    imag = (
        imag * f_pure
        + quenched_imag * f_quenched
        + background_imag * donor_background
    ) / sum

    return real, imag


@cython.ufunc
cdef (double, double) _phasor_from_fret_acceptor(
    double omega,
    double donor_lifetime,
    double acceptor_lifetime,
    double fret_efficiency,
    double donor_fretting,
    double donor_bleedthrough,
    double acceptor_bleedthrough,
    double acceptor_background,
    double background_real,
    double background_imag,
) noexcept nogil:
    """Return phasor coordinates of FRET acceptor channel.

    See :py:func:`phasor_from_fret_acceptor` for parameter definitions.

    """
    cdef:
        double phi, mod
        double donor_real, donor_imag
        double acceptor_real, acceptor_imag
        double quenched_real, quenched_imag  # quenched donor
        double sensitized_real, sensitized_imag  # sensitized acceptor
        double sum, f_donor, f_acceptor

    if fret_efficiency < 0.0:
        fret_efficiency = 0.0
    elif fret_efficiency > 1.0:
        fret_efficiency = 1.0

    if donor_fretting < 0.0:
        donor_fretting = 0.0
    elif donor_fretting > 1.0:
        donor_fretting = 1.0

    if donor_bleedthrough < 0.0:
        donor_bleedthrough = 0.0
    if acceptor_bleedthrough < 0.0:
        acceptor_bleedthrough = 0.0
    if acceptor_background < 0.0:
        acceptor_background = 0.0

    # phasor of pure donor at frequency
    donor_real, donor_imag = phasor_from_lifetime(donor_lifetime, omega)

    if fret_efficiency == 0.0:
        quenched_real = donor_real
        quenched_imag = donor_imag
    else:
        # phasor of quenched donor
        quenched_real, quenched_imag = phasor_from_lifetime(
            donor_lifetime * (1.0 - fret_efficiency), omega
        )

        # phasor of pure and quenched donor
        donor_real, donor_imag = linear_combination(
            1.0,
            0.0,
            donor_real,
            donor_imag,
            quenched_real,
            quenched_imag,
            1.0,
            1.0 - fret_efficiency,
            1.0 - donor_fretting
        )

    # phasor of acceptor at frequency
    acceptor_real, acceptor_imag = phasor_from_lifetime(
        acceptor_lifetime, omega
    )

    # phasor of acceptor sensitized by quenched donor
    # TODO: use rotation formula
    phi = (
        atan2(quenched_imag, quenched_real)
        + atan2(acceptor_imag, acceptor_real)
    )
    mod = (
        hypot(quenched_real, quenched_imag)
        * hypot(acceptor_real, acceptor_imag)
    )
    sensitized_real = mod * cos(phi)
    sensitized_imag = mod * sin(phi)

    # weighted average
    f_donor = donor_bleedthrough * (1.0 - donor_fretting * fret_efficiency)
    f_acceptor = donor_fretting * fret_efficiency
    sum = f_donor + f_acceptor + acceptor_bleedthrough + acceptor_background
    if sum < 1e-9:
        # no signal in acceptor channel
        # do not return 0, 0 to avoid discontinuities
        return sensitized_real, sensitized_imag

    acceptor_real = (
        donor_real * f_donor
        + sensitized_real * f_acceptor
        + acceptor_real * acceptor_bleedthrough
        + background_real * acceptor_background
    ) / sum

    acceptor_imag = (
        donor_imag * f_donor
        + sensitized_imag * f_acceptor
        + acceptor_imag * acceptor_bleedthrough
        + background_imag * acceptor_background
    ) / sum

    return acceptor_real, acceptor_imag


cdef inline (double, double) linear_combination(
    const double real,
    const double imag,
    const double real1,
    const double imag1,
    const double real2,
    const double imag2,
    double int1,
    double int2,
    double frac,
) noexcept nogil:
    """Return linear combinations of phasor coordinates."""
    int1 *= frac
    int2 *= 1.0 - frac
    frac = int1 + int2
    if fabs(frac) < 1e-15:
        return real, imag
    return (
        (int1 * real1 + int2 * real2) / frac,
        (int1 * imag1 + int2 * imag2) / frac
    )


cdef inline (float_t, float_t) phasor_from_lifetime(
    float_t lifetime,
    float_t omega,
) noexcept nogil:
    """Return phasor coordinates from single lifetime component."""
    cdef:
        double t = omega * lifetime
        double mod = 1.0 / sqrt(1.0 + t * t)
        double phi = atan(t)

    return <float_t> (mod * cos(phi)), <float_t> (mod * sin(phi))


###############################################################################
# Phasor conversions


@cython.ufunc
cdef (float_t, float_t) _phasor_transform(
    float_t real,
    float_t imag,
    float_t angle,
    float_t scale,
) noexcept nogil:
    """Return rotated and scaled phasor coordinates."""
    cdef:
        double g, s

    if isnan(real) or isnan(imag) or isnan(angle) or isnan(scale):
        return <float_t> NAN, <float_t> NAN

    g = scale * cos(angle)
    s = scale * sin(angle)

    return <float_t> (real * g - imag * s), <float_t> (real * s + imag * g)


@cython.ufunc
cdef (float_t, float_t) _phasor_transform_const(
    float_t real,
    float_t imag,
    float_t real2,
    float_t imag2,
) noexcept nogil:
    """Return rotated and scaled phasor coordinates."""
    if isnan(real) or isnan(imag) or isnan(real2) or isnan(imag2):
        return <float_t> NAN, <float_t> NAN

    return real * real2 - imag * imag2, real * imag2 + imag * real2


@cython.ufunc
cdef (float_t, float_t) _phasor_to_polar(
    float_t real,
    float_t imag,
) noexcept nogil:
    """Return polar from phasor coordinates."""
    if isnan(real) or isnan(imag):
        return <float_t> NAN, <float_t> NAN

    return (
        <float_t> atan2(imag, real),
        <float_t> sqrt(real * real + imag * imag)
    )


@cython.ufunc
cdef (float_t, float_t) _phasor_from_polar(
    float_t phase,
    float_t modulation,
) noexcept nogil:
    """Return phasor from polar coordinates."""
    if isnan(phase) or isnan(modulation):
        return <float_t> NAN, <float_t> NAN

    return (
        modulation * <float_t> cos(phase),
        modulation * <float_t> sin(phase)
    )


@cython.ufunc
cdef (float_t, float_t) _phasor_to_apparent_lifetime(
    float_t real,
    float_t imag,
    float_t omega,
) noexcept nogil:
    """Return apparent single lifetimes from phasor coordinates."""
    cdef:
        double tauphi = INFINITY
        double taumod = INFINITY
        double t

    if isnan(real) or isnan(imag):
        return <float_t> NAN, <float_t> NAN

    t = real * real + imag * imag
    if omega > 0.0 and t > 0.0:
        if fabs(real * omega) > 0.0:
            tauphi = imag / (real * omega)
        if t <= 1.0:
            taumod = sqrt(1.0 / t - 1.0) / omega
        else:
            taumod = 0.0

    return <float_t> tauphi, <float_t> taumod


@cython.ufunc
cdef (float_t, float_t) _phasor_from_apparent_lifetime(
    float_t tauphi,
    float_t taumod,
    float_t omega,
) noexcept nogil:
    """Return phasor coordinates from apparent single lifetimes."""
    cdef:
        double phi, mod, t

    if isnan(tauphi) or isnan(taumod):
        return <float_t> NAN, <float_t> NAN

    t = omega * taumod
    mod = 1.0 / sqrt(1.0 + t * t)
    phi = atan(omega * tauphi)
    return <float_t> (mod * cos(phi)), <float_t> (mod * sin(phi))


@cython.ufunc
cdef float_t _phasor_to_normal_lifetime(
    float_t real,
    float_t imag,
    float_t omega,
) noexcept nogil:
    """Return normal lifetimes from phasor coordinates."""
    cdef:
        double taunorm = INFINITY
        double t

    if isnan(real) or isnan(imag):
        return <float_t> NAN

    omega *= omega
    if omega > 0.0:
        t = 0.5 * (1.0 + cos(atan2(imag, real - 0.5)))
        if t <= 0.0:
            taunorm = INFINITY
        elif t > 1.0:
            taunorm = NAN
        else:
            taunorm = sqrt((1.0 - t) / (omega * t))

    return <float_t> taunorm


@cython.ufunc
cdef (float_t, float_t) _phasor_from_single_lifetime(
    float_t lifetime,
    float_t omega,
) noexcept nogil:
    """Return phasor coordinates from single lifetime component."""
    cdef:
        double phi, mod, t

    if isnan(lifetime):
        return <float_t> NAN, <float_t> NAN

    t = omega * lifetime
    phi = atan(t)
    mod = 1.0 / sqrt(1.0 + t * t)
    return <float_t> (mod * cos(phi)), <float_t> (mod * sin(phi))


@cython.ufunc
cdef (float_t, float_t) _polar_from_single_lifetime(
    float_t lifetime,
    float_t omega,
) noexcept nogil:
    """Return polar coordinates from single lifetime component."""
    cdef:
        double t

    if isnan(lifetime):
        return <float_t> NAN, <float_t> NAN

    t = omega * lifetime
    return <float_t> atan(t), <float_t> (1.0 / sqrt(1.0 + t * t))


@cython.ufunc
cdef (float_t, float_t) _polar_to_apparent_lifetime(
    float_t phase,
    float_t modulation,
    float_t omega,
) noexcept nogil:
    """Return apparent single lifetimes from polar coordinates."""
    cdef:
        double tauphi = INFINITY
        double taumod = INFINITY
        double t

    if isnan(phase) or isnan(modulation):
        return <float_t> NAN, <float_t> NAN

    t = modulation * modulation
    if omega > 0.0 and t > 0.0:
        tauphi = tan(phase) / omega
        if t <= 1.0:
            taumod = sqrt(1.0 / t - 1.0) / omega
        else:
            taumod = 0.0
    return <float_t> tauphi, <float_t> taumod


@cython.ufunc
cdef (float_t, float_t) _polar_from_apparent_lifetime(
    float_t tauphi,
    float_t taumod,
    float_t omega,
) noexcept nogil:
    """Return polar coordinates from apparent single lifetimes."""
    cdef:
        double t

    if isnan(tauphi) or isnan(taumod):
        return <float_t> NAN, <float_t> NAN

    t = omega * taumod
    return (
        <float_t> (atan(omega * tauphi)),
        <float_t> (1.0 / sqrt(1.0 + t * t))
    )


@cython.ufunc
cdef (float_t, float_t) _polar_from_reference(
    float_t measured_phase,
    float_t measured_modulation,
    float_t known_phase,
    float_t known_modulation,
) noexcept nogil:
    """Return polar coordinates for calibration from reference coordinates."""
    if (
        isnan(measured_phase)
        or isnan(measured_modulation)
        or isnan(known_phase)
        or isnan(known_modulation)
    ):
        return <float_t> NAN, <float_t> NAN

    if fabs(measured_modulation) == 0.0:
        # return known_phase - measured_phase, <float_t> INFINITY
        return (
            known_phase - measured_phase,
            <float_t> (NAN if known_modulation == 0.0 else INFINITY)
        )
    return known_phase - measured_phase, known_modulation / measured_modulation


@cython.ufunc
cdef (float_t, float_t) _polar_from_reference_phasor(
    float_t measured_real,
    float_t measured_imag,
    float_t known_real,
    float_t known_imag,
) noexcept nogil:
    """Return polar coordinates for calibration from reference phasor."""
    cdef:
        double measured_phase, measured_modulation
        double known_phase, known_modulation

    if (
        isnan(measured_real)
        or isnan(measured_imag)
        or isnan(known_real)
        or isnan(known_imag)
    ):
        return <float_t> NAN, <float_t> NAN

    measured_phase = atan2(measured_imag, measured_real)
    known_phase = atan2(known_imag, known_real)
    measured_modulation = hypot(measured_real, measured_imag)
    known_modulation = hypot(known_real, known_imag)

    if fabs(measured_modulation) == 0.0:
        # return <float_t> (known_phase - measured_phase), <float_t> INFINITY
        return (
            <float_t> (known_phase - measured_phase),
            <float_t> (NAN if known_modulation == 0.0 else INFINITY)
        )
    return (
        <float_t> (known_phase - measured_phase),
        <float_t> (known_modulation / measured_modulation)
    )


@cython.ufunc
cdef (float_t, float_t) _phasor_at_harmonic(
    float_t real,
    int harmonic,
    int other_harmonic,
) noexcept nogil:
    """Return phasor coordinates on universal semicircle at other harmonic."""
    if isnan(real):
        return <float_t> NAN, <float_t> NAN

    if real <= 0.0:
        return 0.0, 0.0
    if real >= 1.0:
        return 1.0, 0.0

    harmonic *= harmonic
    other_harmonic *= other_harmonic
    real = (
        harmonic * real / (other_harmonic + (harmonic - other_harmonic) * real)
    )

    return real, <float_t> sqrt(real - real * real)


@cython.ufunc
cdef (float_t, float_t) _phasor_multiply(
    float_t real,
    float_t imag,
    float_t real2,
    float_t imag2,
) noexcept nogil:
    """Return complex multiplication of two phasors."""
    return (
        real * real2 - imag * imag2,
        real * imag2 + imag * real2
    )


@cython.ufunc
cdef (float_t, float_t) _phasor_divide(
    float_t real,
    float_t imag,
    float_t real2,
    float_t imag2,
) noexcept nogil:
    """Return complex division of two phasors."""
    cdef:
        float_t divisor = real2 * real2 + imag2 * imag2

    if divisor != 0.0:
        # includes isnan(divisor)
        return (
            (real * real2 + imag * imag2) / divisor,
            (imag * real2 - real * imag2) / divisor
        )

    real = real * real2 + imag * imag2
    imag = imag * real2 - real * imag2
    return (
        NAN if real == 0.0 else real * INFINITY,
        NAN if imag == 0.0 else imag * INFINITY
    )


###############################################################################
# Geometry ufuncs


@cython.ufunc
cdef unsigned char _is_inside_range(
    float_t x,  # point
    float_t y,
    float_t xmin,  # x range
    float_t xmax,
    float_t ymin,  # y range
    float_t ymax
) noexcept nogil:
    """Return whether point is inside range.

    Range includes lower but not upper limit.

    """
    if isnan(x) or isnan(y):
        return False

    return x >= xmin and x < xmax and y >= ymin and y < ymax


@cython.ufunc
cdef unsigned char _is_inside_rectangle(
    float_t x,  # point
    float_t y,
    float_t x0,  # segment start
    float_t y0,
    float_t x1,  # segment end
    float_t y1,
    float_t r,  # half width
) noexcept nogil:
    """Return whether point is in rectangle.

    The rectangle is defined by central line segment and half width.

    """
    cdef:
        float_t t

    if r <= 0.0 or isnan(x) or isnan(y):
        return False

    # normalize coordinates
    # x1 = 0
    # y1 = 0
    x0 -= x1
    y0 -= y1
    x -= x1
    y -= y1
    # square of line length
    t = x0 * x0 + y0 * y0
    if t <= 0.0:
        return x * x + y * y <= r * r
    # projection of point on line using clamped dot product
    t = (x * x0 + y * y0) / t
    if t < 0.0 or t > 1.0:
        return False
    # compare square of lengths of projection and radius
    x -= t * x0
    y -= t * y0
    return x * x + y * y <= r * r


@cython.ufunc
cdef unsigned char _is_inside_polar_rectangle(
    float_t x,  # point
    float_t y,
    float_t angle_min,  # phase, -pi to pi
    float_t angle_max,
    float_t distance_min,  # modulation
    float_t distance_max,
) noexcept nogil:
    """Return whether point is inside polar rectangle.

    Angles should be in range [-pi, pi], else performance is degraded.

    """
    cdef:
        double t

    if isnan(x) or isnan(y):
        return False

    if distance_min > distance_max:
        distance_min, distance_max = distance_max, distance_min
    t = hypot(x, y)
    if t < distance_min or t > distance_max or t == 0.0:
        return False

    if angle_min < -M_PI or angle_min > M_PI:
        angle_min = <float_t> atan2(sin(angle_min), cos(angle_min))
    if angle_max < -M_PI or angle_max > M_PI:
        angle_max = <float_t> atan2(sin(angle_max), cos(angle_max))
    if angle_min > angle_max:
        angle_min, angle_max = angle_max, angle_min
    t = <float_t> atan2(y, x)
    if t < angle_min or t > angle_max:
        return False

    return True


@cython.ufunc
cdef unsigned char _is_inside_circle(
    float_t x,  # point
    float_t y,
    float_t x0,  # circle center
    float_t y0,
    float_t r,  # circle radius
) noexcept nogil:
    """Return whether point is inside circle."""
    if r <= 0.0 or isnan(x) or isnan(y):
        return False

    x -= x0
    y -= y0
    return x * x + y * y <= r * r


@cython.ufunc
cdef unsigned char _is_inside_ellipse(
    float_t x,  # point
    float_t y,
    float_t x0,  # ellipse center
    float_t y0,
    float_t a,  # ellipse radii
    float_t b,
    float_t phi,  # ellipse angle
) noexcept nogil:
    """Return whether point is inside ellipse.

    Same as _is_inside_circle if a == b.
    Consider using _is_inside_ellipse_ instead, which should be faster
    for arrays.

    """
    cdef:
        float_t sina, cosa

    if a <= 0.0 or b <= 0.0 or isnan(x) or isnan(y):
        return False

    x -= x0
    y -= y0
    if a == b:
        # circle
        return x * x + y * y <= a * a
    sina = <float_t> sin(phi)
    cosa = <float_t> cos(phi)
    x0 = (cosa * x + sina * y) / a
    y0 = (sina * x - cosa * y) / b
    return x0 * x0 + y0 * y0 <= 1.0


@cython.ufunc
cdef unsigned char _is_inside_ellipse_(
    float_t x,  # point
    float_t y,
    float_t x0,  # ellipse center
    float_t y0,
    float_t a,  # ellipse radii
    float_t b,
    float_t sina,  # sin/cos of ellipse angle
    float_t cosa,
) noexcept nogil:
    """Return whether point is inside ellipse.

    Use pre-calculated sin(angle) and cos(angle).

    """
    if a <= 0.0 or b <= 0.0 or isnan(x) or isnan(y):
        return False

    x -= x0
    y -= y0
    if a == b:
        # circle
        return x * x + y * y <= a * a
    x0 = (cosa * x + sina * y) / a
    y0 = (sina * x - cosa * y) / b
    return x0 * x0 + y0 * y0 <= 1.0


@cython.ufunc
cdef unsigned char _is_inside_stadium(
    float_t x,  # point
    float_t y,
    float_t x0,  # line start
    float_t y0,
    float_t x1,  # line end
    float_t y1,
    float_t r,  # radius
) noexcept nogil:
    """Return whether point is inside stadium.

    A stadium shape is a thick line with rounded ends.
    Same as _is_near_segment.

    """
    cdef:
        float_t t

    if r <= 0.0 or isnan(x) or isnan(y):
        return False

    # normalize coordinates
    # x1 = 0
    # y1 = 0
    x0 -= x1
    y0 -= y1
    x -= x1
    y -= y1
    # square of line length
    t = x0 * x0 + y0 * y0
    if t <= 0.0:
        return x * x + y * y <= r * r
    # projection of point on line using clamped dot product
    t = (x * x0 + y * y0) / t
    t = <float_t> max(0.0, min(1.0, t))
    # compare square of lengths of projection and radius
    x -= t * x0
    y -= t * y0
    return x * x + y * y <= r * r


# function alias
_is_near_segment = _is_inside_stadium


@cython.ufunc
cdef unsigned char _is_inside_semicircle(
    float_t x,  # point
    float_t y,
    float_t r,  # distance
) noexcept nogil:
    """Return whether point is inside universal semicircle."""
    if r < 0.0 or isnan(x) or isnan(y):
        return False
    if y < -r:
        return False
    if y <= 0.0:
        if x >= 0.0 and x <= 1.0:
            return True
        # near endpoints?
        if x > 0.5:
            x -= <float_t> 1.0
        return x * x + y * y <= r * r
    return hypot(x - 0.5, y) <= r + 0.5


@cython.ufunc
cdef unsigned char _is_near_semicircle(
    float_t x,  # point
    float_t y,
    float_t r,  # distance
) noexcept nogil:
    """Return whether point is near universal semicircle."""
    if r < 0.0 or isnan(x) or isnan(y):
        return False
    if y < 0.0:
        # near endpoints?
        if x > 0.5:
            x -= <float_t> 1.0
        return x * x + y * y <= r * r
    return fabs(hypot(x - 0.5, y) - 0.5) <= r


@cython.ufunc
cdef unsigned char _is_near_line(
    float_t x,  # point
    float_t y,
    float_t x0,  # line start
    float_t y0,
    float_t x1,  # line end
    float_t y1,
    float_t r,  # distance
) noexcept nogil:
    """Return whether point is close to line."""
    cdef:
        float_t t

    if r <= 0.0 or isnan(x) or isnan(y):
        return False

    # normalize coordinates
    # x1 = 0
    # y1 = 0
    x0 -= x1
    y0 -= y1
    x -= x1
    y -= y1
    # square of line length
    t = x0 * x0 + y0 * y0
    if t <= 0.0:
        return x * x + y * y <= r * r
    # projection of point on line using clamped dot product
    t = (x * x0 + y * y0) / t
    # compare square of lengths of projection and radius
    x -= t * x0
    y -= t * y0
    return x * x + y * y <= r * r


@cython.ufunc
cdef (float_t, float_t) _point_on_segment(
    float_t x,  # point
    float_t y,
    float_t x0,  # segment start
    float_t y0,
    float_t x1,  # segment end
    float_t y1,
) noexcept nogil:
    """Return point projected onto line segment."""
    cdef:
        float_t t

    if isnan(x) or isnan(y):
        return <float_t> NAN, <float_t> NAN

    # normalize coordinates
    # x1 = 0
    # y1 = 0
    x0 -= x1
    y0 -= y1
    x -= x1
    y -= y1
    # square of line length
    t = x0 * x0 + y0 * y0
    if t <= 0.0:
        return x0, y0
    # projection of point on line
    t = (x * x0 + y * y0) / t
    # clamp to line segment
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    x1 += t * x0
    y1 += t * y0
    return x1, y1


@cython.ufunc
cdef (float_t, float_t) _point_on_line(
    float_t x,  # point
    float_t y,
    float_t x0,  # line start
    float_t y0,
    float_t x1,  # line end
    float_t y1,
) noexcept nogil:
    """Return point projected onto line."""
    cdef:
        float_t t

    if isnan(x) or isnan(y):
        return <float_t> NAN, <float_t> NAN

    # normalize coordinates
    # x1 = 0
    # y1 = 0
    x0 -= x1
    y0 -= y1
    x -= x1
    y -= y1
    # square of line length
    t = x0 * x0 + y0 * y0
    if t <= 0.0:
        return x0, y0
    # projection of point on line
    t = (x * x0 + y * y0) / t
    x1 += t * x0
    y1 += t * y0
    return x1, y1


@cython.ufunc
cdef float_t _fraction_on_segment(
    float_t x,  # point
    float_t y,
    float_t x0,  # segment start
    float_t y0,
    float_t x1,  # segment end
    float_t y1,
) noexcept nogil:
    """Return normalized fraction of point projected onto line segment."""
    cdef:
        float_t t

    if isnan(x) or isnan(y):
        return <float_t> NAN

    # normalize coordinates
    x -= x1
    y -= y1
    x0 -= x1
    y0 -= y1
    # x1 = 0
    # y1 = 0
    # square of line length
    t = x0 * x0 + y0 * y0
    if t <= 0.0:
        # not a line segment
        return 0.0
    # projection of point on line
    t = (x * x0 + y * y0) / t
    # clamp to line segment
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    return t


@cython.ufunc
cdef float_t _fraction_on_line(
    float_t x,  # point
    float_t y,
    float_t x0,  # line start
    float_t y0,
    float_t x1,  # line end
    float_t y1,
) noexcept nogil:
    """Return normalized fraction of point projected onto line."""
    cdef:
        float_t t

    if isnan(x) or isnan(y):
        return <float_t> NAN

    # normalize coordinates
    x -= x1
    y -= y1
    x0 -= x1
    y0 -= y1
    # x1 = 0
    # y1 = 0
    # square of line length
    t = x0 * x0 + y0 * y0
    if t <= 0.0:
        # not a line segment
        return 1.0
    # projection of point on line
    t = (x * x0 + y * y0) / t
    return t


@cython.ufunc
cdef float_t _distance_from_point(
    float_t x,  # point
    float_t y,
    float_t x0,  # other point
    float_t y0,
) noexcept nogil:
    """Return distance from point."""
    if isnan(x) or isnan(y):  # or isnan(x0) or isnan(y0)
        return <float_t> NAN

    return <float_t> hypot(x - x0, y - y0)


@cython.ufunc
cdef float_t _distance_from_segment(
    float_t x,  # point
    float_t y,
    float_t x0,  # segment start
    float_t y0,
    float_t x1,  # segment end
    float_t y1,
) noexcept nogil:
    """Return distance from segment."""
    cdef:
        float_t t

    if isnan(x) or isnan(y):
        return <float_t> NAN

    # normalize coordinates
    # x1 = 0
    # y1 = 0
    x0 -= x1
    y0 -= y1
    x -= x1
    y -= y1
    # square of line length
    t = x0 * x0 + y0 * y0
    if t <= 0.0:
        return <float_t> hypot(x, y)
    # projection of point on line using dot product
    t = (x * x0 + y * y0) / t
    if t > 1.0:
        x -= x0
        y -= y0
    elif t > 0.0:
        x -= t * x0
        y -= t * y0
    return <float_t> hypot(x, y)


@cython.ufunc
cdef float_t _distance_from_line(
    float_t x,  # point
    float_t y,
    float_t x0,  # line start
    float_t y0,
    float_t x1,  # line end
    float_t y1,
) noexcept nogil:
    """Return distance from line."""
    cdef:
        float_t t

    if isnan(x) or isnan(y):
        return <float_t> NAN

    # normalize coordinates
    # x1 = 0
    # y1 = 0
    x0 -= x1
    y0 -= y1
    x -= x1
    y -= y1
    # square of line length
    t = x0 * x0 + y0 * y0
    if t <= 0.0:
        return <float_t> hypot(x, y)
    # projection of point on line using dot product
    t = (x * x0 + y * y0) / t
    x -= t * x0
    y -= t * y0
    return <float_t> hypot(x, y)


@cython.ufunc
cdef float_t _distance_from_semicircle(
    float_t x,  # point
    float_t y,
) noexcept nogil:
    """Return distance from universal semicircle."""
    if isnan(x) or isnan(y):
        return NAN
    if y < 0.0:
        # distance to endpoints
        if x > 0.5:
            x -= <float_t> 1.0
        return <float_t> hypot(x, y)
    return <float_t> fabs(hypot(x - 0.5, y) - 0.5)


@cython.ufunc
cdef (float_t, float_t, float_t) _segment_direction_and_length(
    float_t x0,  # segment start
    float_t y0,
    float_t x1,  # segment end
    float_t y1,
) noexcept nogil:
    """Return direction and length of line segment."""
    cdef:
        float_t length

    if isnan(x0) or isnan(y0) or isnan(x1) or isnan(y1):
        return NAN, NAN, 0.0

    x1 -= x0
    y1 -= y0
    length = <float_t> hypot(x1, y1)
    if length <= 0.0:
        return NAN, NAN, 0.0
    x1 /= length
    y1 /= length
    return x1, y1, length


@cython.ufunc
cdef (float_t, float_t, float_t, float_t) _intersect_circle_circle(
    float_t x0,  # circle 0
    float_t y0,
    float_t r0,
    float_t x1,  # circle 1
    float_t y1,
    float_t r1,
) noexcept nogil:
    """Return coordinates of intersections of two circles."""
    cdef:
        double dx, dy, dr, ll, dd, hd, ld

    if (
        isnan(x0)
        or isnan(y0)
        or isnan(r0)
        or isnan(x1)
        or isnan(y1)
        or isnan(r1)
        or r0 == 0.0
        or r1 == 0.0
    ):
        return NAN, NAN, NAN, NAN

    dx = x1 - x0
    dy = y1 - y0
    dr = hypot(dx, dy)
    if dr <= 0.0:
        # circle positions identical
        return NAN, NAN, NAN, NAN
    ll = (r0 * r0 - r1 * r1 + dr * dr) / (dr + dr)
    dd = r0 * r0 - ll * ll
    if dd < 0.0 or dr <= 0.0:
        # circles not intersecting
        return NAN, NAN, NAN, NAN
    hd = sqrt(dd) / dr
    ld = ll / dr
    return (
        <float_t> (ld * dx + hd * dy + x0),
        <float_t> (ld * dy - hd * dx + y0),
        <float_t> (ld * dx - hd * dy + x0),
        <float_t> (ld * dy + hd * dx + y0),
    )


@cython.ufunc
cdef (float_t, float_t, float_t, float_t) _intersect_circle_line(
    float_t x,  # circle
    float_t y,
    float_t r,
    float_t x0,  # line start
    float_t y0,
    float_t x1,  # line end
    float_t y1,
) noexcept nogil:
    """Return coordinates of intersections of circle and line."""
    cdef:
        double dx, dy, dr, dd, rdd

    if (
        isnan(r)
        or isnan(x)
        or isnan(y)
        or isnan(x0)
        or isnan(y0)
        or isnan(x1)
        or isnan(y1)
        or r == 0.0
    ):
        return NAN, NAN, NAN, NAN

    dx = x1 - x0
    dy = y1 - y0
    dr = dx * dx + dy * dy
    dd = (x0 - x) * (y1 - y) - (x1 - x) * (y0 - y)
    rdd = r * r * dr - dd * dd  # discriminant
    if rdd < 0.0 or dr <= 0.0:
        # no intersection
        return NAN, NAN, NAN, NAN
    rdd = sqrt(rdd)
    return (
        x + <float_t> ((dd * dy + copysign(1.0, dy) * dx * rdd) / dr),
        y + <float_t> ((-dd * dx + fabs(dy) * rdd) / dr),
        x + <float_t> ((dd * dy - copysign(1.0, dy) * dx * rdd) / dr),
        y + <float_t> ((-dd * dx - fabs(dy) * rdd) / dr),
    )


@cython.ufunc
cdef (float_t, float_t, float_t, float_t) _intersect_semicircle_line(
    float_t x0,  # line start
    float_t y0,
    float_t x1,  # line end
    float_t y1,
) noexcept nogil:
    """Return coordinates of intersections of line and universal semicircle."""
    cdef:
        double dx, dy, dr, dd, rdd

    if isnan(x0) or isnan(x1) or isnan(y0) or isnan(y1):
        return NAN, NAN, NAN, NAN

    dx = x1 - x0
    dy = y1 - y0
    dr = dx * dx + dy * dy
    dd = (x0 - 0.5) * y1 - (x1 - 0.5) * y0
    rdd = 0.25 * dr - dd * dd  # discriminant
    if rdd < 0.0 or dr <= 0.0:
        # no intersection
        return NAN, NAN, NAN, NAN
    rdd = sqrt(rdd)
    x0 = <float_t> ((dd * dy - copysign(1.0, dy) * dx * rdd) / dr + 0.5)
    y0 = <float_t> ((-dd * dx - fabs(dy) * rdd) / dr)
    x1 = <float_t> ((dd * dy + copysign(1.0, dy) * dx * rdd) / dr + 0.5)
    y1 = <float_t> ((-dd * dx + fabs(dy) * rdd) / dr)
    if y0 < 0.0:
        x0 = NAN
        y0 = NAN
    if y1 < 0.0:
        x1 = NAN
        y1 = NAN
    return x0, y0, x1, y1


def _nearest_neighbor_2d(
    int_t[::1] indices,
    const float_t[::1] x0,
    const float_t[::1] y0,
    const float_t[::1] x1,
    const float_t[::1] y1,
    const float_t distance_max,
    const int num_threads
):
    """Find nearest neighbors in 2D.

    For each point in the first set of arrays (x0, y0) find the nearest point
    in the second set of arrays (x1, y1) and store the index of the nearest
    point in the second array in the indices array.
    If any coordinates are NaN, or the distance to the nearest point
    is larger than distance_max, the index is set to -1.

    """
    cdef:
        ssize_t i, j, index
        float_t x, y, dmin
        float_t distance_max_squared = distance_max * distance_max

    if (
        indices.shape[0] != x0.shape[0]
        or x0.shape[0] != y0.shape[0]
        or x1.shape[0] != y1.shape[0]
    ):
        raise ValueError('input array size mismatch')

    with nogil, parallel(num_threads=num_threads):
        for i in prange(x0.shape[0]):
            x = x0[i]
            y = y0[i]
            if isnan(x) or isnan(y):
                indices[i] = -1
                continue
            index = -1
            dmin = INFINITY
            for j in range(x1.shape[0]):
                x = x0[i] - x1[j]
                y = y0[i] - y1[j]
                x = x * x + y * y
                if x < dmin:
                    dmin = x
                    index = j
            indices[i] = -1 if dmin > distance_max_squared else <int_t> index


###############################################################################
# Blend ufuncs


@cython.ufunc
cdef float_t _blend_and(
    float_t a,  # base layer
    float_t b,  # blend layer
) noexcept nogil:
    """Return blended layers using `and` mode."""
    if isnan(a):
        return NAN
    return b


@cython.ufunc
cdef float_t _blend_normal(
    float_t a,  # base layer
    float_t b,  # blend layer
) noexcept nogil:
    """Return blended layers using `normal` mode."""
    if isnan(b):
        return a
    return b


@cython.ufunc
cdef float_t _blend_multiply(
    float_t a,  # base layer
    float_t b,  # blend layer
) noexcept nogil:
    """Return blended layers using `multiply` mode."""
    if isnan(b):
        return a
    return a * b


@cython.ufunc
cdef float_t _blend_screen(
    float_t a,  # base layer
    float_t b,  # blend layer
) noexcept nogil:
    """Return blended layers using `screen` mode."""
    if isnan(b):
        return a
    return <float_t> (1.0 - (1.0 - a) * (1.0 - b))


@cython.ufunc
cdef float_t _blend_overlay(
    float_t a,  # base layer
    float_t b,  # blend layer
) noexcept nogil:
    """Return blended layers using `overlay` mode."""
    if isnan(b) or isnan(a):
        return a
    if a < 0.5:
        return <float_t> (2.0 * a * b)
    return <float_t> (1.0 - 2.0 * (1.0 - a) * (1.0 - b))


@cython.ufunc
cdef float_t _blend_darken(
    float_t a,  # base layer
    float_t b,  # blend layer
) noexcept nogil:
    """Return blended layers using `darken` mode."""
    if isnan(b) or isnan(a):
        return a
    return <float_t> min(a, b)


@cython.ufunc
cdef float_t _blend_lighten(
    float_t a,  # base layer
    float_t b,  # blend layer
) noexcept nogil:
    """Return blended layers using `lighten` mode."""
    if isnan(b) or isnan(a):
        return a
    return <float_t> max(a, b)


###############################################################################
# Threshold ufuncs


@cython.ufunc
cdef (float_t, float_t, float_t) _phasor_threshold_open(
    float_t mean,
    float_t real,
    float_t imag,
    float_t mean_min,
    float_t mean_max,
    float_t real_min,
    float_t real_max,
    float_t imag_min,
    float_t imag_max,
    float_t phase_min,
    float_t phase_max,
    float_t modulation_min,
    float_t modulation_max,
) noexcept nogil:
    """Return thresholded values by open intervals."""
    cdef:
        double phi = NAN
        double mod = NAN

    if isnan(mean) or isnan(real) or isnan(imag):
        return <float_t> NAN, <float_t> NAN, <float_t> NAN

    if not isnan(mean_min) and mean <= mean_min:
        return <float_t> NAN, <float_t> NAN, <float_t> NAN
    if not isnan(mean_max) and mean >= mean_max:
        return <float_t> NAN, <float_t> NAN, <float_t> NAN

    if not isnan(real_min) and real <= real_min:
        return <float_t> NAN, <float_t> NAN, <float_t> NAN
    if not isnan(real_max) and real >= real_max:
        return <float_t> NAN, <float_t> NAN, <float_t> NAN

    if not isnan(imag_min) and imag <= imag_min:
        return <float_t> NAN, <float_t> NAN, <float_t> NAN
    if not isnan(imag_max) and imag >= imag_max:
        return <float_t> NAN, <float_t> NAN, <float_t> NAN

    if not isnan(modulation_min):
        mod = real * real + imag * imag
        if mod <= modulation_min * modulation_min:
            return <float_t> NAN, <float_t> NAN, <float_t> NAN
    if not isnan(modulation_max):
        if isnan(mod):
            mod = real * real + imag * imag
        if mod >= modulation_max * modulation_max:
            return <float_t> NAN, <float_t> NAN, <float_t> NAN

    if not isnan(phase_min):
        phi = atan2(imag, real)
        if phi <= phase_min:
            return <float_t> NAN, <float_t> NAN, <float_t> NAN
    if not isnan(phase_max):
        if isnan(phi):
            phi = atan2(imag, real)
        if phi >= phase_max:
            return <float_t> NAN, <float_t> NAN, <float_t> NAN

    return mean, real, imag


@cython.ufunc
cdef (float_t, float_t, float_t) _phasor_threshold_closed(
    float_t mean,
    float_t real,
    float_t imag,
    float_t mean_min,
    float_t mean_max,
    float_t real_min,
    float_t real_max,
    float_t imag_min,
    float_t imag_max,
    float_t phase_min,
    float_t phase_max,
    float_t modulation_min,
    float_t modulation_max,
) noexcept nogil:
    """Return thresholded values by closed intervals."""
    cdef:
        double phi = NAN
        double mod = NAN

    if isnan(mean) or isnan(real) or isnan(imag):
        return <float_t> NAN, <float_t> NAN, <float_t> NAN

    if not isnan(mean_min) and mean < mean_min:
        return <float_t> NAN, <float_t> NAN, <float_t> NAN
    if not isnan(mean_max) and mean > mean_max:
        return <float_t> NAN, <float_t> NAN, <float_t> NAN

    if not isnan(real_min) and real < real_min:
        return <float_t> NAN, <float_t> NAN, <float_t> NAN
    if not isnan(real_max) and real > real_max:
        return <float_t> NAN, <float_t> NAN, <float_t> NAN

    if not isnan(imag_min) and imag < imag_min:
        return <float_t> NAN, <float_t> NAN, <float_t> NAN
    if not isnan(imag_max) and imag > imag_max:
        return <float_t> NAN, <float_t> NAN, <float_t> NAN

    if not isnan(modulation_min):
        mod = real * real + imag * imag
        if mod < modulation_min * modulation_min:
            return <float_t> NAN, <float_t> NAN, <float_t> NAN
    if not isnan(modulation_max):
        if isnan(mod):
            mod = real * real + imag * imag
        if mod > modulation_max * modulation_max:
            return <float_t> NAN, <float_t> NAN, <float_t> NAN

    if not isnan(phase_min):
        phi = atan2(imag, real)
        if phi < phase_min:
            return <float_t> NAN, <float_t> NAN, <float_t> NAN
    if not isnan(phase_max):
        if isnan(phi):
            phi = atan2(imag, real)
        if phi > phase_max:
            return <float_t> NAN, <float_t> NAN, <float_t> NAN

    return mean, real, imag


@cython.ufunc
cdef (float_t, float_t, float_t) _phasor_threshold_mean_open(
    float_t mean,
    float_t real,
    float_t imag,
    float_t mean_min,
    float_t mean_max,
) noexcept nogil:
    """Return thresholded values only by open interval of `mean`."""
    if isnan(mean) or isnan(real) or isnan(imag):
        return <float_t> NAN, <float_t> NAN, <float_t> NAN

    if not isnan(mean_min) and mean <= mean_min:
        return <float_t> NAN, <float_t> NAN, <float_t> NAN
    if not isnan(mean_max) and mean >= mean_max:
        return <float_t> NAN, <float_t> NAN, <float_t> NAN

    return mean, real, imag


@cython.ufunc
cdef (float_t, float_t, float_t) _phasor_threshold_mean_closed(
    float_t mean,
    float_t real,
    float_t imag,
    float_t mean_min,
    float_t mean_max,
) noexcept nogil:
    """Return thresholded values only by closed interval of `mean`."""
    if isnan(mean) or isnan(real) or isnan(imag):
        return <float_t> NAN, <float_t> NAN, <float_t> NAN

    if not isnan(mean_min) and mean < mean_min:
        return <float_t> NAN, <float_t> NAN, <float_t> NAN
    if not isnan(mean_max) and mean > mean_max:
        return <float_t> NAN, <float_t> NAN, <float_t> NAN

    return mean, real, imag


@cython.ufunc
cdef (float_t, float_t, float_t) _phasor_threshold_nan(
    float_t mean,
    float_t real,
    float_t imag,
) noexcept nogil:
    """Return the input values if any of them is not NaN."""
    if isnan(mean) or isnan(real) or isnan(imag):
        return <float_t> NAN, <float_t> NAN, <float_t> NAN

    return mean, real, imag


###############################################################################
# Unary ufuncs


@cython.ufunc
cdef float_t _anscombe(
    float_t x,
) noexcept nogil:
    """Return anscombe variance stabilizing transformation."""
    if isnan(x):
        return <float_t> NAN

    return <float_t> (2.0 * sqrt(<double> x + 0.375))


@cython.ufunc
cdef float_t _anscombe_inverse(
    float_t x,
) noexcept nogil:
    """Return inverse anscombe transformation."""
    if isnan(x):
        return <float_t> NAN

    return <float_t> (x * x / 4.0 - 0.375)  # 3/8


@cython.ufunc
cdef float_t _anscombe_inverse_approx(
    float_t x,
) noexcept nogil:
    """Return inverse anscombe transformation.

    Using approximation of exact unbiased inverse.

    """
    if isnan(x):
        return <float_t> NAN

    return <float_t> (
        0.25 * x * x  # 1/4
        + 0.30618621784789724 / x  # 1/4 * sqrt(3/2)
        - 1.375 / (x * x)  # 11/8
        + 0.7654655446197431 / (x * x * x)  # 5/8 * sqrt(3/2)
        - 0.125  # 1/8
    )


###############################################################################
# Denoising in spectral space


def _phasor_from_signal_vector(
    float_t[:, ::1] phasor,
    const signal_t[:, ::1] signal,
    const double[:, :, ::1] sincos,
    const int num_threads
):
    """Calculate phasor coordinate vectors from signal along last axis.

    Parameters
    ----------
    phasor : 2D memoryview of float32 or float64
        Writable buffer of two dimensions where calculated phasor
        vectors are stored:

        0. other dimensions flat
        1. real and imaginary components

    signal : 2D memoryview of float32 or float64
        Buffer of two dimensions containing signal:

        0. other dimensions flat
        1. dimension over which to compute FFT, number samples

    sincos : 3D memoryview of float64
        Buffer of three dimensions containing sine and cosine terms to be
        multiplied with signal:

        0. number harmonics
        1. number samples
        2. cos and sin

    num_threads : int
        Number of OpenMP threads to use for parallelization.

    Notes
    -----
    This implementation requires contiguous input arrays.

    """
    cdef:
        ssize_t size = signal.shape[0]
        ssize_t samples = signal.shape[1]
        ssize_t harmonics = sincos.shape[0]
        ssize_t i, j, k, h
        double dc, re, im, sample

    if (
        samples < 2
        or harmonics > samples // 2
        or phasor.shape[0] != size
        or phasor.shape[1] != harmonics * 2
    ):
        raise ValueError('invalid shape of phasor or signal')
    if sincos.shape[1] != samples or sincos.shape[2] != 2:
        raise ValueError('invalid shape of sincos')

    with nogil, parallel(num_threads=num_threads):
        for i in prange(signal.shape[0]):
            j = 0
            for h in range(harmonics):
                dc = 0.0
                re = 0.0
                im = 0.0
                for k in range(samples):
                    sample = <double> signal[i, k]
                    dc = dc + sample
                    re = re + sample * sincos[h, k, 0]
                    im = im + sample * sincos[h, k, 1]
                if dc != 0.0:
                    re = re / dc
                    im = im / dc
                else:
                    re = NAN if re == 0.0 else re * INFINITY
                    im = NAN if im == 0.0 else im * INFINITY
                phasor[i, j] = <float_t> re
                j = j + 1
                phasor[i, j] = <float_t> im
                j = j + 1


def _signal_denoise_vector(
    float_t[:, ::1] denoised,
    float_t[::1] integrated,
    const signal_t[:, ::1] signal,
    const float_t[:, ::1] spectral_vector,
    const double sigma,
    const double vmin,
    const int num_threads
):
    """Calculate denoised signal from spectral_vector."""
    cdef:
        ssize_t size = signal.shape[0]
        ssize_t samples = signal.shape[1]
        ssize_t dims = spectral_vector.shape[1]
        ssize_t i, j, m
        float_t n
        double weight, sum, t
        double sigma2 = -1.0 / (2.0 * sigma * sigma)
        double threshold = 9.0 * sigma * sigma

    if denoised.shape[0] != size or denoised.shape[1] != samples:
        raise ValueError('signal and denoised shape mismatch')
    if integrated.shape[0] != size:
        raise ValueError('integrated.shape[0] != signal.shape[0]')
    if spectral_vector.shape[0] != size:
        raise ValueError('spectral_vector.shape[0] != signal.shape[0]')

    with nogil, parallel(num_threads=num_threads):

        # integrate channel intensities for each pixel
        # and filter low intensities
        for i in prange(size):
            sum = 0.0
            for m in range(samples):
                sum = sum + <double> signal[i, m]
            if sum < vmin:
                sum = NAN
            integrated[i] = <float_t> sum

        # loop over all pixels
        for i in prange(size):

            n = integrated[i]
            if not n > 0.0:
                # n is NaN or zero; cannot denoise; return original signal
                continue

            for m in range(samples):
                denoised[i, m] /= n  # weight = 1.0

            # loop over other pixels
            for j in range(size):
                if i == j:
                    # weight = 1.0 already accounted for
                    continue

                n = integrated[j]
                if not n > 0.0:
                    # n is NaN or zero
                    continue

                # calculate weight from Euclidean distance of
                # pixels i and j in spectral vector space
                sum = 0.0
                for m in range(dims):
                    t = spectral_vector[i, m] - spectral_vector[j, m]
                    sum = sum + t * t
                    if sum > threshold:
                        sum = -1.0
                        break
                if sum >= 0.0:
                    weight = exp(sum * sigma2) / n
                else:
                    # sum is NaN or greater than threshold
                    continue

                # add weighted signal[j] to denoised[i]
                for m in range(samples):
                    denoised[i, m] += <float_t> (weight * signal[j, m])

            # re-normalize to original intensity
            # sum cannot be zero because integrated == 0 was filtered
            sum = 0.0
            for m in range(samples):
                sum = sum + denoised[i, m]
            n = <float_t> (<double> integrated[i] / sum)
            for m in range(samples):
                denoised[i, m] *= n


###############################################################################
# Filtering functions


cdef float_t _median(float_t *values, const ssize_t size) noexcept nogil:
    """Return median of array values using Quickselect algorithm."""
    cdef:
        ssize_t i, pivot_index, pivot_index_new
        ssize_t left = 0
        ssize_t right = size - 1
        ssize_t middle = size // 2
        float_t pivot_value, temp

    if size % 2 == 0:
        middle -= 1  # Quickselect sorts on right

    while left <= right:
        pivot_index = left + (right - left) // 2
        pivot_value = values[pivot_index]
        temp = values[pivot_index]
        values[pivot_index] = values[right]
        values[right] = temp
        pivot_index_new = left
        for i in range(left, right):
            if values[i] < pivot_value:
                temp = values[i]
                values[i] = values[pivot_index_new]
                values[pivot_index_new] = temp
                pivot_index_new += 1
        temp = values[right]
        values[right] = values[pivot_index_new]
        values[pivot_index_new] = temp

        if pivot_index_new == middle:
            if size % 2 == 0:
                return (values[middle] + values[middle + 1]) / <float_t> 2.0
            return values[middle]
        if pivot_index_new < middle:
            left = pivot_index_new + 1
        else:
            right = pivot_index_new - 1

    return values[middle]  # unreachable code?


def _median_filter_2d(
    float_t[:, :] image,
    float_t[:, ::1] filtered_image,
    const ssize_t kernel_size,
    const int repeat=1,
    const int num_threads=1,
):
    """Apply 2D median filter ignoring NaN."""
    cdef:
        ssize_t rows = image.shape[0]
        ssize_t cols = image.shape[1]
        ssize_t k = kernel_size // 2
        ssize_t i, j, r, di, dj, ki, kj, valid_count
        float_t element
        float_t *kernel

    if kernel_size <= 0:
        raise ValueError('kernel_size must be greater than 0')

    with nogil, parallel(num_threads=num_threads):

        kernel = <float_t *> malloc(
            kernel_size * kernel_size * sizeof(float_t)
        )
        if kernel == NULL:
            with gil:
                raise MemoryError('failed to allocate kernel')

        for r in range(repeat):
            for i in prange(rows):
                for j in range(cols):
                    if isnan(image[i, j]):
                        filtered_image[i, j] = <float_t> NAN
                        continue
                    valid_count = 0
                    for di in range(kernel_size):
                        ki = i - k + di
                        if ki < 0:
                            ki = 0
                        elif ki >= rows:
                            ki = rows - 1
                        for dj in range(kernel_size):
                            kj = j - k + dj
                            if kj < 0:
                                kj = 0
                            elif kj >= cols:
                                kj = cols - 1
                            element = image[ki, kj]
                            if not isnan(element):
                                kernel[valid_count] = element
                                valid_count = valid_count + 1
                    filtered_image[i, j] = _median(kernel, valid_count)

            for i in prange(rows):
                for j in range(cols):
                    image[i, j] = filtered_image[i, j]

        free(kernel)


###############################################################################
# Decoder functions


@cython.boundscheck(True)
def _flimlabs_signal(
    uint_t[:, :, ::] signal,   # channel, pixel, bin
    list data,  # list[list[list[[int, int]]]]
    ssize_t channel = -1  # -1 == None
):
    """Return TCSPC histogram image from FLIM LABS JSON intensity data."""
    cdef:
        uint_t[::] signal_
        list channels, pixels
        ssize_t c, i, h, count

    if channel < 0:
        c = 0
        for channels in data:
            i = 0
            for pixels in channels:
                signal_ = signal[c, i]
                for h, count in pixels:
                    signal_[h] = <uint_t> count
                i += 1
            c += 1
    else:
        i = 0
        for pixels in data[channel]:
            signal_ = signal[0, i]
            for h, count in pixels:
                signal_[h] = <uint_t> count
            i += 1


@cython.boundscheck(True)
def _flimlabs_mean(
    float_t[:, ::] mean,   # channel, pixel
    list data,  # list[list[list[[int, int]]]]
    ssize_t channel = -1  # -1 == None
):
    """Return mean intensity image from FLIM LABS JSON intensity data."""
    cdef:
        float_t[::] mean_
        list channels, pixels
        ssize_t c, i, h, count
        double sum

    if channel < 0:
        c = 0
        for channels in data:
            mean_ = mean[c]
            i = 0
            for pixels in channels:
                sum = 0.0
                for h, count in pixels:
                    sum += <double> count
                mean_[i] = <float_t> (sum / 256.0)
                i += 1
            c += 1
    else:
        i = 0
        mean_ = mean[0]
        for pixels in data[channel]:
            sum = 0.0
            for h, count in pixels:
                sum += <double> count
            mean_[i] = <float_t> (sum / 256.0)
            i += 1
