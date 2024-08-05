# distutils: language = c
# cython: language_level = 3
# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: nonecheck = False

"""Cython implementation of low-level functions for the PhasorPy package."""

# TODO: replace short with unsigned char when Cython supports it
# https://github.com/cython/cython/pull/6196#issuecomment-2209509572

# TODO: use fused return types for functions returning more than two items
# https://github.com/cython/cython/issues/6328

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
                    if fabs(sum) < 1e-16:
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
                    if fabs(sum) < 1e-16:
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
                    if fabs(sum) < 1e-16:
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
    """Return normal distribution, wrapped around at borders."""
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
    double donor_freting,
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

    if donor_freting < 0.0:
        donor_freting = 0.0
    elif donor_freting > 1.0:
        donor_freting = 1.0

    if donor_background < 0.0:
        donor_background = 0.0

    f_pure = 1.0 - donor_freting
    f_quenched = (1.0 - fret_efficiency) * donor_freting
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
    double donor_freting,
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

    if donor_freting < 0.0:
        donor_freting = 0.0
    elif donor_freting > 1.0:
        donor_freting = 1.0

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
            1.0 - donor_freting
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
    f_donor = donor_bleedthrough * (1.0 - donor_freting * fret_efficiency)
    f_acceptor = donor_freting * fret_efficiency
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
    if frac == 0.0:
        return real, imag
    return (
        (int1 * real1 + int2 * real2) / frac,
        (int1 * imag1 + int2 * imag2) / frac
    )

###############################################################################
# Phasor conversions

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


cdef inline (float_t, float_t) phasor_to_apparent_lifetime(
    float_t real,
    float_t imag,
    float_t omega,
) noexcept nogil:
    """Return apparent single lifetimes from phasor coordinates."""
    cdef:
        double tauphi = INFINITY
        double taumod = INFINITY
        double t = real * real + imag * imag

    if omega > 0.0 and t > 0.0:
        if fabs(real * omega) > 0.0:
            tauphi = imag / (real * omega)
        if t <= 1.0:
            taumod = sqrt(1.0 / t - 1.0) / omega
        else:
            taumod = 0.0

    return <float_t> tauphi, <float_t> taumod


cdef inline (float_t, float_t) phasor_from_apparent_lifetime(
    float_t tauphi,
    float_t taumod,
    float_t omega,
) noexcept nogil:
    """Return phasor coordinates from apparent single lifetimes."""
    cdef:
        double t = omega * taumod
        double mod = 1.0 / sqrt(1.0 + t * t)
        double phi = atan(omega * tauphi)

    return <float_t> (mod * cos(phi)), <float_t> (mod * sin(phi))


@cython.ufunc
cdef (float_t, float_t) _phasor_transform(
    float_t real,
    float_t imag,
    float_t angle,
    float_t scale,
) noexcept nogil:
    """Return rotated and scaled phasor coordinates."""
    cdef:
        double g = scale * cos(angle)
        double s = scale * sin(angle)

    return <float_t> (real * g - imag * s), <float_t> (real * s + imag * g)


@cython.ufunc
cdef (float_t, float_t) _phasor_transform_const(
    float_t real,
    float_t imag,
    float_t real2,
    float_t imag2,
) noexcept nogil:
    """Return rotated and scaled phasor coordinates."""
    return real * real2 - imag * imag2, real * imag2 + imag * real2


@cython.ufunc
cdef (float_t, float_t) _phasor_to_polar(
    float_t real,
    float_t imag,
) noexcept nogil:
    """Return polar from phasor coordinates."""
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
    return phasor_to_apparent_lifetime(real, imag, omega)


@cython.ufunc
cdef (float_t, float_t) _phasor_from_apparent_lifetime(
    float_t tauphi,
    float_t taumod,
    float_t omega,
) noexcept nogil:
    """Return phasor coordinates from apparent single lifetimes."""
    return phasor_from_apparent_lifetime(tauphi, taumod, omega)


@cython.ufunc
cdef (float_t, float_t) _phasor_from_single_lifetime(
    float_t lifetime,
    float_t omega,
) noexcept nogil:
    """Return phasor coordinates from single lifetime component."""
    cdef:
        double t = omega * lifetime
        double phi = atan(t)
        double mod = 1.0 / sqrt(1.0 + t * t)

    return <float_t> (mod * cos(phi)), <float_t> (mod * sin(phi))


@cython.ufunc
cdef (float_t, float_t) _polar_from_single_lifetime(
    float_t lifetime,
    float_t omega,
) noexcept nogil:
    """Return polar coordinates from single lifetime component."""
    cdef:
        double t = omega * lifetime

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
        double t = modulation * modulation

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
        double t = omega * taumod

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
    if fabs(measured_modulation) == 0.0:
        return known_phase - measured_phase, <float_t> INFINITY
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
        double measured_phase = atan2(measured_imag, measured_real)
        double known_phase = atan2(known_imag, known_real)
        double measured_modulation = hypot(measured_real, measured_imag)
        double known_modulation = hypot(known_real, known_imag)

    if fabs(measured_modulation) == 0.0:
        return <float_t> (known_phase - measured_phase), <float_t> INFINITY
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
    """Return phasor coordinates on semicircle at other harmonic."""
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
    float_t real1,
    float_t imag1,
    float_t real2,
    float_t imag2,
) noexcept nogil:
    """Return multiplication of two phasors."""
    return real1 * real2 - imag1 * imag2, real1 * imag2 + imag1 * real2


@cython.ufunc
cdef (float_t, float_t) _phasor_divide(
    float_t real1,
    float_t imag1,
    float_t real2,
    float_t imag2,
) noexcept nogil:
    """Return division of two phasors."""
    cdef:
        float_t denom = real2 * real2 + imag2 * imag2

    if denom == 0.0:
        return NAN, NAN

    return (
        (real1 * real2 + imag1 * imag2) / denom,
        (imag1 * real2 - real1 * imag2) / denom
    )


###############################################################################
# Geometry ufuncs

@cython.ufunc
cdef short _is_inside_range(
    float_t x,  # point
    float_t y,
    float_t xmin,  # x range
    float_t xmax,
    float_t ymin,  # y range
    float_t ymax,
    short mask,
) noexcept nogil:
    """Return whether point is inside range.

    Range includes lower but not upper limit.

    """
    return mask and x >= xmin and x < xmax and y >= ymin and y < ymax


@cython.ufunc
cdef short _is_inside_rectangle(
    float_t x,  # point
    float_t y,
    float_t x0,  # segment start
    float_t y0,
    float_t x1,  # segment end
    float_t y1,
    float_t r,  # half width
    short mask,
) noexcept nogil:
    """Return whether point is in rectangle.

    The rectangle is defined by central line segment and half width.

    """
    cdef:
        float_t t

    if not mask or r <= 0.0:
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
cdef short _is_inside_polar_rectangle(
    float_t x,  # point
    float_t y,
    float_t angle_min,  # phase, -pi to pi
    float_t angle_max,
    float_t distance_min,  # modulation
    float_t distance_max,
    short mask,
) noexcept nogil:
    """Return whether point is inside polar rectangle.

    Angles should be in range -pi to pi, else performance is degraded.

    """
    cdef:
        double t

    if not mask:
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
cdef short _is_inside_circle(
    float_t x,  # point
    float_t y,
    float_t x0,  # circle center
    float_t y0,
    float_t r,  # circle radius
    short mask,
) noexcept nogil:
    """Return whether point is inside circle."""
    if not mask or r <= 0.0:
        return False
    x -= x0
    y -= y0
    return x * x + y * y <= r * r


@cython.ufunc
cdef short _is_inside_ellipse(
    float_t x,  # point
    float_t y,
    float_t x0,  # ellipse center
    float_t y0,
    float_t a,  # ellipse radii
    float_t b,
    float_t phi,  # ellipse angle
    short mask,
) noexcept nogil:
    """Return whether point is inside ellipse.

    Same as _is_inside_circle if a == b.
    Consider using _is_inside_ellipse_ instead, which should be faster
    for arrays.

    """
    cdef:
        float_t sina, cosa

    if not mask or a <= 0.0 or b <= 0.0:
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
cdef short _is_inside_ellipse_(
    float_t x,  # point
    float_t y,
    float_t x0,  # ellipse center
    float_t y0,
    float_t a,  # ellipse radii
    float_t b,
    float_t sina,  # sin/cos of ellipse angle
    float_t cosa,
    short mask,
) noexcept nogil:
    """Return whether point is inside ellipse.

    Use pre-calculated sin(angle) and cos(angle).

    """
    if not mask or a <= 0.0 or b <= 0.0:
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
cdef short _is_inside_stadium(
    float_t x,  # point
    float_t y,
    float_t x0,  # line start
    float_t y0,
    float_t x1,  # line end
    float_t y1,
    float_t r,  # radius
    short mask,
) noexcept nogil:
    """Return whether point is inside stadium.

    A stadium shape is a thick line with rounded ends.
    Same as _is_near_segment.

    """
    cdef:
        float_t t

    if not mask or r <= 0.0:
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
cdef short _is_near_line(
    float_t x,  # point
    float_t y,
    float_t x0,  # line start
    float_t y0,
    float_t x1,  # line end
    float_t y1,
    float_t r,  # distance
    short mask,
) noexcept nogil:
    """Return whether point is close to line."""
    cdef:
        float_t t

    if not mask or r <= 0.0:
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
cdef (double, double, double) _segment_direction_and_length(
    float_t x0,  # segment start
    float_t y0,
    float_t x1,  # segment end
    float_t y1,
) noexcept nogil:
    """Return direction and length of line segment."""
    cdef:
        float_t length

    x1 -= x0
    y1 -= y0
    length = <float_t> hypot(x1, y1)
    if length <= 0.0:
        return NAN, NAN, 0.0
    x1 /= length
    y1 /= length
    return x1, y1, length


@cython.ufunc
cdef (double, double, double, double) _intersection_circle_circle(
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
        ld * dx + hd * dy + x0,
        ld * dy - hd * dx + y0,
        ld * dx - hd * dy + x0,
        ld * dy + hd * dx + y0,
    )


@cython.ufunc
cdef (double, double, double, double) _intersection_circle_line(
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
        x + (dd * dy + copysign(1.0, dy) * dx * rdd) / dr,
        y + (-dd * dx + abs(dy) * rdd) / dr,
        x + (dd * dy - copysign(1.0, dy) * dx * rdd) / dr,
        y + (-dd * dx - abs(dy) * rdd) / dr,
    )

###############################################################################
# Blend ufuncs


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
