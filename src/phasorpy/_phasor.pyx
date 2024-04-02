# distutils: language = c
# cython: language_level = 3
# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: nonecheck = False
# cython: linetrace = True

"""Cython implementation of low-level functions for the PhasorPy package."""

# these lines must be at top of module
# TODO: https://github.com/cython/cython/issues/6064

cimport numpy

numpy.import_array()
numpy.import_ufunc()

cimport cython

from cython.parallel import parallel, prange

from libc.math cimport (
    INFINITY,
    M_PI,
    NAN,
    atan,
    atan2,
    cos,
    fabs,
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
        double twopi = 2.0 * M_PI * unit_conversion
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

    # clamp fractions to 0..1
    fret_efficiency = clamp(fret_efficiency)
    donor_freting = clamp(donor_freting)
    donor_background = clamp(donor_background)

    # phasor of pure donor at frequency
    real, imag = phasor_from_lifetime(donor_lifetime, omega)

    if fret_efficiency > 0.0:
        # phasor of quenched donor
        quenched_real, quenched_imag = phasor_from_lifetime(
            donor_lifetime * (1.0 -  fret_efficiency), omega
        )

        # phasor of pure and quenched donor
        real, imag = linear_combination(
            1.0,
            0.0,
            real,
            imag,
            quenched_real,
            quenched_imag,
            1.0,
            1.0 - fret_efficiency,
            1.0 - donor_freting
        )

    if donor_background > 0.0:
        # phasor of pure and quenched donor with background
        real, imag = linear_combination(
            real,
            imag,
            real,
            imag,
            background_real,
            background_imag,
            1.0 - donor_freting * fret_efficiency,
            1.0,
            1.0 - donor_background
        )

    return real, imag


@cython.ufunc
cdef (double, double) _phasor_from_fret_acceptor(
    double omega,
    double donor_lifetime,
    double acceptor_lifetime,
    double fret_efficiency,
    double donor_freting,
    double donor_bleedthrough,
    double acceptor_excitation,
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

    # clamp fractions to 0..1
    fret_efficiency = clamp(fret_efficiency)
    donor_freting = clamp(donor_freting)
    donor_bleedthrough = clamp(donor_bleedthrough)
    acceptor_excitation = clamp(acceptor_excitation)
    acceptor_background = clamp(acceptor_background)

    # phasor of pure donor at frequency
    donor_real, donor_imag = phasor_from_lifetime(donor_lifetime, omega)

    if fret_efficiency == 0.0:
        quenched_real = donor_real
        quenched_imag = donor_imag
    else:
        # phasor of quenched donor
        quenched_real, quenched_imag = phasor_from_lifetime(
            donor_lifetime * (1.0 -  fret_efficiency), omega
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
        sqrt(
            quenched_real * quenched_real
            + quenched_imag * quenched_imag
        )
        * sqrt(
            acceptor_real * acceptor_real
            + acceptor_imag * acceptor_imag
        )
    )
    sensitized_real = mod * cos(phi)
    sensitized_imag = mod * sin(phi)

    # phasor of acceptor excited by quenched donor and directly
    acceptor_real, acceptor_imag = linear_combination(
        sensitized_real,
        sensitized_imag,
        sensitized_real,
        sensitized_imag,
        acceptor_real,
        acceptor_imag,
        donor_freting * fret_efficiency,  # SimFCS uses 1.0
        1.0,
        1.0 - acceptor_excitation
    )

    # phasor of excited acceptor with background
    if acceptor_background > 0.0:
        acceptor_real, acceptor_imag = linear_combination(
            acceptor_real,
            acceptor_imag,
            acceptor_real,
            acceptor_imag,
            background_real,
            background_imag,
            # SimFCS uses 1.0
            donor_freting * fret_efficiency + acceptor_excitation,
            1.0,
            1.0 - acceptor_background
        )

    # phasor of excited acceptor with background and donor bleedthrough
    if donor_bleedthrough > 0.0:
        # SimFCS also includes donor channel background in donor bleedthrough
        acceptor_real, acceptor_imag = linear_combination(
            acceptor_real,
            acceptor_imag,
            acceptor_real,
            acceptor_imag,
            donor_real,
            donor_imag,
            (
                donor_freting * fret_efficiency
                + acceptor_excitation + acceptor_background
            ),  # SimFCS uses 1.0
            1.0 - donor_freting * fret_efficiency,
            1.0 - donor_bleedthrough
        )

    return acceptor_real, acceptor_imag


cdef inline double clamp(const double value) noexcept nogil:
    """Return value between 0.0 and 1.0."""
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


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


cdef inline (double, double) phasor_from_lifetime(
    float_t lifetime,
    float_t omega,
) noexcept nogil:
    """Return phasor coordinates from single lifetime component."""
    cdef:
        double t = omega * lifetime
        double mod = 1.0 / sqrt(1.0 + t * t)
        double phi = atan(t)

    return mod * cos(phi), mod * sin(phi)


cdef inline (double, double) phasor_to_apparent_lifetime(
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

    return tauphi, taumod


cdef inline (double, double) phasor_from_apparent_lifetime(
    float_t tauphi,
    float_t taumod,
    float_t omega,
) noexcept nogil:
    """Return phasor coordinates from apparent single lifetimes."""
    cdef:
        double t = omega * taumod
        double mod = 1.0 / sqrt(1.0 + t * t)
        double phi = atan(omega * tauphi)

    return mod * cos(phi), mod * sin(phi)


@cython.ufunc
cdef (double, double) _phasor_transform(
    float_t real,
    float_t imag,
    float_t angle,
    float_t scale,
) noexcept nogil:
    """Return rotated and scaled phasor coordinates."""
    cdef:
        double g = scale * cos(angle)
        double s = scale * sin(angle)

    return real * g - imag * s, real * s + imag * g


@cython.ufunc
cdef (double, double) _phasor_transform_const(
    float_t real,
    float_t imag,
    float_t real2,
    float_t imag2,
) noexcept nogil:
    """Return rotated and scaled phasor coordinates."""
    return real * real2 - imag * imag2, real * imag2 + imag * real2


@cython.ufunc
cdef (double, double) _phasor_to_polar(
    float_t real,
    float_t imag,
) noexcept nogil:
    """Return polar from phasor coordinates."""
    return atan2(imag, real), sqrt(real * real + imag * imag)


@cython.ufunc
cdef (double, double) _phasor_from_polar(
    float_t phase,
    float_t modulation,
) noexcept nogil:
    """Return phasor from polar coordinates."""
    return modulation * cos(phase), modulation * sin(phase)


@cython.ufunc
cdef (double, double) _phasor_to_apparent_lifetime(
    float_t real,
    float_t imag,
    float_t omega,
) noexcept nogil:
    """Return apparent single lifetimes from phasor coordinates."""
    return phasor_to_apparent_lifetime(real, imag, omega)


@cython.ufunc
cdef (double, double) _phasor_from_apparent_lifetime(
    float_t tauphi,
    float_t taumod,
    float_t omega,
) noexcept nogil:
    """Return phasor coordinates from apparent single lifetimes."""
    return phasor_from_apparent_lifetime(tauphi, taumod, omega)


@cython.ufunc
cdef (double, double) _phasor_from_single_lifetime(
    float_t lifetime,
    float_t omega,
) noexcept nogil:
    """Return phasor coordinates from single lifetime component."""
    cdef:
        double t = omega * lifetime
        double phi = atan(t)
        double mod = 1.0 / sqrt(1.0 + t * t)

    return mod * cos(phi), mod * sin(phi)


@cython.ufunc
cdef (double, double) _polar_from_single_lifetime(
    float_t lifetime,
    float_t omega,
) noexcept nogil:
    """Return polar coordinates from single lifetime component."""
    cdef:
        double t = omega * lifetime

    return atan(t), 1.0 / sqrt(1.0 + t * t)


@cython.ufunc
cdef (double, double) _polar_to_apparent_lifetime(
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
    return tauphi, taumod


@cython.ufunc
cdef (double, double) _polar_from_apparent_lifetime(
    float_t tauphi,
    float_t taumod,
    float_t omega,
) noexcept nogil:
    """Return polar coordinates from apparent single lifetimes."""
    cdef:
        double t = omega * taumod

    return atan(omega * tauphi), 1.0 / sqrt(1.0 + t * t)


@cython.ufunc
cdef (double, double) _polar_from_reference(
    float_t measured_phase,
    float_t measured_modulation,
    float_t known_phase,
    float_t known_modulation,
) noexcept nogil:
    """Return polar coordinates for calibration from reference coordinates."""
    if fabs(measured_modulation) == 0.0:
        return known_phase - measured_phase, INFINITY
    return known_phase - measured_phase, known_modulation / measured_modulation


@cython.ufunc
cdef (double, double) _polar_from_reference_phasor(
    float_t measured_real,
    float_t measured_imag,
    float_t known_real,
    float_t known_imag,
) noexcept nogil:
    """Return polar coordinates for calibration from reference phasor."""
    cdef:
        double measured_phase = atan2(measured_imag, measured_real)
        double known_phase = atan2(known_imag, known_real)
        double measured_modulation = sqrt(
            measured_real * measured_real + measured_imag * measured_imag
        )
        double known_modulation = sqrt(
            known_real * known_real + known_imag * known_imag
        )

    if fabs(measured_modulation) == 0.0:
        return known_phase - measured_phase, INFINITY
    return known_phase - measured_phase, known_modulation / measured_modulation


@cython.ufunc
cdef float_t _fractional_intensity_to_preexponential_amplitude(
    float_t fractional_intensity,
    float_t lifetime,
    float_t mean,
) noexcept nogil:
    """Return preexponential amplitude from fractional intensity."""
    if fabs(mean) == 0.0:
        return INFINITY
    return fractional_intensity * lifetime / mean


@cython.ufunc
cdef float_t _fractional_intensity_from_preexponential_amplitude(
    float_t preexponential_amplitude,
    float_t lifetime,
    float_t mean,
) noexcept nogil:
    """Return fractional intensity from preexponential amplitude."""
    if fabs(lifetime) == 0.0:
        return INFINITY
    return preexponential_amplitude / lifetime * mean
