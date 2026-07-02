"""SpectralPlots class."""

from __future__ import annotations

__all__ = ['SpectralPlots']

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.lines import Line2D

    from .._typing import Any, ArrayLike, NDArray


import numpy
from matplotlib import pyplot
from matplotlib.ticker import MaxNLocator
from matplotlib.widgets import RangeSlider, Slider

from .._phasorpy import _gaussian_signal
from .._utils import update_kwargs
from ..experimental import signal_from_dho
from ..phasor import phasor_from_signal
from ..plot._phasorplot import CircleTicks, PhasorPlot


class SpectralPlots:
    r"""Plot spectra in wavelength and phasor space.

    Plot emission spectra and the corresponding spectral phasor coordinates
    for a set of spectral components and their mixture.

    By default, each spectral component is modelled as a Gaussian in
    wavelength space, where shifting the origin produces a pure phase
    rotation in the phasor without changing the modulation, provided the
    Gaussian fits within the wavelength window.
    Optionally (``dho=True``), the Displaced Harmonic Oscillator (DHO) model
    can be used to approximate the vibronic structure of real fluorophore
    spectra.
    Gaussian mode supports up to four components, while DHO mode is limited to
    two components.

    Parameters
    ----------
    origin : array_like
        Center wavelength of spectrum peak in nm.
        In Gaussian mode, the mean of the Gaussian.
        In DHO mode, the 0->0 electronic origin transition.
        Typically in the range 400 to 700 nm.
    sigma : array_like, optional
        Spectral broadening factor.
        In Gaussian mode, the standard deviation in nm (typically 10 to 50).
        In DHO mode, the broadening in :math:`cm^{-1}` (typically 200 to 600).
        Defaults to 30 for Gaussian or 500 :math:`cm^{-1}` for DHO.
    fraction : array_like, optional
        Fractional intensities of spectral components.
        Fractions are normalized to sum to 1.
        If not given, all components are assumed to have equal fractions.
    hr_factor : array_like, optional, default: 0.4
        Huang-Rhys structural coupling/displacement parameter (dimensionless).
        Broadcast to the number of components if scalar.
        Typically in the range 0.1 to 2.0.
        Only used for DHO model.
    vib_frequency : array_like, optional, default: 1200.0
        Vibrational spacing frequency in :math:`cm^{-1}`.
        Broadcast to the number of components if scalar.
        Typically in the range 1000 to 1600 :math:`cm^{-1}`.
        Only used for DHO model.
    wavelength_range : tuple[float, float], optional
        Wavelength window (min, max) in nm over which to display and compute
        phasor coordinates. The default is (400.0, 800.0).
    origin_range : tuple[float, float, float], optional
        Range of origin wavelengths in nm for origin sliders (min, max, step).
        The default is (450.0, 750.0, 1.0).
    sigma_range : tuple[float, float, float], optional
        Range for sigma sliders (min, max, step).
        Defaults to (10, 2000, 10) :math:`cm^{-1}` for DHO or (1, 200, 1) nm
        for Gaussian.
    hr_factor_range : tuple[float, float, float], optional
        Range of Huang-Rhys factor for slider (min, max, step).
        The default is (0.0, 2.0, 0.05).
    vib_frequency_range : tuple[float, float, float], optional
        Range of vibrational frequency in :math:`cm^{-1}` for slider
        (min, max, step). The default is (0.0, 2000.0, 10.0).
    samples : int, optional, default: 128
        Number of uniformly-spaced wavelength samples used to compute
        spectra and phasor coordinates.
    dho : bool, optional, default: False
        Use Displaced Harmonic Oscillator (DHO) instead of Gaussian model.
    interactive : bool, optional, default: False
        Add sliders to change parameters interactively.
    **kwargs
        Optional arguments passed to :py:func:`matplotlib.pyplot.figure`.

    See Also
    --------
    phasorpy.experimental.signal_from_dho
    :ref:`sphx_glr_tutorials_misc_phasorpy_apps.py`

    """

    _samples: int  # number of wavelength samples
    _wl_min: float  # current wavelength range min
    _wl_max: float  # current wavelength range max
    _dho: bool  # True → DHO model, False → Gaussian model

    _spectrum_plot: Axes
    _phasor_plot: Axes

    _wavelength_slider: RangeSlider
    _origin_sliders: list[Slider]
    _sigma_sliders: list[Slider]
    _fraction_sliders: list[Slider]
    _hr_factor_sliders: list[Slider]
    _vib_frequency_sliders: list[Slider]
    _samples_slider: Slider

    _spectrum_line: Line2D
    # _wl_min_line: Line2D
    # _wl_max_line: Line2D
    _phasor_point: Line2D
    _samples_polygon: Line2D | None
    _tick_line: Line2D | None
    _unitcircle_ticks: CircleTicks | None

    _component_spectrum_lines: list[Line2D]
    _component_phasor_points: list[Line2D]
    _phasor_lines: list[Line2D]

    _component_colors = (
        'tab:orange',
        'tab:green',
        'tab:purple',
        'tab:pink',
        'tab:olive',
        'tab:cyan',
        'tab:brown',
    )

    def __init__(
        self,
        origin: ArrayLike = 518.0,
        sigma: ArrayLike | None = None,
        fraction: ArrayLike | None = None,
        *,
        hr_factor: ArrayLike = 0.4,
        vib_frequency: ArrayLike = 1200.0,
        wavelength_range: tuple[float, float] = (400.0, 800.0),
        origin_range: tuple[float, float, float] = (450.0, 750.0, 1.0),
        sigma_range: tuple[float, float, float] | None = None,
        hr_factor_range: tuple[float, float, float] = (0.0, 2.0, 0.05),
        vib_frequency_range: tuple[float, float, float] = (0.0, 2000.0, 10.0),
        samples: int = 128,
        dho: bool = False,
        interactive: bool = False,
        **kwargs: Any,
    ) -> None:
        self._samples = int(samples)
        self._dho = bool(dho)
        origins = numpy.atleast_1d(numpy.asarray(origin, dtype=float))

        num_components = origins.size
        if num_components > (2 if dho else 4):
            msg = f'{num_components=} > {2 if dho else 4}'
            raise ValueError(msg)

        if sigma is None:
            sigmas = numpy.full(num_components, 500.0 if dho else 30.0)
        else:
            sigmas = numpy.atleast_1d(numpy.asarray(sigma, dtype=float))
            if sigmas.size == 1:
                sigmas = numpy.full(num_components, sigmas[0])
            elif sigmas.size != num_components:
                msg = f'{sigmas.size=} != {num_components=}'
                raise ValueError(msg)

        if sigma_range is None:
            sigma_range = (10.0, 2000.0, 10.0) if dho else (1.0, 200.0, 1.0)

        hr_factors = numpy.atleast_1d(numpy.asarray(hr_factor, dtype=float))
        if hr_factors.size == 1:
            hr_factors = numpy.full(num_components, hr_factors[0])
        elif hr_factors.size != num_components:
            msg = f'{hr_factors.size=} != {num_components=}'
            raise ValueError(msg)

        vib_frequencies = numpy.atleast_1d(
            numpy.asarray(vib_frequency, dtype=float)
        )
        if vib_frequencies.size == 1:
            vib_frequencies = numpy.full(num_components, vib_frequencies[0])
        elif vib_frequencies.size != num_components:
            msg = f'{vib_frequencies.size=} != {num_components=}'
            raise ValueError(msg)

        if fraction is None:
            fractions = numpy.ones(num_components) / num_components
        else:
            fractions = numpy.asarray(fraction, dtype=float)
            if fractions.size != num_components:
                msg = f'{fractions.size=} != {num_components=}'
                raise ValueError(msg)
            s = fractions.sum()
            if s > 0.0:
                fractions = numpy.clip(fractions / s, 0.0, 1.0)
            else:
                fractions = numpy.ones(num_components) / num_components

        wl_min, wl_max = float(wavelength_range[0]), float(wavelength_range[1])

        # pre-compute slider counts so figsize and layout can be derived
        # before the figure is created
        num_fraction = len(
            [
                i
                for i in range(num_components)
                if not (
                    num_components == 1 or (i == 1 and num_components == 2)
                )
            ]
        )
        num_left = 2  # wavelength + samples
        num_right = (4 if dho else 2) * num_components + num_fraction
        if interactive:
            max_rows = max(num_left, num_right)
            slider_row_in = 0.3  # inches per slider row (height + spacing)
            bottom_in = 0.25  # margin below the last slider
            subplot_in = 4.8  # fixed subplot area height in inches
            fig_height_in = subplot_in + max_rows * slider_row_in + bottom_in
            y_step = slider_row_in / fig_height_in
            y_start = (
                (max_rows - 1) * slider_row_in + bottom_in
            ) / fig_height_in
            # bottom of the subplot area, matching the dynamic figsize
            rect_bottom = y_start + y_step
        else:
            fig_height_in = 5.12

        (
            wavelengths,
            signal,
            component_signals,
            component_real,
            component_imag,
            real,
            imag,
        ) = self._calculate(
            wl_min,
            wl_max,
            origins,
            sigmas,
            fractions,
            hr_factors,
            vib_frequencies,
        )

        # create figure
        update_kwargs(kwargs, figsize=(10.24, fig_height_in))
        fig = pyplot.figure(**kwargs)
        spectrum_plot, phasor_plot = fig.subplots(
            1, 2, gridspec_kw={'width_ratios': [2.45, 2]}
        )

        if interactive:
            fcm = fig.canvas.manager
            if fcm is not None:
                fcm.set_window_title('PhasorPy spectral plots')

        self._component_spectrum_lines = []
        self._component_phasor_points = []
        self._phasor_lines = []

        spectrum_plot.set_title('Spectrum')
        spectrum_plot.set_xlabel('Wavelength [nm]')
        spectrum_plot.set_ylabel('Intensity [normalized]')

        if num_components > 1:
            for i in range(num_components):
                lines = spectrum_plot.plot(
                    wavelengths,
                    component_signals[i],
                    color=self._component_colors[i],
                    linewidth=0.8,
                    alpha=0.5,
                    label=f'Component {i}',
                )
                self._component_spectrum_lines.append(lines[0])

        lines = spectrum_plot.plot(
            wavelengths,
            signal,
            label='Mixture',
            color='tab:blue',
            linewidth=2,
            zorder=10,
        )
        self._spectrum_line = lines[0]

        # lines = spectrum_plot.axvline(
        #     wl_min, color='gray', linestyle='--', linewidth=0.8, alpha=0.5
        # )
        # self._wl_min_line = lines
        # lines = spectrum_plot.axvline(
        #     wl_max, color='gray', linestyle='--', linewidth=0.8, alpha=0.5
        # )
        # self._wl_max_line = lines
        if num_components > 1:
            spectrum_plot.legend(loc='upper right')

        # pass wavelength ticks into PhasorPlot's polar grid
        tick_real, tick_imag, tick_labels = self._phasor_wavelength_ticks(
            wl_min, wl_max
        )
        phasorplot = PhasorPlot(
            ax=phasor_plot,
            allquadrants=True,
            pad=0.3,
            grid={
                'ticks': numpy.arctan2(tick_imag, tick_real),
                'labels': tick_labels,
                'radii': 2,
                'angles': 8,
                # 'samples': self._samples,
            },
            title='Phasor plot',
        )

        self._unitcircle_ticks = phasorplot._unitcircle_ticks  # noqa: SLF001
        self._tick_line = phasorplot._unitcircle_tick_line  # noqa: SLF001

        if num_components > 1:
            for i in range(num_components):
                lines = phasorplot.plot(
                    (real, component_real[i]),
                    (imag, component_imag[i]),
                    color=self._component_colors[i],
                    linestyle='-',
                    linewidth=0.8,
                    alpha=0.5,
                )
                self._phasor_lines.append(lines[0])
                lines = phasorplot.plot(
                    component_real[i],
                    component_imag[i],
                    'o',
                    color=self._component_colors[i],
                )
                self._component_phasor_points.append(lines[0])

        lines = phasorplot.plot(
            real, imag, 'o', color='tab:blue', markersize=10, zorder=10
        )
        self._phasor_point = lines[0]

        # inscribed polygon: vertices on unit circle at DFT sample positions.
        # visible only when samples is small enough to see the discretisation
        poly_angles = numpy.linspace(
            0, 2.0 * math.pi, self._samples, endpoint=False
        )
        poly_angles = numpy.append(poly_angles, poly_angles[0])  # close
        lines = phasor_plot.plot(
            numpy.cos(poly_angles),
            numpy.sin(poly_angles),
            color='0.5',
            linestyle=':',
            linewidth=0.5,
            zorder=2,
            visible=self._samples <= 16,
        )
        self._samples_polygon = lines[0]

        self._spectrum_plot = spectrum_plot
        self._phasor_plot = phasor_plot
        self._wl_min = wl_min
        self._wl_max = wl_max

        if interactive:
            # restrict subplot area to the top, leaving room for sliders
            fig.tight_layout(rect=(0, rect_bottom, 1, 1))
        else:
            fig.tight_layout()

        if not interactive:
            return

        # add sliders
        slider_height = 0.01
        left_axes_iter = iter(
            fig.add_axes((0.13, y_start - i * y_step, 0.25, slider_height))
            for i in range(num_left)
        )
        right_axes_iter = iter(
            fig.add_axes((0.65, y_start - i * y_step, 0.25, slider_height))
            for i in range(num_right)
        )

        self._wavelength_slider = RangeSlider(
            ax=next(left_axes_iter),
            label='Wavelengths ',
            valfmt='%.0f nm',
            valmin=350.0,
            valmax=850.0,
            valstep=1.0,
            valinit=(wl_min, wl_max),
        )
        self._wavelength_slider.on_changed(self._on_changed)

        self._samples_slider = Slider(
            ax=next(left_axes_iter),
            label='Samples ',
            valfmt=' %.0f',
            valmin=3.0,
            valmax=256.0,
            valstep=1.0,
            valinit=float(self._samples),
        )
        self._samples_slider.on_changed(self._on_changed)

        self._origin_sliders = []
        for i, (valinit, color) in enumerate(
            zip(origins, self._component_colors[:num_components], strict=True)
        ):
            slider = Slider(
                ax=next(right_axes_iter),
                label=f'Origin {i} ',
                valfmt=' %.0f nm',
                valmin=origin_range[0],
                valmax=origin_range[1],
                valstep=origin_range[2],
                valinit=valinit,
                facecolor=color,
            )
            slider.on_changed(self._on_changed)
            self._origin_sliders.append(slider)

        self._sigma_sliders = []
        for i, (valinit, color) in enumerate(
            zip(sigmas, self._component_colors[:num_components], strict=True)
        ):
            slider = Slider(
                ax=next(right_axes_iter),
                label=f'Sigma {i} ',
                valfmt=' %.0f',
                valmin=sigma_range[0],
                valmax=sigma_range[1],
                valstep=sigma_range[2],
                valinit=valinit,
                facecolor=color,
            )
            slider.on_changed(self._on_changed)
            self._sigma_sliders.append(slider)

        self._fraction_sliders = []
        for i, (valinit, color) in enumerate(
            zip(
                fractions,
                self._component_colors[:num_components],
                strict=True,
            )
        ):
            if num_components == 1 or (i == 1 and num_components == 2):
                break
            slider = Slider(
                ax=next(right_axes_iter),
                label=f'Fraction {i} ',
                valfmt=' %.2f',
                valmin=0.0,
                valmax=1.0,
                valstep=0.01,
                valinit=valinit,
                facecolor=color,
            )
            slider.on_changed(self._on_changed)
            self._fraction_sliders.append(slider)

        self._hr_factor_sliders = []
        self._vib_frequency_sliders = []
        if dho:
            for i, (valinit, color) in enumerate(
                zip(
                    hr_factors,
                    self._component_colors[:num_components],
                    strict=True,
                )
            ):
                slider = Slider(
                    ax=next(right_axes_iter),
                    label=f'HR factor {i} ',
                    valfmt=' %.2f',
                    valmin=hr_factor_range[0],
                    valmax=hr_factor_range[1],
                    valstep=hr_factor_range[2],
                    valinit=valinit,
                    facecolor=color,
                )
                slider.on_changed(self._on_changed)
                self._hr_factor_sliders.append(slider)

            for i, (valinit, color) in enumerate(
                zip(
                    vib_frequencies,
                    self._component_colors[:num_components],
                    strict=True,
                )
            ):
                slider = Slider(
                    ax=next(right_axes_iter),
                    label=f'Vib. freq. {i} ',
                    valfmt=' %.0f',
                    valmin=vib_frequency_range[0],
                    valmax=vib_frequency_range[1],
                    valstep=vib_frequency_range[2],
                    valinit=valinit,
                    facecolor=color,
                )
                slider.on_changed(self._on_changed)
                self._vib_frequency_sliders.append(slider)

    def _calculate(
        self,
        wl_min: float,
        wl_max: float,
        origins: NDArray[Any],
        sigmas: NDArray[Any],
        fractions: NDArray[Any],
        hr_factors: NDArray[Any],
        vib_frequencies: NDArray[Any],
        /,
    ) -> tuple[
        NDArray[Any],  # wavelengths
        NDArray[Any],  # signal (mixed, normalized)
        NDArray[Any],  # component_signals (N x samples, fraction-weighted)
        NDArray[Any],  # component_real
        NDArray[Any],  # component_imag
        float,  # real (mixed phasor)
        float,  # imag (mixed phasor)
    ]:
        """Return values for plotting."""
        wavelengths = numpy.linspace(wl_min, wl_max, self._samples)
        num_components = origins.size

        if self._dho:
            component_spectra = numpy.stack(
                [
                    signal_from_dho(
                        wavelengths,
                        float(origins[i]),
                        float(sigmas[i]),
                        float(hr_factors[i]),
                        float(vib_frequencies[i]),
                    )
                    for i in range(num_components)
                ]
            )
        else:
            # Gaussian model
            span = wl_max - wl_min
            scale = (self._samples - 1) / span if span > 0 else 1.0
            spectra = []
            for i in range(num_components):
                spec = numpy.zeros(self._samples, dtype=numpy.float64)
                mean_s = (float(origins[i]) - wl_min) * scale
                stdev_s = float(sigmas[i]) * scale
                _gaussian_signal(spec, mean_s, stdev_s, folds=0)
                spectra.append(spec)
            component_spectra = numpy.stack(spectra)

        # normalize fractions to sum to 1
        fractions = numpy.asarray(fractions, dtype=numpy.float64)
        s = fractions.sum()
        if s > 0.0:
            fractions = numpy.clip(fractions / s, 0.0, 1.0)
        else:
            fractions = numpy.ones(num_components) / num_components

        # weight spectra by fraction
        component_signals = component_spectra * fractions[:, numpy.newaxis]

        # mixed spectrum
        signal = component_signals.sum(axis=0)

        # normalize mixed spectrum to peak = 1
        peak = signal.max()
        if peak > 0.0:
            signal = signal / peak
            component_signals = component_signals / peak

        # phasor of mixed spectrum
        _, real_arr, imag_arr = phasor_from_signal(signal)
        real = float(real_arr)
        imag = float(imag_arr)

        # phasors of individual (unweighted) component spectra
        component_real = numpy.empty(num_components)
        component_imag = numpy.empty(num_components)
        for i in range(num_components):
            spec = component_spectra[i]
            _, cr, ci = phasor_from_signal(spec)
            component_real[i] = float(cr)
            component_imag[i] = float(ci)

        return (
            wavelengths,
            signal,
            component_signals,
            component_real,
            component_imag,
            real,
            imag,
        )

    def _phasor_wavelength_ticks(
        self,
        wl_min: float,
        wl_max: float,
        /,
    ) -> tuple[NDArray[Any], NDArray[Any], list[str]]:
        """Return unit-circle (x, y) and labels for wavelength tick marks."""
        tick_space = numpy.linspace(wl_min, wl_max, self._samples)
        # use matplotlib's auto-locator to pick nice tick wavelengths
        # TODO: move this method to PhasorPlot class?
        # TODO: should we limit this to samples > 8?
        locator = MaxNLocator(nbins=8)
        ticks = numpy.asarray(locator.tick_values(wl_min, wl_max))
        ticks = ticks[(ticks >= wl_min) & (ticks <= wl_max)]
        # linear mapping from wavelength range to [0, 2pi)
        divisor = tick_space[-1] + tick_space[1] - 2.0 * tick_space[0]
        angles = (ticks - tick_space[0]) / divisor * 2.0 * math.pi
        labels = [
            # skip label for ticks near 360 to avoid overlap with 0 label
            '' if (a % (2.0 * math.pi)) > math.radians(355) else f'{t:.0f}'
            for t, a in zip(ticks, angles, strict=True)
        ]
        return numpy.cos(angles), numpy.sin(angles), labels

    def _on_changed(self, value: Any) -> None:
        """Update plot with current slider values."""
        del value  # unused

        wl_min, wl_max = self._wavelength_slider.val
        self._samples = int(self._samples_slider.val)

        origins = numpy.asarray([s.val for s in self._origin_sliders])
        sigmas = numpy.asarray([s.val for s in self._sigma_sliders])
        fractions_raw = numpy.asarray([s.val for s in self._fraction_sliders])
        hr_factors = numpy.asarray([s.val for s in self._hr_factor_sliders])
        vib_frequencies = numpy.asarray(
            [s.val for s in self._vib_frequency_sliders]
        )
        if not self._dho:
            # pass empty arrays since not used by _calculate in Gaussian mode
            hr_factors = numpy.empty(origins.size)
            vib_frequencies = numpy.empty(origins.size)

        num_components = origins.size
        if num_components == 1:
            fractions = numpy.asarray([1.0])
        elif num_components == 2:
            fractions = numpy.asarray(
                [fractions_raw[0], 1.0 - fractions_raw[0]]
            )
        else:
            fractions = fractions_raw

        (
            wavelengths,
            signal,
            component_signals,
            component_real,
            component_imag,
            real,
            imag,
        ) = self._calculate(
            wl_min,
            wl_max,
            origins,
            sigmas,
            fractions,
            hr_factors,
            vib_frequencies,
        )

        # update spectrum plot
        self._spectrum_line.set_data(wavelengths, signal)
        for i, line in enumerate(self._component_spectrum_lines):
            line.set_data(wavelengths, component_signals[i])
        # self._wl_min_line.set_xdata([wl_min, wl_min])
        # self._wl_max_line.set_xdata([wl_max, wl_max])
        self._spectrum_plot.relim()
        self._spectrum_plot.autoscale_view()

        # update phasor plot
        self._phasor_point.set_data([real], [imag])
        for i, (pt, ln) in enumerate(
            zip(
                self._component_phasor_points,
                self._phasor_lines,
                strict=True,
            )
        ):
            pt.set_data([component_real[i]], [component_imag[i]])
            ln.set_data([real, component_real[i]], [imag, component_imag[i]])

        # update inscribed polygon
        if self._samples_polygon is not None:
            if self._samples <= 16:
                poly_angles = numpy.linspace(
                    0, 2.0 * math.pi, self._samples, endpoint=False
                )
                poly_angles = numpy.append(poly_angles, poly_angles[0])
                self._samples_polygon.set_data(
                    numpy.cos(poly_angles), numpy.sin(poly_angles)
                )
                self._samples_polygon.set_visible(True)
            else:
                self._samples_polygon.set_visible(False)

        # update wavelength tick labels and positions
        if self._unitcircle_ticks is not None and self._tick_line is not None:
            tick_real, tick_imag, tick_labels = self._phasor_wavelength_ticks(
                wl_min, wl_max
            )
            self._unitcircle_ticks.labels = tick_labels
            self._tick_line.set_data(tick_real, tick_imag)
            self._wl_min = wl_min
            self._wl_max = wl_max

        self._spectrum_plot.figure.canvas.draw_idle()

    def show(self) -> None:
        """Show all open figures. Call :py:func:`matplotlib.pyplot.show`."""
        pyplot.show()
