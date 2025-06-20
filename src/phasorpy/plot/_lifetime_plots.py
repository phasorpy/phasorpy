"""LifetimePlots class."""

from __future__ import annotations

__all__ = ['LifetimePlots']

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .._typing import Any, NDArray, ArrayLike

    from matplotlib.axes import Axes

import numpy
from matplotlib import pyplot
from matplotlib.lines import Line2D
from matplotlib.widgets import Slider

from .._utils import update_kwargs
from ..phasor import (
    lifetime_to_frequency,
    lifetime_to_signal,
    phasor_from_lifetime,
    phasor_to_polar,
    phasor_transform,
)
from ..plot._phasorplot import (
    PhasorPlot,
    SemicircleTicks,
    _semicircle_ticks,
)


class LifetimePlots:
    """Plot lifetimes in time domain, frequency domain, and phasor plot.

    Plot the time domain signals, phasor coordinates, and multi-frequency
    phase and modulation curves for a set of lifetime components and their
    mixture at given frequency and fractional intensities.

    Parameters
    ----------
    frequency : float
        Fundamental laser pulse or modulation frequency in MHz.
        If None, an optimal frequency is calculated from the mean of the
        lifetime components.
    lifetime : array_like
        Lifetime components in ns. Up to 6 components are supported.
    fraction : array_like, optional
        Fractional intensities of lifetime components.
        Fractions are normalized to sum to 1.
        If not given, all components are assumed to have equal fractions.
    frequency_range : tuple[float, float, float], optional
        Range of frequencies in MHz for frequency slider.
        Default is (10.0, 200.0, 1.0).
    lifetime_range : tuple[float, float, float], optional
        Range of lifetimes in ns for lifetime sliders.
        Default is (0.0, 20.0, 0.1).
    interactive: bool
        If True, add sliders to change frequency and lifetimes interactively.
        Default is False.
    **kwargs:
        Additional arguments passed to matplotlib figure.

    """

    _samples: int = 256  # number of frequencies and samples in signal
    _frequency: float  # current frequency in MHz
    _zero_phase: float | None = None  # location of IRF peak in the phase
    _zero_stdev: float | None = None  # standard deviation of IRF in radians
    _frequencies: NDArray[Any]  # for frequency domain plot

    _time_plot: Axes
    _phasor_plot: Axes
    _phase_plot: Axes
    _modulation_plot: Axes

    _frequency_slider: Slider
    _lifetime_sliders: list[Slider]
    _fraction_sliders: list[Slider]

    _signal_line: Line2D
    _frequency_line: Line2D
    _phase_point: Line2D
    _modulation_point: Line2D

    _signal_lines: list[Line2D]
    _phasor_lines: list[Line2D]
    _phasor_points: list[Line2D]
    _phase_lines: list[Line2D]
    _modulation_lines: list[Line2D]

    _semicircle_line: Line2D
    _semicircle_ticks: SemicircleTicks | None

    _component_colors = (
        # 'tab:blue',  # main
        # 'tab:red',  # modulation
        # 'tab:gray',  # irf
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
        frequency: float | None,
        lifetime: ArrayLike,
        fraction: ArrayLike | None = None,
        *,
        frequency_range: tuple[float, float, float] = (10.0, 200.0, 1.0),
        lifetime_range: tuple[float, float, float] = (0.0, 20.0, 0.1),
        interactive: bool = False,
        **kwargs: Any,
    ) -> None:
        self._frequencies = numpy.logspace(-1, 4, self._samples)

        (
            frequency,
            lifetimes,
            fractions,
            signal,
            irf,
            times,
            real,
            imag,
            phase,
            modulation,
            phase_,
            modulation_,
            component_signal,
            component_real,
            component_imag,
            component_phase_,
            component_modulation_,
        ) = self._calculate(frequency, lifetime, fraction)

        self._frequency = frequency

        num_components = max(lifetimes.size, 1)
        if num_components > 6:
            raise ValueError(f'too many components {num_components} > 6')

        # create plots
        update_kwargs(kwargs, figsize=(10.24, 7.68))
        fig, ((time_plot, phasor_plot), (phase_plot, ax4)) = pyplot.subplots(
            2, 2, **kwargs
        )

        if interactive:
            fcm = fig.canvas.manager
            if fcm is not None:
                fcm.set_window_title('PhasorPy lifetime plots')

        self._signal_lines = []
        self._phasor_lines = []
        self._phasor_points = []
        self._phase_lines = []
        self._modulation_lines = []

        # time domain plot
        time_plot.set_title('Time domain')
        time_plot.set_xlabel('Time [ns]')
        time_plot.set_ylabel('Intensity [normalized]')
        lines = time_plot.plot(
            times, signal, label='Signal', color='tab:blue', lw=2, zorder=10
        )
        self._signal_lines.append(lines[0])
        if num_components > 1:
            for i in range(num_components):
                lines = time_plot.plot(
                    times,
                    component_signal[i],
                    label=f'Lifetime {i}',
                    color=self._component_colors[i],
                    lw=0.8,
                    alpha=0.5,
                )
                self._signal_lines.append(lines[0])
        lines = time_plot.plot(
            times,
            irf,
            label='Instrument response',
            color='tab:grey',
            lw=0.8,
            alpha=0.5,
        )
        self._signal_lines.append(lines[0])
        time_plot.legend()

        # phasor plot
        phasorplot = PhasorPlot(ax=phasor_plot)
        lines = phasorplot.semicircle(frequency)
        self._semicircle_line = lines[0]
        self._semicircle_ticks = phasorplot._semicircle_ticks
        lines = phasorplot.plot(
            real, imag, 'o', color='tab:blue', markersize=10, zorder=10
        )
        self._phasor_points.append(lines[0])
        if num_components > 1:
            for i in range(num_components):
                lines = phasorplot.plot(
                    [real, component_real[i]],
                    [imag, component_imag[i]],
                    color=self._component_colors[i],
                    ls='-',
                    lw=0.8,
                    alpha=0.5,
                )
                self._phasor_lines.append(lines[0])
                lines = phasorplot.plot(
                    component_real[i],
                    component_imag[i],
                    'o',
                    color=self._component_colors[i],
                )
                self._phasor_points.append(lines[0])

        # frequency domain plot
        phase_plot.set_title('Frequency domain')
        phase_plot.set_xscale('log', base=10)
        phase_plot.set_xlabel('Frequency (MHz)')
        phase_plot.set_ylabel('Phase (°)', color='tab:blue')
        phase_plot.set_yticks([0.0, 30.0, 60.0, 90.0])
        phase_plot.plot([1, 1], [0.0, 90.0], alpha=0.0)  # set autoscale
        lines = phase_plot.plot(
            [frequency, frequency],
            [0, 90],
            '--',
            color='gray',
            lw=0.8,
            alpha=0.5,
        )
        self._frequency_line = lines[0]
        lines = phase_plot.plot(
            frequency, phase, 'o', color='tab:blue', markersize=8, zorder=2
        )
        self._phase_point = lines[0]
        lines = phase_plot.plot(
            self._frequencies, phase_, color='tab:blue', lw=2, zorder=2
        )
        self._phase_lines.append(lines[0])
        if num_components > 1:
            for i in range(num_components):
                lines = phase_plot.plot(
                    self._frequencies,
                    component_phase_[i],
                    color=self._component_colors[i],
                    lw=0.5,
                    alpha=0.5,
                )
                self._phase_lines.append(lines[0])
        # phase_plot.text(0.1, 1, 'Phase', ha='left', va='bottom')

        # TODO: zorder doesn't work.
        # twinx modulation_plot is always plotted on top of phase_plot
        modulation_plot = phase_plot.twinx()
        modulation_plot.set_ylabel('Modulation (%)', color='tab:red')
        modulation_plot.set_yticks([0.0, 25.0, 50.0, 75.0, 100.0])
        modulation_plot.plot([1, 1], [0.0, 100.0], alpha=0.0)  # set autoscale
        lines = modulation_plot.plot(
            frequency, modulation, 'o', color='tab:red', markersize=8, zorder=2
        )
        self._modulation_point = lines[0]
        lines = modulation_plot.plot(
            self._frequencies, modulation_, color='tab:red', lw=2, zorder=2
        )
        self._modulation_lines.append(lines[0])
        if num_components > 1:
            for i in range(num_components):
                lines = modulation_plot.plot(
                    self._frequencies,
                    component_modulation_[i],
                    color=self._component_colors[i],
                    lw=0.5,
                    alpha=0.5,
                )
                self._modulation_lines.append(lines[0])
        # modulation_plot.text(0.1, 98, 'Modulation', ha='left', va='top')

        ax4.axis('off')
        self._time_plot = time_plot
        self._phasor_plot = phasor_plot
        self._phase_plot = phase_plot
        self._modulation_plot = modulation_plot

        fig.tight_layout()

        if not interactive:
            return

        # add sliders
        axes = (
            fig.add_axes((0.65, 0.45 - i * 0.035, 0.25, 0.01))
            for i in range(1 + 2 * num_components)
        )

        self._frequency_slider = Slider(
            ax=next(axes),
            label='Frequency ',
            valfmt=' %.0f MHz',
            valmin=frequency_range[0],
            valmax=frequency_range[1],
            valstep=frequency_range[2],
            valinit=frequency,
        )
        self._frequency_slider.on_changed(self._on_changed)

        self._lifetime_sliders = []
        for i, (lifetime, color) in enumerate(
            zip(numpy.atleast_1d(lifetimes), self._component_colors)
        ):
            slider = Slider(
                ax=next(axes),
                label=f'Lifetime {i} ',
                valfmt=' %.2f ns',
                valmin=lifetime_range[0],
                valmax=lifetime_range[1],
                valstep=lifetime_range[2],
                valinit=lifetime,  # type: ignore[arg-type]
                facecolor=color,
            )
            slider.on_changed(self._on_changed)
            self._lifetime_sliders.append(slider)

        self._fraction_sliders = []
        for i, (fraction, color) in enumerate(
            zip(numpy.atleast_1d(fractions), self._component_colors)
        ):
            if num_components == 1 or (i == 1 and num_components == 2):
                break
            slider = Slider(
                ax=next(axes),
                label=f'Fraction {i} ',
                valfmt=' %.2f',
                valmin=0.0,
                valmax=1.0,
                valstep=0.01,
                valinit=fraction,  # type: ignore[arg-type]
                facecolor=color,
            )
            slider.on_changed(self._on_changed)
            self._fraction_sliders.append(slider)

    def _calculate(
        self,
        frequency: float | None,
        lifetimes: ArrayLike,
        fractions: ArrayLike | None,
        /,
    ) -> tuple[
        float,  # frequency
        NDArray[Any],  # lifetimes
        NDArray[Any],  # fractions
        NDArray[Any],  # signal
        NDArray[Any],  # irf
        NDArray[Any],  # times
        float,  # real
        float,  # imag
        float,  # phase
        float,  # modulation
        NDArray[Any],  # phase_
        NDArray[Any],  # modulation_
        NDArray[Any],  # component_signal
        NDArray[Any],  # component_real
        NDArray[Any],  # component_imag
        NDArray[Any],  # component_phase_
        NDArray[Any],  # component_modulation_
    ]:
        """Return values for plotting."""
        lifetimes = numpy.asarray(lifetimes)
        num_components = max(lifetimes.size, 1)

        if fractions is None:
            fractions = numpy.ones(num_components) / num_components
        else:
            fractions = numpy.asarray(fractions)
            num_fractions = max(fractions.size, 1)
            if num_fractions != num_components:
                raise ValueError(f'{num_fractions=} != {num_components=}')
            s = fractions.sum()
            if s > 0.0:
                fractions = numpy.clip(fractions / s, 0.0, 1.0)
            else:
                fractions = numpy.ones(num_components) / num_components

        if frequency is None:
            frequency = float(
                # lifetime_to_frequency(numpy.atleast_1d(lifetimes)[0])
                # lifetime_to_frequency(numpy.mean(lifetimes * fractions))
                lifetime_to_frequency(numpy.mean(lifetimes))
            )

        signal, irf, times = lifetime_to_signal(
            frequency,
            lifetimes,
            fractions,
            mean=1.0,
            samples=self._samples,
            zero_phase=self._zero_phase,
            zero_stdev=self._zero_stdev,
        )
        signal_max = signal.max()
        signal /= signal_max
        irf /= signal_max

        component_signal = lifetime_to_signal(
            frequency,
            lifetimes,
            mean=fractions,
            samples=self._samples,
            zero_phase=self._zero_phase,
            zero_stdev=self._zero_stdev,
        )[0]
        component_signal /= signal_max

        real, imag = phasor_from_lifetime(frequency, lifetimes, fractions)
        component_real, component_imag = phasor_from_lifetime(
            frequency, lifetimes
        )

        phase, modulation = _degpct(*phasor_to_polar(real, imag))
        phase_, modulation_ = _degpct(
            *phasor_to_polar(
                *phasor_from_lifetime(self._frequencies, lifetimes, fractions)
            )
        )
        component_phase_, component_modulation_ = phasor_to_polar(
            *phasor_from_lifetime(self._frequencies, lifetimes)
        )
        component_phase_, component_modulation_ = _degpct(
            component_phase_.T, component_modulation_.T
        )

        return (
            frequency,
            lifetimes,
            fractions,
            signal,
            irf,
            times,
            float(real),
            float(imag),
            float(phase),
            float(modulation),
            phase_,
            modulation_,
            component_signal,
            component_real,
            component_imag,
            component_phase_,
            component_modulation_,
        )  # type: ignore[return-value]

    def _on_changed(self, value: Any) -> None:
        """Callback function to update plot with current slider values."""
        frequency = self._frequency_slider.val

        if frequency != self._frequency:
            if self._semicircle_ticks is not None:
                lifetime, labels = _semicircle_ticks(frequency)
                self._semicircle_ticks.labels = labels
                self._semicircle_line.set_data(
                    *phasor_transform(
                        *phasor_from_lifetime(frequency, lifetime)
                    )
                )
            self._frequency_line.set_data([frequency, frequency], [0, 90])
            # self._time_plot.set_title(f'Time domain ({frequency:.0f} MHz)')
            # self._phasor_plot.set_title(f'Phasor plot ({frequency:.0f} MHz)')

        lifetimes = numpy.asarray([s.val for s in self._lifetime_sliders])
        fractions = numpy.asarray([s.val for s in self._fraction_sliders])

        num_components = len(lifetimes)
        if num_components == 1:
            fractions = numpy.asarray([1.0])
        elif num_components == 2:
            fractions = numpy.asarray([fractions[0], 1.0 - fractions[0]])

        (
            frequency,
            lifetimes,
            fractions,
            signal,
            irf,
            times,
            real,
            imag,
            phase,
            modulation,
            phase_,
            modulation_,
            component_signal,
            component_real,
            component_imag,
            component_phase_,
            component_modulation_,
        ) = self._calculate(frequency, lifetimes, fractions)

        # time domain plot
        self._signal_lines[0].set_data(times, signal)
        if num_components > 1:
            for i in range(num_components):
                self._signal_lines[i + 1].set_data(times, component_signal[i])
        self._signal_lines[-1].set_data(times, irf)
        if frequency != self._frequency:
            self._time_plot.relim()
            self._time_plot.autoscale_view()

        # phasor plot
        self._phasor_points[0].set_data([real], [imag])
        if num_components > 1:
            for i in range(num_components):
                self._phasor_lines[i].set_data(
                    [real, component_real[i]], [imag, component_imag[i]]
                )
                self._phasor_points[i + 1].set_data(
                    [component_real[i]],
                    [component_imag[i]],
                )

        # frequency domain plot
        self._frequency_line.set_data([frequency, frequency], [0, 90])
        self._phase_point.set_data([frequency], [phase])
        self._modulation_point.set_data([frequency], [modulation])
        self._phase_lines[0].set_data(self._frequencies, phase_)
        self._modulation_lines[0].set_data(self._frequencies, modulation_)
        if num_components > 1:
            for i in range(num_components):
                self._phase_lines[i + 1].set_data(
                    self._frequencies, component_phase_[i]
                )
                self._modulation_lines[i + 1].set_data(
                    self._frequencies, component_modulation_[i]
                )

        self._frequency = frequency

    def show(self) -> None:
        """Display all open figures. Call :py:func:`matplotlib.pyplot.show`."""
        pyplot.show()


def _degpct(
    phase: ArrayLike, modulation: ArrayLike, /
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Return phase in degrees and modulation in percent."""
    return numpy.rad2deg(phase), numpy.multiply(modulation, 100.0)
