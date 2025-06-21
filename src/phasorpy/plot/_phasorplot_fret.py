"""PhasorPlotFret class."""

from __future__ import annotations

__all__ = ['PhasorPlotFret']

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .._typing import Any, NDArray

    from matplotlib.axes import Axes

import numpy
from matplotlib import pyplot
from matplotlib.lines import Line2D
from matplotlib.widgets import Slider

from .._utils import update_kwargs
from ..phasor import (
    phasor_from_fret_acceptor,
    phasor_from_fret_donor,
    phasor_from_lifetime,
    phasor_semicircle,
    phasor_to_polar,
    phasor_transform,
)
from ._phasorplot import PhasorPlot, SemicircleTicks, _semicircle_ticks


class PhasorPlotFret(PhasorPlot):
    """FRET phasor plot.

    Plot Förster Resonance Energy Transfer efficiency trajectories
    of donor and acceptor channels in phasor space.

    Parameters
    ----------
    frequency : array_like
        Laser pulse or modulation frequency in MHz.
    donor_lifetime : array_like
        Lifetime of donor without FRET in ns.
    acceptor_lifetime : array_like
        Lifetime of acceptor in ns.
    fret_efficiency : array_like, optional, default 0
        FRET efficiency in range [0, 1].
    donor_fretting : array_like, optional, default 1
        Fraction of donors participating in FRET. Range [0, 1].
    donor_bleedthrough : array_like, optional, default 0
        Weight of donor fluorescence in acceptor channel
        relative to fluorescence of fully sensitized acceptor.
        A weight of 1 means the fluorescence from donor and fully sensitized
        acceptor are equal.
        The background in the donor channel does not bleed through.
    acceptor_bleedthrough : array_like, optional, default 0
        Weight of fluorescence from directly excited acceptor
        relative to fluorescence of fully sensitized acceptor.
        A weight of 1 means the fluorescence from directly excited acceptor
        and fully sensitized acceptor are equal.
    acceptor_background : array_like, optional, default 0
        Weight of background fluorescence in acceptor channel
        relative to fluorescence of fully sensitized acceptor.
        A weight of 1 means the fluorescence of background and fully
        sensitized acceptor are equal.
    donor_background : array_like, optional, default 0
        Weight of background fluorescence in donor channel
        relative to fluorescence of donor without FRET.
        A weight of 1 means the fluorescence of background and donor
        without FRET are equal.
    background_real : array_like, optional, default 0
        Real component of background fluorescence phasor coordinate
        at `frequency`.
    background_imag : array_like, optional, default 0
        Imaginary component of background fluorescence phasor coordinate
        at `frequency`.
    ax : matplotlib axes, optional
        Matplotlib axes used for plotting.
        By default, a new subplot axes is created.
        Cannot be used with `interactive` mode.
    interactive : bool, optional, default: False
        Use matplotlib slider widgets to interactively control parameters.
    **kwargs
        Additional parameters passed to :py:class:`phasorpy.plot.PhasorPlot`.

    See Also
    --------
    phasorpy.phasor.phasor_from_fret_donor
    phasorpy.phasor.phasor_from_fret_acceptor
    :ref:`sphx_glr_tutorials_api_phasorpy_fret.py`

    """

    _fret_efficiencies: NDArray[Any]

    _frequency_slider: Slider
    _donor_lifetime_slider: Slider
    _acceptor_lifetime_slider: Slider
    _fret_efficiency_slider: Slider
    _donor_fretting_slider: Slider
    _donor_bleedthrough_slider: Slider
    _acceptor_bleedthrough_slider: Slider
    _acceptor_background_slider: Slider
    _donor_background_slider: Slider
    _background_real_slider: Slider
    _background_imag_slider: Slider

    _donor_line: Line2D
    _donor_only_line: Line2D
    _donor_fret_line: Line2D
    _donor_trajectory_line: Line2D
    _donor_semicircle_line: Line2D
    _donor_donor_line: Line2D
    _donor_background_line: Line2D
    _acceptor_line: Line2D
    _acceptor_only_line: Line2D
    _acceptor_trajectory_line: Line2D
    _acceptor_semicircle_line: Line2D
    _acceptor_background_line: Line2D
    _background_line: Line2D

    _donor_semicircle_ticks: SemicircleTicks | None

    def __init__(
        self,
        *,
        frequency: float = 60.0,
        donor_lifetime: float = 4.2,
        acceptor_lifetime: float = 3.0,
        fret_efficiency: float = 0.5,
        donor_fretting: float = 1.0,
        donor_bleedthrough: float = 0.0,
        acceptor_bleedthrough: float = 0.0,
        acceptor_background: float = 0.0,
        donor_background: float = 0.0,
        background_real: float = 0.0,
        background_imag: float = 0.0,
        ax: Axes | None = None,
        interactive: bool = False,
        **kwargs: Any,
    ) -> None:
        update_kwargs(
            kwargs,
            title='PhasorPy FRET phasor plot',
            xlim=[-0.2, 1.1],
            ylim=[-0.1, 0.8],
        )
        kwargs['allquadrants'] = False
        kwargs['grid'] = False

        if ax is not None:
            interactive = False
        else:
            fig = pyplot.figure()
            ax = fig.add_subplot()
            if interactive:
                w, h = fig.get_size_inches()
                fig.set_size_inches(w, h * 1.66)
                fig.subplots_adjust(bottom=0.45)
                fcm = fig.canvas.manager
                if fcm is not None:
                    fcm.set_window_title(kwargs['title'])

        super().__init__(ax=ax, **kwargs)

        self._fret_efficiencies = numpy.linspace(0.0, 1.0, 101)

        donor_real, donor_imag = phasor_from_lifetime(
            frequency, donor_lifetime
        )
        donor_fret_real, donor_fret_imag = phasor_from_lifetime(
            frequency, donor_lifetime * (1.0 - fret_efficiency)
        )
        acceptor_real, acceptor_imag = phasor_from_lifetime(
            frequency, acceptor_lifetime
        )
        donor_trajectory_real, donor_trajectory_imag = phasor_from_fret_donor(
            frequency,
            donor_lifetime,
            fret_efficiency=self._fret_efficiencies,
            donor_fretting=donor_fretting,
            donor_background=donor_background,
            background_real=background_real,
            background_imag=background_imag,
        )
        (
            acceptor_trajectory_real,
            acceptor_trajectory_imag,
        ) = phasor_from_fret_acceptor(
            frequency,
            donor_lifetime,
            acceptor_lifetime,
            fret_efficiency=self._fret_efficiencies,
            donor_fretting=donor_fretting,
            donor_bleedthrough=donor_bleedthrough,
            acceptor_bleedthrough=acceptor_bleedthrough,
            acceptor_background=acceptor_background,
            background_real=background_real,
            background_imag=background_imag,
        )

        # add plots
        lines = self.semicircle(frequency=frequency)
        self._donor_semicircle_line = lines[0]
        self._donor_semicircle_ticks = self._semicircle_ticks

        lines = self.semicircle(
            phasor_reference=(float(acceptor_real), float(acceptor_imag)),
            use_lines=True,
        )
        self._acceptor_semicircle_line = lines[0]

        if donor_fretting < 1.0 and donor_background == 0.0:
            lines = self.line(
                [donor_real, donor_fret_real],
                [donor_imag, donor_fret_imag],
            )
        else:
            lines = self.line([0.0, 0.0], [0.0, 0.0])
        self._donor_donor_line = lines[0]

        if acceptor_background > 0.0:
            lines = self.line(
                [float(acceptor_real), float(background_real)],
                [float(acceptor_imag), float(background_imag)],
            )
        else:
            lines = self.line([0.0, 0.0], [0.0, 0.0])
        self._acceptor_background_line = lines[0]

        if donor_background > 0.0:
            lines = self.line(
                [float(donor_real), float(background_real)],
                [float(donor_imag), float(background_imag)],
            )
        else:
            lines = self.line([0.0, 0.0], [0.0, 0.0])
        self._donor_background_line = lines[0]

        lines = self.plot(
            donor_trajectory_real,
            donor_trajectory_imag,
            '-',
            color='tab:green',
        )
        self._donor_trajectory_line = lines[0]

        lines = self.plot(
            acceptor_trajectory_real,
            acceptor_trajectory_imag,
            '-',
            color='tab:red',
        )
        self._acceptor_trajectory_line = lines[0]

        lines = self.plot(
            donor_real,
            donor_imag,
            '.',
            color='tab:green',
        )
        self._donor_only_line = lines[0]

        lines = self.plot(
            donor_real,
            donor_imag,
            '.',
            color='tab:green',
        )
        self._donor_fret_line = lines[0]

        lines = self.plot(
            acceptor_real,
            acceptor_imag,
            '.',
            color='tab:red',
        )
        self._acceptor_only_line = lines[0]

        lines = self.plot(
            donor_trajectory_real[int(fret_efficiency * 100.0)],
            donor_trajectory_imag[int(fret_efficiency * 100.0)],
            'o',
            color='tab:green',
            label='Donor',
        )
        self._donor_line = lines[0]

        lines = self.plot(
            acceptor_trajectory_real[int(fret_efficiency * 100.0)],
            acceptor_trajectory_imag[int(fret_efficiency * 100.0)],
            'o',
            color='tab:red',
            label='Acceptor',
        )
        self._acceptor_line = lines[0]

        lines = self.plot(
            background_real,
            background_imag,
            'o',
            color='black',
            label='Background',
        )
        self._background_line = lines[0]

        if not interactive:
            return

        # add sliders
        axes = []
        for i in range(11):
            axes.append(fig.add_axes((0.33, 0.05 + i * 0.03, 0.45, 0.01)))

        self._frequency_slider = Slider(
            ax=axes[10],
            label='Frequency ',
            valfmt=' %.0f MHz',
            valmin=10,
            valmax=200,
            valstep=1,
            valinit=frequency,
        )
        self._frequency_slider.on_changed(self._on_semicircle_changed)

        self._donor_lifetime_slider = Slider(
            ax=axes[9],
            label='Donor lifetime ',
            valfmt=' %.1f ns',
            valmin=0.1,
            valmax=16.0,
            valstep=0.1,
            valinit=donor_lifetime,
            # facecolor='tab:green',
            handle_style={'edgecolor': 'tab:green'},
        )
        self._donor_lifetime_slider.on_changed(self._on_changed)

        self._acceptor_lifetime_slider = Slider(
            ax=axes[8],
            label='Acceptor lifetime ',
            valfmt=' %.1f ns',
            valmin=0.1,
            valmax=16.0,
            valstep=0.1,
            valinit=acceptor_lifetime,
            # facecolor='tab:red',
            handle_style={'edgecolor': 'tab:red'},
        )
        self._acceptor_lifetime_slider.on_changed(self._on_semicircle_changed)

        self._fret_efficiency_slider = Slider(
            ax=axes[7],
            label='FRET efficiency ',
            valfmt=' %.2f',
            valmin=0.0,
            valmax=1.0,
            valstep=0.01,
            valinit=fret_efficiency,
        )
        self._fret_efficiency_slider.on_changed(self._on_changed)

        self._donor_fretting_slider = Slider(
            ax=axes[6],
            label='Donors fretting ',
            valfmt=' %.2f',
            valmin=0.0,
            valmax=1.0,
            valstep=0.01,
            valinit=donor_fretting,
            # facecolor='tab:green',
            handle_style={'edgecolor': 'tab:green'},
        )
        self._donor_fretting_slider.on_changed(self._on_changed)

        self._donor_bleedthrough_slider = Slider(
            ax=axes[5],
            label='Donor bleedthrough ',
            valfmt=' %.2f',
            valmin=0.0,
            valmax=5.0,
            valstep=0.01,
            valinit=donor_bleedthrough,
            # facecolor='tab:red',
            handle_style={'edgecolor': 'tab:red'},
        )
        self._donor_bleedthrough_slider.on_changed(self._on_changed)

        self._acceptor_bleedthrough_slider = Slider(
            ax=axes[4],
            label='Acceptor bleedthrough ',
            valfmt=' %.2f',
            valmin=0.0,
            valmax=5.0,
            valstep=0.01,
            valinit=acceptor_bleedthrough,
            # facecolor='tab:red',
            handle_style={'edgecolor': 'tab:red'},
        )
        self._acceptor_bleedthrough_slider.on_changed(self._on_changed)

        self._acceptor_background_slider = Slider(
            ax=axes[3],
            label='Acceptor background ',
            valfmt=' %.2f',
            valmin=0.0,
            valmax=5.0,
            valstep=0.01,
            valinit=acceptor_background,
            # facecolor='tab:red',
            handle_style={'edgecolor': 'tab:red'},
        )
        self._acceptor_background_slider.on_changed(self._on_changed)

        self._donor_background_slider = Slider(
            ax=axes[2],
            label='Donor background ',
            valfmt=' %.2f',
            valmin=0.0,
            valmax=5.0,
            valstep=0.01,
            valinit=donor_background,
            # facecolor='tab:green',
            handle_style={'edgecolor': 'tab:green'},
        )
        self._donor_background_slider.on_changed(self._on_changed)

        self._background_real_slider = Slider(
            ax=axes[1],
            label='Background real ',
            valfmt=' %.2f',
            valmin=0.0,
            valmax=1.0,
            valstep=0.01,
            valinit=background_real,
        )
        self._background_real_slider.on_changed(self._on_changed)

        self._background_imag_slider = Slider(
            ax=axes[0],
            label='Background imag ',
            valfmt=' %.2f',
            valmin=0.0,
            valmax=0.6,
            valstep=0.01,
            valinit=background_imag,
        )
        self._background_imag_slider.on_changed(self._on_changed)

    def _on_semicircle_changed(self, value: Any) -> None:
        """Callback function to update semicircles."""
        self._frequency = frequency = self._frequency_slider.val
        acceptor_lifetime = self._acceptor_lifetime_slider.val
        if self._donor_semicircle_ticks is not None:
            lifetime, labels = _semicircle_ticks(frequency)
            self._donor_semicircle_ticks.labels = labels
            self._donor_semicircle_line.set_data(
                *phasor_transform(*phasor_from_lifetime(frequency, lifetime))
            )
        self._acceptor_semicircle_line.set_data(
            *phasor_transform(
                *phasor_semicircle(),
                *phasor_to_polar(
                    *phasor_from_lifetime(frequency, acceptor_lifetime)
                ),
            )
        )
        self._on_changed(value)

    def _on_changed(self, value: Any) -> None:
        """Callback function to update plot with current slider values."""
        frequency = self._frequency_slider.val
        donor_lifetime = self._donor_lifetime_slider.val
        acceptor_lifetime = self._acceptor_lifetime_slider.val
        fret_efficiency = self._fret_efficiency_slider.val
        donor_fretting = self._donor_fretting_slider.val
        donor_bleedthrough = self._donor_bleedthrough_slider.val
        acceptor_bleedthrough = self._acceptor_bleedthrough_slider.val
        acceptor_background = self._acceptor_background_slider.val
        donor_background = self._donor_background_slider.val
        background_real = self._background_real_slider.val
        background_imag = self._background_imag_slider.val
        e = int(self._fret_efficiency_slider.val * 100)

        donor_real, donor_imag = phasor_from_lifetime(
            frequency, donor_lifetime
        )
        donor_fret_real, donor_fret_imag = phasor_from_lifetime(
            frequency, donor_lifetime * (1.0 - fret_efficiency)
        )
        acceptor_real, acceptor_imag = phasor_from_lifetime(
            frequency, acceptor_lifetime
        )
        donor_trajectory_real, donor_trajectory_imag = phasor_from_fret_donor(
            frequency,
            donor_lifetime,
            fret_efficiency=self._fret_efficiencies,
            donor_fretting=donor_fretting,
            donor_background=donor_background,
            background_real=background_real,
            background_imag=background_imag,
        )
        (
            acceptor_trajectory_real,
            acceptor_trajectory_imag,
        ) = phasor_from_fret_acceptor(
            frequency,
            donor_lifetime,
            acceptor_lifetime,
            fret_efficiency=self._fret_efficiencies,
            donor_fretting=donor_fretting,
            donor_bleedthrough=donor_bleedthrough,
            acceptor_bleedthrough=acceptor_bleedthrough,
            acceptor_background=acceptor_background,
            background_real=background_real,
            background_imag=background_imag,
        )

        if donor_background > 0.0:
            self._donor_background_line.set_data(
                [float(donor_real), float(background_real)],
                [float(donor_imag), float(background_imag)],
            )
        else:
            self._donor_background_line.set_data([0.0, 0.0], [0.0, 0.0])

        if donor_fretting < 1.0 and donor_background == 0.0:
            self._donor_donor_line.set_data(
                [donor_real, donor_fret_real],
                [donor_imag, donor_fret_imag],
            )
        else:
            self._donor_donor_line.set_data([0.0, 0.0], [0.0, 0.0])

        if acceptor_background > 0.0:
            self._acceptor_background_line.set_data(
                [float(acceptor_real), float(background_real)],
                [float(acceptor_imag), float(background_imag)],
            )
        else:
            self._acceptor_background_line.set_data([0.0, 0.0], [0.0, 0.0])

        self._background_line.set_data([background_real], [background_imag])

        self._donor_only_line.set_data([donor_real], [donor_imag])
        self._donor_fret_line.set_data([donor_fret_real], [donor_fret_imag])
        self._donor_trajectory_line.set_data(
            donor_trajectory_real, donor_trajectory_imag
        )
        self._donor_line.set_data(
            [donor_trajectory_real[e]], [donor_trajectory_imag[e]]
        )

        self._acceptor_only_line.set_data([acceptor_real], [acceptor_imag])
        self._acceptor_trajectory_line.set_data(
            acceptor_trajectory_real, acceptor_trajectory_imag
        )
        self._acceptor_line.set_data(
            [acceptor_trajectory_real[e]], [acceptor_trajectory_imag[e]]
        )
