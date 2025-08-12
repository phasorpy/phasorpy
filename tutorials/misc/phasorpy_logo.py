"""
PhasorPy logo
=============

Render the PhasorPy logo.

This script generates the PhasorPy logo using the PhasorPy library and
matplotlib.
The schematic logo shows a universal semicircle and the phasor coordinates of
a mixture of two lifetime components. The corresponding time-domain and
first harmonic signals are shown in insets. An optional arrow is drawn
between the time-domain signal and phasor coordinates to indicate the
phasor transformation.
The logo is scalable, works with white and dark backgrounds, and can be
cropped to a circle.

"""

# %%
# Import required modules and define functions:

from matplotlib import pyplot

from phasorpy.lifetime import lifetime_to_signal, phasor_from_lifetime
from phasorpy.phasor import phasor_from_signal, phasor_to_signal
from phasorpy.plot import PhasorPlot

# %%
# Create signals and phasor coordinates for the logo:

frequency = 80.0  # MHz
lifetime = 0.5, 5.0  # ns
fraction = 0.6, 0.4
samples = 256  # number of signal samples

signal, irf, times = lifetime_to_signal(
    frequency * 4,  # spread out the decay
    lifetime,
    fraction,
    mean=1.0,
    samples=samples,
    zero_phase=1.1,
    zero_stdev=0.25,
)
signal /= signal.max()

phasor = phasor_from_lifetime(frequency, lifetime, fraction)
harmonic = phasor_to_signal(*phasor_from_signal(signal), samples=samples)
harmonic /= harmonic.max()

# %%
# Create the PhasorPy logo and save it as a scalable vector graphics (SVG)
# file:

phasor_color = 'tab:blue'
signal_color = 'tab:orange'
harmonic_color = 'tab:green'
arrow_color = None  # set to show arrow
outline_color = None  # set to show outline
linewidth = 20

fig, ax = pyplot.subplots(figsize=(6.4, 6.4))
ax.set_axis_off()

plot = PhasorPlot(
    ax=ax,
    frequency=frequency,
    grid=False,
    allquadrants=False,
    title='',
    xlim=(-0.04, 1.04),
    ylim=(-0.55, 0.55),
)

plot.semicircle(lw=linewidth, ls='-', color=phasor_color, capstyle='round')

# draw marker at phasor coordinate of mixture
plot.plot(
    *phasor,
    marker='o',
    markersize=3 * linewidth,
    markeredgewidth=linewidth,
    markeredgecolor=phasor_color,
    markerfacecolor='None',
    zorder=3,
)

# draw time-domain signal in inset
cax = ax.inset_axes((0.015, 0.32, 0.97, 0.45))
cax.set_ylim((0.02, 1.1))
cax.set_axis_off()
cax.plot(
    times,
    signal,
    '-',
    linewidth=linewidth,
    solid_capstyle='round',
    color=signal_color,
)

# draw harmonics in second inset
cax = ax.inset_axes((0.125, 0.105, 0.75, 0.25))
cax.set_ylim(-0.05, 1.1)
cax.set_axis_off()
cax.plot(
    times,
    harmonic,
    '-',
    linewidth=linewidth,
    solid_capstyle='round',
    color=harmonic_color,
)

# draw arrow indicating phasor transformation
if arrow_color is not None:
    point = 0.28, -0.04
    length = 0.4, 0.77
    plot.arrow(
        (
            point[0] + length[0] * (phasor[0] - point[0]),
            point[1] + length[0] * (phasor[1] - point[1]),
        ),
        (
            point[0] + length[1] * (phasor[0] - point[0]),
            point[1] + length[1] * (phasor[1] - point[1]),
        ),
        color=arrow_color,
        linewidth=linewidth,
        mutation_scale=55,
        zorder=5,
        angle=0.0,
    )

# draw circle outline
if outline_color is not None:
    plot.circle(
        0.5,
        0.0,
        0.5,
        linestyle='-',
        linewidth=linewidth,
        color=outline_color,
        fill=False,
    )

pyplot.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)

# save the figure to a SVG file:
plot.save('phasorpy_logo.svg', dpi=160, transparent=True, bbox_inches='tight')
plot.show()

# %%
# mypy: disable-error-code="unreachable"
