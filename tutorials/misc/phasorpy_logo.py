"""
PhasorPy logo
=============

Render the PhasorPy logo.

This script generates the PhasorPy logo using the PhasorPy library and
matplotlib.
The schematic logo shows a universal semicircle and a cursor around the
phasor coordinates of a mixture of two lifetime components.
The corresponding time-domain and first harmonic signals are shown in insets.
An optional arrow is drawn between the time-domain signal and phasor
coordinates to indicate the phasor transformation.
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
lifetime = [0.5, 6.0]  # ns
fraction = [0.66, 0.34]
samples = 256  # number of signal samples

signal, irf, times = lifetime_to_signal(
    frequency * 4,  # spread out the decay
    lifetime,
    fraction,
    samples=samples,
    zero_phase=1.2,
    zero_stdev=0.25,
)
signal /= signal.max()

phasor = phasor_from_lifetime(frequency, lifetime, fraction)
harmonic = phasor_to_signal(*phasor_from_signal(signal), samples=samples)
harmonic /= harmonic.max()

# %%
# Create the PhasorPy logo and save it as a scalable vector graphics (SVG)
# file:

semicircle_color = 'tab:blue'
cursor_color = 'tab:blue'
signal_color = 'tab:orange'
harmonic_color = 'tab:green'
arrow_color = None  # set to show arrow
background_color = None  # set to fill circle
linewidth = 25

fig, ax = pyplot.subplots(figsize=(6.4, 6.4))
ax.set_axis_off()

plot = PhasorPlot(ax=ax, xlim=(-0.04, 1.04), ylim=(-0.55, 0.55), title=None)

plot.semicircle(
    frequency=frequency,
    lifetime=[],  # turn off ticks
    color=semicircle_color,
    linewidth=linewidth,
    capstyle='round',
)

# draw cursor around phasor coordinate of mixture
plot.cursor(*phasor, radius=0.08, linewidth=linewidth, color=cursor_color)

# draw time-domain signal in inset
cax = ax.inset_axes((0.015, 0.325, 0.97, 0.45))
cax.set_ylim(0.02, 1.1)
cax.set_axis_off()
cax.plot(
    times,
    signal,
    '-',
    color=signal_color,
    linewidth=linewidth,
    solid_capstyle='round',
)

# draw harmonics in second inset
cax = ax.inset_axes((0.125, 0.113, 0.75, 0.25))
cax.set_ylim(-0.11, 1.14)
cax.set_axis_off()
cax.plot(
    times,
    harmonic,
    '-',
    color=harmonic_color,
    linewidth=linewidth,
    solid_capstyle='round',
)

# draw arrow indicating phasor transformation
if arrow_color is not None:
    point = 0.28, -0.04
    length = 0.46, 0.74
    plot.arrow(
        [
            point[0] + length[0] * (phasor[0] - point[0]),
            point[1] + length[0] * (phasor[1] - point[1]),
        ],
        [
            point[0] + length[1] * (phasor[0] - point[0]),
            point[1] + length[1] * (phasor[1] - point[1]),
        ],
        color=arrow_color,
        linewidth=linewidth,
    )

# draw circle outline
if background_color is not None:
    plot.circle(
        0.5,
        0.0,
        0.5,
        color=background_color,
        linewidth=linewidth,
        linestyle='-',
        fill=True,
        zorder=0,
    )

pyplot.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)

# save the figure to a SVG file:
plot.save('phasorpy_logo.svg', dpi=160, transparent=True, bbox_inches='tight')
plot.show()

# %%
# mypy: disable-error-code="unreachable"
