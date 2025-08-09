"""
PhasorPy Logo
=============

Render the PhasorPy logo.

This script generates the PhasorPy logo using the PhasorPy library and
matplotlib.
The schematic logo shows a universal semicircle and the phasor coordinates of
a mixture of two lifetime components. The corresponding time-domain signal is
shown in an inset. An arrow is drawn between the time-domain signal and
phasor coordinates to indicate the phasor transformation.
The logo is scalable and works with white and dark backgrounds.

"""

# %%
# Import required modules and define functions:

from matplotlib import pyplot

from phasorpy.lifetime import lifetime_to_signal, phasor_from_lifetime
from phasorpy.plot import PhasorPlot

# %%
# Create data for the logo:

frequency = 80.0  # MHz
lifetime = 0.5, 5.0  # ns
fraction = 0.6, 0.4

signal, irf, times = lifetime_to_signal(
    frequency * 4,  # spread out the decay
    lifetime,
    fraction,
    mean=1.0,
    samples=256,  # smooth
    zero_phase=1.1,
    zero_stdev=0.25,
)
signal /= signal.max()

center = phasor_from_lifetime(frequency, lifetime, fraction)
phasor = phasor_from_lifetime(frequency, lifetime)

# %%
# Create the PhasorPy logo and save it as a scalable vector graphics (SVG)
# file:

scale = 1.2  # scale factor for lines and markers
phasor_color = 'tab:blue'
signal_color = 'tab:green'

fig, ax = pyplot.subplots(figsize=(6.4, 4.2))
ax.set_axis_off()

plot = PhasorPlot(
    ax=ax,
    frequency=frequency,
    grid=False,
    allquadrants=False,
    title='',
    xlim=(-0.04, 1.04),
    ylim=(-0.15, 0.55),
)

plot.semicircle(lw=15 * scale, ls='-', color=phasor_color, capstyle='round')

# draw marker at phasor coordinates
plot.plot(
    *center,
    marker='o',
    markersize=45 * scale,
    markeredgewidth=15 * scale,
    markeredgecolor=phasor_color,
    markerfacecolor='None',
    zorder=3,
)

# draw arrow indicating phasor transformation
point = 0.4, 0.13
length = 0.66
plot.arrow(
    point,
    (
        point[0] + length * (center[0] - point[0]),
        point[1] + length * (center[1] - point[1]),
    ),
    color='tab:orange',
    lw=15 * scale,
    zorder=5,
    angle=0.0,
    mutation_scale=66,
)

# draw time-domain signal in inset
cax = ax.inset_axes((0.005, 0.0, 0.99, 0.66))
cax.set_axis_off()
cax.set_ylim((0.02, 1.1))
cax.plot(
    times,
    signal,
    '-',
    lw=15 * scale,
    solid_capstyle='round',
    color=signal_color,
)
pyplot.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)

# save the logo to a SVG file:
plot.save('phasorpy_logo.svg', dpi=200, transparent=True, bbox_inches='tight')
plot.show()

# %%
