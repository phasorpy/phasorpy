r"""
Geometrical interpretation of lifetimes
=======================================

Demonstrate the geometrical interpretation of fluorescence lifetimes in the
phasor plot.

The PhasorPy library is used to demonstrate the geometrical interpretation
of fluorescence lifetimes and other quantities in the phasor plot:

- single exponential lifetimes
- fractional intensities of lifetime components
- apparent single lifetime from phase and modulation
- normal lifetime
- phase and modulation
- phasor coordinates

"""

# %%
# Import required modules and functions:

import numpy

from phasorpy.lifetime import (
    phasor_from_lifetime,
    phasor_to_apparent_lifetime,
    phasor_to_normal_lifetime,
)
from phasorpy.phasor import phasor_to_polar
from phasorpy.plot import PhasorPlot

# %%
# Calculate phasor coordinates, polar coordinates, and apparent lifetimes
# from a set of lifetimes and their fractional intensities at a given
# frequency:

frequency = 80.0  # MHz
lifetime = [0.4, 4.0]  # ns
fraction = [0.6, 0.4]

real, imag = phasor_from_lifetime(frequency, lifetime, fraction)
phase, modulation = phasor_to_polar(real, imag)
component_real, component_imag = phasor_from_lifetime(frequency, lifetime)
tau_phi, tau_mod = phasor_to_apparent_lifetime(real, imag, frequency)
tau_norm = phasor_to_normal_lifetime(real, imag, frequency)
tau_phi_re, tau_phi_im = phasor_from_lifetime(frequency, tau_phi)
tau_mod_re, tau_mod_im = phasor_from_lifetime(frequency, tau_mod)
tau_norm_re, tau_norm_im = phasor_from_lifetime(frequency, tau_norm)

# %%
# Plot the phasor coordinates and lifetimes, and annotate the plot:

color_phasor = 'black'
color_component = 'tab:orange'
color_phase = 'tab:blue'
color_modulation = 'tab:red'
color_normal = 'tab:green'
fontsize = 18
linewidth = 3.0
textoffset = 0.01

plot = PhasorPlot(
    grid=False,
    xlim=(0.0, 1.01),
    ylim=(0.0, 0.6),
    xticks=[0, 0.5, 1],
    yticks=[0, 0.5],
    xticklabels=['0', '1/2', '1'],
    yticklabels=['0', '1/2'],
    xlabel=None,
    ylabel=None,
    title='Geometrical interpretation of lifetimes in the phasor plot',
)
plot.ax.tick_params(axis='both', which='major', labelsize=fontsize * 2 / 3)
plot.ax.spines['top'].set_visible(False)
plot.ax.spines['right'].set_visible(False)
# plot.ax.spines['left'].set_linewidth(linewidth)
# plot.ax.spines['bottom'].set_linewidth(linewidth)

plot.semicircle(color='tab:gray', linewidth=linewidth, zorder=0)

# line to G coordinate
plot.line(
    [real, real],
    [0.0, imag],
    linestyle='--',
    color=color_phasor,
    linewidth=1,
    zorder=0,
)

# line to S coordinate
plot.line(
    [real, 0],
    [imag, imag],
    linestyle='--',
    color=color_phasor,
    linewidth=1,
    zorder=0,
)

component_real = numpy.atleast_1d(component_real)
component_imag = numpy.atleast_1d(component_imag)

# arc indicating phase angle
plot.arrow(
    [modulation / 4, 0.0],
    [real / 4, imag / 4],
    angle=phase,
    color=color_phase,
    arrowstyle='-',
    linewidth=linewidth,
)

# phase line
plot.arrow(
    [0, 0],
    [real, imag],
    color=color_phase,
    linewidth=linewidth,
    arrowstyle='-',
    # zorder=1,
)

# modulation arc
plot.arrow(
    [modulation, 0.0],
    [real, imag],
    angle=phase,
    color=color_modulation,
    arrowstyle='-',
    linewidth=linewidth,
)

# normal lifetime line
plot.arrow(
    [0.5, 0.0],
    [real, imag],
    color=color_normal,
    arrowstyle='-',
    linewidth=linewidth,
)

# arrows to single lifetimes
for i in range(len(lifetime)):
    plot.arrow(
        [real, imag],
        [component_real[i], component_imag[i]],
        color=color_component,
        linewidth=linewidth,
    )

# arrow to phase lifetime
plot.arrow(
    [real, imag],
    [tau_phi_re, tau_phi_im],
    color=color_phase,
    linewidth=linewidth,
)

# arced arrow to modulation lifetime
plot.arrow(
    [real, imag],
    [tau_mod_re, tau_mod_im],
    angle=phase,
    color=color_modulation,
    linewidth=linewidth,
)

# arrow to normal lifetime
plot.arrow(
    [real, imag],
    [tau_norm_re, tau_norm_im],
    color=color_normal,
    linewidth=linewidth,
)

# mark phasor coordinate
plot.plot(
    real,
    imag,
    marker='o',
    markersize=12,
    markeredgewidth=linewidth,
    markeredgecolor=color_phasor,
    markerfacecolor='white',
    zorder=1,
)

# label G
plot.ax.text(
    real,
    -textoffset,
    '$G$',
    color=color_phasor,
    fontsize=fontsize,
    ha='center',
    va='top',
)

# label S
plot.ax.text(
    -textoffset,
    imag,
    '$S$',
    color=color_phasor,
    fontsize=fontsize,
    ha='right',
    va='center',
)

# label phase
plot.ax.text(
    modulation / 4 + textoffset,
    imag / 8,
    '$\\varphi$',
    color=color_phase,
    fontsize=fontsize,
    ha='left',
    va='center',
)

# label modulation
plot.ax.text(
    modulation,
    -textoffset,
    '$M$',
    color=color_modulation,
    fontsize=fontsize,
    ha='center',
    va='top',
)

# label single lifetimes
plot.components(
    component_real,
    component_imag,
    fraction=fraction,
    color=color_component,
    fontsize=fontsize,
    linestyle=' ',
    labels=['$\\tau_{%d}$' % i for i in range(len(lifetime))],
)

# label fractions of single lifetimes
if len(lifetime) == 2:
    for i in range(len(lifetime)):
        plot.ax.text(
            real + (component_real[i] - real) / 2,
            imag + (component_imag[i] - imag) / 2 - textoffset,
            '$\\alpha_{%i}$' % i,
            color=color_component,
            fontsize=fontsize,
            ha='center',
            va='top',
        )

# label phase lifetime
plot.components(
    tau_phi_re,
    tau_phi_im,
    fontsize=fontsize,
    color=color_phase,
    labels=['$\\tau_{\\varphi}$'],
)

# label modulation lifetime
plot.components(
    tau_mod_re,
    tau_mod_im,
    color=color_modulation,
    fontsize=fontsize,
    labels=['$\\tau_{M}$'],
)

# label normal lifetime
plot.components(
    tau_norm_re,
    tau_norm_im,
    color=color_normal,
    fontsize=fontsize,
    labels=['$\\tau_{N}$'],
)

plot.show()

# %%
# The figure demonstrates:
#
# - **single exponential lifetimes** :math:`\tau_{i}` correspond to
#   phasor coordinates on the **universal semicircle**.
# - the **phasor coordinates** :math:`G` and :math:`S` of a mixture of
#   two components with single-exponential lifetimes, weighted by their
#   fractional intensities :math:`\alpha_{i}`, lie on a line between the
#   phasor coordinates of the single components.
# - the **phase** :math:`\varphi` of the phasor coordinates :math:`G` and
#   :math:`S` is the angle of the phasor coordinates with respect to the
#   origin.
# - the **modulation** :math:`M` of the phasor coordinates :math:`G` and
#   :math:`S` is the distance from the origin to the phasor coordinates.
# - the **apparent single lifetime from phase** :math:`\tau_{\varphi}`
#   of the component mixture is the single exponential lifetime corresponding
#   to the intersection of the universal circle with a line through the origin
#   and the phasor coordinates :math:`G` and :math:`S`.
# - the **apparent single lifetime from modulation** :math:`\tau_{M}`
#   of the component mixture is the single exponential lifetime corresponding
#   to the intersection of the universal circle with a circle around the origin
#   of radius equal to the modulation :math:`M`.
# - the **normal lifetime** :math:`\tau_{N}` of the component mixture
#   is the single exponential lifetime corresponding to the nearest point on
#   the universal circle to the phasor coordinates :math:`G` and :math:`S`,
#   which is the intersection of the universal circle with the line through the
#   center of the universal circle and the phasor coordinates :math:`G` and
#   :math:`S`.

# %%
# sphinx_gallery_thumbnail_number = -1
# mypy: allow-untyped-defs, allow-untyped-calls
# mypy: disable-error-code="arg-type, assignment"
