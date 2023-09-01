Phasor approach
===============

“The phasor approach to fluorescence lifetime imaging, and more recently
hyperspectral fluorescence imaging, has increased the use of these
techniques, and improved the ease and intuitiveness of the data analysis.
The fit-free nature of the phasor plots increases the speed of the analysis
and reduces the dimensionality, optimization of data handling and storage.
The reciprocity principle between the real and imaginary space—where the
phasor and the pixel that the phasor originated from are linked and can be
converted from one another—has helped the expansion of this method.
The phasor coordinates calculated from a pixel, where multiple fluorescent
species are present, depends on the phasor positions of those components.
The relative positions are governed by the linear combination properties of
the phasor space. According to this principle, the phasor position of a
pixel with multiple components lies inside the polygon whose vertices are
occupied by the phasor positions of these individual components and the
distance between the image phasor to any of the vertices is inversely
proportional to the fractional intensity contribution of that component to
the total fluorescence from that image pixel.
The higher the fractional intensity contribution of a vertex, the closer is
the resultant phasor. The linear additivity in the phasor space can be
exploited to obtain the fractional intensity contribution from multiple
species and quantify their contribution.”
(quoted from :ref:`Malacrida et al., 2021 <malacrida-2021>`)

The following resources provide an overview of the history, theory,
applications, and implementations of the phasor approach:

Wiki
----

- `Phasor approach to fluorescence lifetime and spectral imaging
  <https://en.wikipedia.org/wiki/Phasor_approach_to_fluorescence_lifetime_and_spectral_imaging>`_.
  *Wikipedia* (2023)

Talks
-----

- Enrico Gratton.
  `The phasor approach to FLIM and FRET
  <https://www.lfd.uci.edu/workshop/2021/>`_.
  *15th LFD Workshop* (2021)

- Leonel Malacrida.
  `Spectral phasors
  <https://www.lfd.uci.edu/workshop/2021/>`_.
  *15th LFD Workshop* (2021)

- Alexander Vallmitjana Lees.
  `Clustering and unmixing in the phasor space
  <https://www.lfd.uci.edu/workshop/2021/>`_.
  *15th LFD Workshop* (2021)

- David Jameson.
  `Introduction to phasors <https://www.lfd.uci.edu/workshop/2019/>`_.
  *14th LFD Workshop* (2019)

- David Jameson.
  `A brief history of phasors <https://www.lfd.uci.edu/colloquium/>`_.
  *LFD Colloquium* (2020)

Articles
--------

- Vallmitjana A, Lepanto P, Irigoin F, Malacrida L.
  `Phasor-based multi-harmonic unmixing for in-vivo hyperspectral imaging
  <https://doi.org/10.1088/2050-6120/ac9ae9>`_.
  *Methods Appl Fluoresc*. 11(1): 014001 (2022)

- Torrado B, Malacrida L, Ranjit S.
  `Linear combination properties of the phasor space in fluorescence imaging
  <https://doi.org/10.3390/s22030999>`_.
  *Sensors*. 22(3): 999 (2022)

  .. _malacrida-2021:
- Malacrida L, Ranjit S, Jameson DM, Gratton E.
  `The phasor plot: a universal circle to advance fluorescence lifetime
  analysis and interpretation
  <https://doi.org/10.1146/annurev-biophys-062920-063631>`_.
  *Annu Rev Biophys*. 50:575-593 (2021)

- Ranjit S, Malacrida L, Jameson DM, Gratton E.
  `Fit-free analysis of fluorescence lifetime imaging data using the phasor
  approach <https://doi.org/10.1038/s41596-018-0026-5>`_.
  *Nat Protoc*. 13(9): 1979-2004 (2018)

- Malacrida L, Jameson DM, Gratton E.
  `A multidimensional phasor approach reveals LAURDAN photophysics in NIH-3T3
  cell membranes <https://doi.org/10.1038/s41598-017-08564-z>`_.
  *Sci Rep*. 7(1): 9215 (2017)

- Fereidouni F, Bader AN, Gerritsen HC.
  `Spectral phasor analysis allows rapid and reliable unmixing of fluorescence
  microscopy spectral images <https://doi.org/10.1364/OE.20.012729>`_.
  *Opt Express*. 20(12): 12729-41 (2012)

- Digman MA, Caiolfa VR, Zamai M, Gratton E.
  `The phasor approach to fluorescence lifetime imaging analysis
  <https://doi.org/10.1529/biophysj.107.120154>`_.
  *Biophys J*. 94(2): L14-16 (2008)

- Redford GI, Clegg RM.
  `Polar plot representation for frequency-domain analysis of fluorescence
  lifetimes <https://doi.org/10.1007/s10895-005-2990-8>`_.
  *J Fluoresc*. 15(5): 805-15 (2005)

- Clayton AHA, Hanley QS, Verveer PJ.
  `Graphical representation and multicomponent analysis of single-frequency
  fluorescence lifetime imaging microscopy data
  <https://doi.org/10.1111/j.1365-2818.2004.01265.x>`_.
  *J Microscopy*. 213(1): 1-5 (2004)

Software
--------

Besides the PhasorPy library, several other software implemented the phasor
approach to analyze fluorescence time-resolved or spectral images:

- `Globals for Images · SimFCS <https://www.lfd.uci.edu/globals/>`_
  is a free, closed-source, Windows desktop application for fluorescence image
  analysis, visualization, simulation, and acquisition.
  The software was developed by Enrico Gratton during 1998-2022 at the
  Laboratory for Fluorescence Dynamics. It provides the most comprehensive
  set of features for phasor analysis of fluorescence lifetime and
  hyperspectral images.
  Many `tutorials <https://www.lfd.uci.edu/globals/tutorials/>`_ are available.

- `Spectral Phasor PlugIn <http://spechron.com/Spectral%20Phasor-Download.aspx>`_
  and
  `Time Gated Phasor PlugIn <http://spechron.com/Time%20gated%20Phasor-Download.aspx>`_
  are open-source ImageJ plugins by Farzad Fereidouni, which provide
  visualization, segmentation, and unmixing of time-resolved and spectral
  images using the phasor approach. The software is distributed under an
  unknown license and was last updated in 2013.

- `Napari-flim-phasor-plotter <https://github.com/zoccoler/napari-flim-phasor-plotter>`_
  is a napari plugin to interactively load and show raw FLIM single images
  and series and generate phasor plots.

- `FLUTE <https://github.com/LaboratoryOpticsBiosciences/FLUTE>`_,
  the Fluorescence Lifetime Ultimate Explorer, is an open-source Python GUI
  for interactive phasor analysis of FLIM data developed by Chiara Stringari
  and others. The software is distributed under the GPL license.

- `FLIM_tools <https://github.com/jayunruh/FLIM_tools>`_
  is an open-source Python library for linear unmixing and phasor tools for
  FLIM analysis developed by Jay Unruh. The library is distributed under
  the GPL2 license.

- `PhasorIdentifier <https://github.com/Mariochem92/PhasorIdentifier>`_
  is a Jupyter notebook to analyze FLIM files, including masking, cell
  segmentation, pH correlation, nanoscale effects, and precise quantification.
  The notebook is distributed under CC BY-NC 4.0.

- `FLIMLib <https://flimlib.github.io>`_ is an exponential curve fitting
  library used for Fluorescent Lifetime Imaging (FLIM). It includes a function
  to calculate phasor coordinates from time-resolved signals.
  FLIMLib is licensed under the GPL v3.

- `Instant-FLIM-Analysis <https://github.com/yzhang34/Instant-FLIM-Analysis>`_
  is an open-source Matlab program that analyzes data acquired with an
  "instant FLIM" system. It supports image segmentation based on phasor plot
  regions of interest and K-means clustering.

- `FLIM Studio <https://www.flimlabs.com/software/>`_
  is a commercial software by FLIM LABS, a vendor of portable devices for
  fluorescence lifetime imaging and spectroscopy. The software provides
  real-time FLIM phasor-plot analysis, AI-driven phasor-plot analysis
  techniques, and an application programming interface.

- `VistaVision <https://iss.com/software/vistavision>`_
  is a commercial Windows desktop software by ISS, Inc., for confocal
  microscopy applications, including instrument control, data acquisition,
  and data processing. It performs image segmentation of FLIM images via
  the phasor plot.

- `SPCImage <https://www.becker-hickl.com/literature/documents/flim/spcimage-ng/>`_
  is a commercial Windows desktop software by Becker & Hickl (tm) for
  TCSPC-FLIM data analysis. It performs image segmentation of time-resolved
  data via the phasor plot.

- `LAS X <https://www.leica-microsystems.com/science-lab/phasor-analysis-for-flim-fluorescence-lifetime-imaging-microscopy/>`_
  is a commercial Windows desktop software by Leica Microsystems.
  The software allows, by using phasors, to follow microenvironmental changes,
  select components to multiplex signal, and determine FRET efficiency.

- `Luminosa <https://www.picoquant.com/products/category/fluorescence-microscopes/luminosa-single-photon-counting-confocal-microscope#documents>`_
  is a commercial Windows desktop software by PicoQuant GmbH, which includes
  single molecule detection, FCS, and time-resolved imaging methods.
  The InstaFLIM module allows simultaneous TCSPC and phasor analysis
  options for ROI determination.

- `FlimFast <https://www.cgohlke.com/flimfast/>`_
  was a research-grade Windows desktop software for frequency-domain,
  full-field, fluorescence lifetime imaging at video rate, developed by
  Christoph Gohlke during 2000-2002 at UIUC. It enabled phasor vs intensity
  plots of FLIM images during real-time acquisition.
