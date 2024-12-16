Release notes
=============

This document describes changes to the PhasorPy library that are specific to
a release. It includes descriptions of bug fixes, feature enhancements,
documentation and maintenance changes.

.. note::
    The PhasorPy library is in its early stages of development.
    It is not nearly feature complete.
    Large, backwards-incompatible changes may occur between revisions.

0.3 (2024.12.16)
----------------

This is the third alpha release of the PhasorPy library.
It fixes the averaging of phasor coordinates in the `phasor_center` and
`phasor_calibrate` functions, replaces the `phasor_filter` function
with `phasor_filter_median`, adds support for multiple harmonics to
`phasor_threshold`, and adds the `real_imspector_tiff` function to read
ImSpector FLIM TIFF files. This release supports Python 3.10 to 3.13.

What's Changed
..............

* Bump version by @cgohlke in https://github.com/phasorpy/phasorpy/pull/152
* Bump the github-actions group with 2 updates by @dependabot in https://github.com/phasorpy/phasorpy/pull/153
* Mention GSLab software by @cgohlke in https://github.com/phasorpy/phasorpy/pull/156
* Mention BrightEyes software by @cgohlke in https://github.com/phasorpy/phasorpy/pull/157
* Pin numpy<2.2.0 for static code analysis by @cgohlke in https://github.com/phasorpy/phasorpy/pull/159
* Fix mypy errors with matplotlib-3.10 by @cgohlke in https://github.com/phasorpy/phasorpy/pull/160
* Fix averaging phasor coordinates and related issues by @cgohlke in https://github.com/phasorpy/phasorpy/pull/155
* Add support for ImSpector FLIM TIFF files by @cgohlke in https://github.com/phasorpy/phasorpy/pull/161
* Release v0.3 by @cgohlke in https://github.com/phasorpy/phasorpy/pull/162

**Full Changelog**: https://github.com/phasorpy/phasorpy/compare/v0.2...v0.3

0.2 (2024.11.30)
----------------

This is the second alpha release of the PhasorPy library.
It fixes NaN handling in the median filter, simplifies multiple harmonic
calibration, and adds functions for spectral vector denoising and Anscombe
transformation. This release supports Python 3.10 to 3.13.

What's Changed
..............

* Bump version by @cgohlke in https://github.com/phasorpy/phasorpy/pull/132
* Add documentation version switcher config file by @cgohlke in https://github.com/phasorpy/phasorpy/pull/134
* Bump pypa/cibuildwheel from 2.20.0 to 2.21.1 in the github-actions group by @dependabot in https://github.com/phasorpy/phasorpy/pull/133
* Update FLUTE license by @cgohlke in https://github.com/phasorpy/phasorpy/pull/137
* Support Linux on AArch64 by @cgohlke in https://github.com/phasorpy/phasorpy/pull/135
* Improve private parse_harmonic function by @cgohlke in https://github.com/phasorpy/phasorpy/pull/138
* Add Anscombe transformation functions by @cgohlke in https://github.com/phasorpy/phasorpy/pull/139
* Mention PhasorPlots for dummies by @cgohlke in https://github.com/phasorpy/phasorpy/pull/140
* Simplify multiple harmonic calibration by @bruno-pannunzio in https://github.com/phasorpy/phasorpy/pull/124
* Add documentation version switcher dropdown by @cgohlke in https://github.com/phasorpy/phasorpy/pull/136
* Mention AlliGator software by @cgohlke in https://github.com/phasorpy/phasorpy/pull/141
* Bump pypa/cibuildwheel from 2.21.1 to 2.21.3 in the github-actions group by @dependabot in https://github.com/phasorpy/phasorpy/pull/144
* Add tool to print SHA256 hashes of dataset files by @cgohlke in https://github.com/phasorpy/phasorpy/pull/143
* Add Convallaria dataset by @bruno-pannunzio in https://github.com/phasorpy/phasorpy/pull/145
* Mention LIFA software by @cgohlke in https://github.com/phasorpy/phasorpy/pull/146
* Upgrade GitHub Actions to macOS-13 environment by @cgohlke in https://github.com/phasorpy/phasorpy/pull/149
* Add spectral vector denoising by @cgohlke in https://github.com/phasorpy/phasorpy/pull/148
* Replace median filter implementation for NaN handling consistency by @bruno-pannunzio in https://github.com/phasorpy/phasorpy/pull/147
* Improve median filter by @cgohlke in https://github.com/phasorpy/phasorpy/pull/150
* Release v0.2 by @cgohlke in https://github.com/phasorpy/phasorpy/pull/151

**Full Changelog**: https://github.com/phasorpy/phasorpy/compare/v0.1...v0.2

0.1 (2024.9.30)
---------------

This is the first alpha release of the PhasorPy library.
It contains over 70 documented and tested functions and class methods to
calculate, calibrate, filter, transform, store, analyze, and visualize
phasor coordinates, as well as to read fluorescence lifetime and hyperspectral
signals from PTU, SDT, LSM, and other file formats.
Ten tutorials demonstrate the use of the programming interface.
An interactive app calculates and plots phasor coordinates of FRET donor and
acceptor channels as a function of many model parameters.
This release supports Python 3.10 to 3.13.

What's Changed
..............

* Create initial project infrastructure by @cgohlke in https://github.com/phasorpy/phasorpy/pull/1
* Mention FLIMLib by @cgohlke in https://github.com/phasorpy/phasorpy/pull/2
* Mention PhasorIdentifier by @cgohlke in https://github.com/phasorpy/phasorpy/pull/3
* Mention PAM by @cgohlke in https://github.com/phasorpy/phasorpy/pull/4
* Add color module by @cgohlke in https://github.com/phasorpy/phasorpy/pull/5
* Add io and datasets modules by @cgohlke in https://github.com/phasorpy/phasorpy/pull/7
* Add datasets and enable mass downloads by @cgohlke in https://github.com/phasorpy/phasorpy/pull/8
* Add link to GitHub repo by @cgohlke in https://github.com/phasorpy/phasorpy/pull/9
* Update .gitignore by @cgohlke in https://github.com/phasorpy/phasorpy/pull/14
* Add link to FLIM LABS GitHub by @cgohlke in https://github.com/phasorpy/phasorpy/pull/16
* Improve contributing guide and create PR template by @cgohlke in https://github.com/phasorpy/phasorpy/pull/15
* Update workflows by @cgohlke in https://github.com/phasorpy/phasorpy/pull/18
* Enable Dependabot version updates for actions by @cgohlke in https://github.com/phasorpy/phasorpy/pull/22
* Bump the github-actions group with 3 updates by @dependabot in https://github.com/phasorpy/phasorpy/pull/23
* Update copyright year by @cgohlke in https://github.com/phasorpy/phasorpy/pull/24
* Add read functions for PTU, FBD, and FLIF files by @cgohlke in https://github.com/phasorpy/phasorpy/pull/25
* Fix target-version for black 24.1 by @cgohlke in https://github.com/phasorpy/phasorpy/pull/29
* Mention tttrlib by @cgohlke in https://github.com/phasorpy/phasorpy/pull/30
* Add calibration functions to the phasor module by @bruno-pannunzio in https://github.com/phasorpy/phasorpy/pull/28
* Various additions and improvements to the phasor module by @cgohlke in https://github.com/phasorpy/phasorpy/pull/32
* Fix datasets.fetch with pooch 1.8.1 by @cgohlke in https://github.com/phasorpy/phasorpy/pull/34
* Add phasor_from_signal function by @cgohlke in https://github.com/phasorpy/phasorpy/pull/35
* Mention code contributions by @cgohlke in https://github.com/phasorpy/phasorpy/pull/38
* Add plot module by @cgohlke in https://github.com/phasorpy/phasorpy/pull/36
* Fix PhasorPlot.semicircle changes axes limits by @cgohlke in https://github.com/phasorpy/phasorpy/pull/39
* Fix contour offsets by @cgohlke in https://github.com/phasorpy/phasorpy/pull/40
* Higher level calibration function by @bruno-pannunzio in https://github.com/phasorpy/phasorpy/pull/37
* Hide typehints in API documentation by @cgohlke in https://github.com/phasorpy/phasorpy/pull/41
* Add skip_axes parameter to phasor_calibrate function by @bruno-pannunzio in https://github.com/phasorpy/phasorpy/pull/42
* Additions and improvements to the phasor module by @cgohlke in https://github.com/phasorpy/phasorpy/pull/44
* Improve typing by @cgohlke in https://github.com/phasorpy/phasorpy/pull/45
* Add dtime parameter to read_ptu function by @cgohlke in https://github.com/phasorpy/phasorpy/pull/46
* Add phasor_from_fret functions by @cgohlke in https://github.com/phasorpy/phasorpy/pull/49
* Bump the github-actions group with 1 update by @dependabot in https://github.com/phasorpy/phasorpy/pull/50
* Use Scientific Python SPEC0 instead of NEP29 by @cgohlke in https://github.com/phasorpy/phasorpy/pull/51
* Add interactive FRET phasor plot by @cgohlke in https://github.com/phasorpy/phasorpy/pull/52
* Add PhasorPlot.cursor method by @cgohlke in https://github.com/phasorpy/phasorpy/pull/53
* Fix linting errors in Cython code by @cgohlke in https://github.com/phasorpy/phasorpy/pull/54
* Improve phasor_from_lifetime tutorial by @cgohlke in https://github.com/phasorpy/phasorpy/pull/55
* Add functions to convert optimal frequency and lifetime by @cgohlke in https://github.com/phasorpy/phasorpy/pull/56
* Mention napari-live-flim by @cgohlke in https://github.com/phasorpy/phasorpy/pull/57
* Mention HySP software by @cgohlke in https://github.com/phasorpy/phasorpy/pull/58
* Add functions to convert between lifetime fractions and amplitudes by @cgohlke in https://github.com/phasorpy/phasorpy/pull/60
* Add components module by @bruno-pannunzio in https://github.com/phasorpy/phasorpy/pull/59
* Support NumPy 2 by @cgohlke in https://github.com/phasorpy/phasorpy/pull/61
* Build with NumPy 2 by @cgohlke in https://github.com/phasorpy/phasorpy/pull/62
* Bump peaceiris/actions-gh-pages from 3 to 4 in the github-actions group by @dependabot in https://github.com/phasorpy/phasorpy/pull/64
* Format docstring examples with blackdoc by @cgohlke in https://github.com/phasorpy/phasorpy/pull/66
* Add phasor_at_harmonic function by @cgohlke in https://github.com/phasorpy/phasorpy/pull/65
* Fix phasor_calibrate function to handle multi harmonic calibration by @bruno-pannunzio in https://github.com/phasorpy/phasorpy/pull/69
* Mention FLIMfit software by @cgohlke in https://github.com/phasorpy/phasorpy/pull/70
* Fix spelling by @cgohlke in https://github.com/phasorpy/phasorpy/pull/72
* Bump pypa/cibuildwheel from 2.17.0 to 2.18.1 in the github-actions group by @dependabot in https://github.com/phasorpy/phasorpy/pull/71
* Numpy 2 is released by @cgohlke in https://github.com/phasorpy/phasorpy/pull/73
* Mention HORIBA EzTime software by @cgohlke in https://github.com/phasorpy/phasorpy/pull/75
* Use phasorpy-data repo instead of Zenodo in GitHub Actions by @cgohlke in https://github.com/phasorpy/phasorpy/pull/74
* Enable code coverage via codecov.io by @cgohlke in https://github.com/phasorpy/phasorpy/pull/76
* Bump pypa/cibuildwheel from 2.18.1 to 2.19.1 in the github-actions group by @dependabot in https://github.com/phasorpy/phasorpy/pull/77
* Seed random number generator with constant in tutorials by @cgohlke in https://github.com/phasorpy/phasorpy/pull/80
* Add graphical component analysis in components module by @bruno-pannunzio in https://github.com/phasorpy/phasorpy/pull/79
* Add cursors module by @schutyb in https://github.com/phasorpy/phasorpy/pull/48
* Add geometric helper functions by @cgohlke in https://github.com/phasorpy/phasorpy/pull/81
* Improve cursors module by @cgohlke in https://github.com/phasorpy/phasorpy/pull/82
* Add function to project multi-harmonic phasor coordinates onto principal plane by @cgohlke in https://github.com/phasorpy/phasorpy/pull/78
* Add elliptic cursors by @cgohlke in https://github.com/phasorpy/phasorpy/pull/84
* Add phasor_to_signal function by @cgohlke in https://github.com/phasorpy/phasorpy/pull/86
* Add median filtering function by @bruno-pannunzio in https://github.com/phasorpy/phasorpy/pull/85
* Bump pypa/cibuildwheel from 2.19.1 to 2.19.2 in the github-actions group by @dependabot in https://github.com/phasorpy/phasorpy/pull/89
* Sort fractions in PhasorPlot.components by @cgohlke in https://github.com/phasorpy/phasorpy/pull/90
* Fix type of harmonic parameter in phasor_to_signal by @cgohlke in https://github.com/phasorpy/phasorpy/pull/91
* Add LFD workshop FLIM tutorial by @cgohlke in https://github.com/phasorpy/phasorpy/pull/63
* Add lifetime_to_signal function by @cgohlke in https://github.com/phasorpy/phasorpy/pull/93
* Use Cython>=3.0.11 by @cgohlke in https://github.com/phasorpy/phasorpy/pull/94
* Fix phasor_center with NaN input by @cgohlke in https://github.com/phasorpy/phasorpy/pull/96
* Fix RuntimeWarning in plot_phasor_image by @cgohlke in https://github.com/phasorpy/phasorpy/pull/97
* Unify phasor_from_signal functions by @cgohlke in https://github.com/phasorpy/phasorpy/pull/98
* Add phasor_threshold function to phasor module by @bruno-pannunzio in https://github.com/phasorpy/phasorpy/pull/88
* Fix undefined and not defined function names in _utils module by @bruno-pannunzio in https://github.com/phasorpy/phasorpy/pull/100
* Improve handling of NaN values by @cgohlke in https://github.com/phasorpy/phasorpy/pull/101
* Add default fractions to graphical_component_analysis by @bruno-pannunzio in https://github.com/phasorpy/phasorpy/pull/103
* Improve tutorials by @cgohlke in https://github.com/phasorpy/phasorpy/pull/102
* Support writing multi-harmonic phasors to OME-TIFF by @cgohlke in https://github.com/phasorpy/phasorpy/pull/104
* Bump pypa/cibuildwheel from 2.19.2 to 2.20.0 in the github-actions group by @dependabot in https://github.com/phasorpy/phasorpy/pull/107
* Do not test wheels on Python 3.13 for now by @cgohlke in https://github.com/phasorpy/phasorpy/pull/108
* Add dataset from zenodo.org/records/13625087 by @cgohlke in https://github.com/phasorpy/phasorpy/pull/109
* Test minimum runtime requirements by @cgohlke in https://github.com/phasorpy/phasorpy/pull/110
* Add EOSS badge and use recommended language by @cgohlke in https://github.com/phasorpy/phasorpy/pull/111
* Improve io module by @cgohlke in https://github.com/phasorpy/phasorpy/pull/112
* Disable scikit-learn in requirements_min.txt for now by @cgohlke in https://github.com/phasorpy/phasorpy/pull/113
* Configure pre-commit hooks by @cgohlke in https://github.com/phasorpy/phasorpy/pull/114
* Enable mypy strict mode by @cgohlke in https://github.com/phasorpy/phasorpy/pull/115
* Configure mypy enable_error_code by @cgohlke in https://github.com/phasorpy/phasorpy/pull/116
* Configure mypy for tests and tutorials by @cgohlke in https://github.com/phasorpy/phasorpy/pull/117
* Revise phasor_from_ometiff by @cgohlke in https://github.com/phasorpy/phasorpy/pull/119
* Increase API documentation toctree depth by @cgohlke in https://github.com/phasorpy/phasorpy/pull/120
* Mention FLIMPA software by @cgohlke in https://github.com/phasorpy/phasorpy/pull/121
* Publish docs in subfolder by @cgohlke in https://github.com/phasorpy/phasorpy/pull/123
* Use Sphinx dirhtml by @cgohlke in https://github.com/phasorpy/phasorpy/pull/125
* Add links to PyPI and Zenodo to readme by @cgohlke in https://github.com/phasorpy/phasorpy/pull/126
* Update project URLs by @cgohlke in https://github.com/phasorpy/phasorpy/pull/127
* Update pull request template by @cgohlke in https://github.com/phasorpy/phasorpy/pull/128
* Change release-pypi to build_sdist workflow by @cgohlke in https://github.com/phasorpy/phasorpy/pull/129
* Enable testing wheels on Python 3.13 by @cgohlke in https://github.com/phasorpy/phasorpy/pull/131
* Release v0.1 by @cgohlke in https://github.com/phasorpy/phasorpy/pull/130

New Contributors
................

* @cgohlke made their first contribution in https://github.com/phasorpy/phasorpy/pull/1
* @dependabot made their first contribution in https://github.com/phasorpy/phasorpy/pull/23
* @bruno-pannunzio made their first contribution in https://github.com/phasorpy/phasorpy/pull/28
* @schutyb made their first contribution in https://github.com/phasorpy/phasorpy/pull/48

**Full Changelog**: https://github.com/phasorpy/phasorpy/commits/v0.1
