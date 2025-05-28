Contributing guide
==================

As a community-maintained project, PhasorPy welcomes contributions in the form
of bug reports, bug fixes, feature implementations, documentation, datasets,
and enhancement proposals.
This document provides information on how to contribute.

The :doc:`code_of_conduct` should be honored by everyone participating in the
PhasorPy community.

Ask for help
------------

To ask questions about the PhasorPy library, open a
`GitHub issue <https://github.com/phasorpy/phasorpy/issues>`_.

Propose enhancements
--------------------

To suggest a new feature or other improvement to the PhasorPy library, open a
`GitHub issue <https://github.com/phasorpy/phasorpy/issues>`_.

Share data files
----------------

The PhasorPy project strives to support reading image and metadata from many
time-resolved and hyperspectral file formats used in bio-imaging.
Consider sharing datasets for testing and use in tutorials, preferably with the
`PhasorPy community on Zenodo <https://zenodo.org/communities/phasorpy/>`_.

Report bugs
-----------

To report a bug in the PhasorPy library, please open a
`GitHub issue <https://github.com/phasorpy/phasorpy/issues>`_
and include the following items in the bug report:

- A minimal, self-contained Python code reproducing the problem.
  Format the code using markdown, for example::

    ```Python
    import phasorpy
    phasorpy.do_something('my.file')
    ```
- A Python traceback if available, for example::

    ```Python traceback
    Traceback (most recent call last):
    File "<stdin>", line 1, in <module>
    AttributeError: module 'phasorpy' has no attribute 'do_something'
    ```
- Any data files necessary to run the code can be attached to the GitHub issue
  or shared via cloud storage, preferably on
  `Zenodo.org <https://zenodo.org/communities/phasorpy/>`_.

- An explanation why the current behavior is wrong and what is expected
  instead.

- Information how PhasorPy was installed (pip, conda, or other) and the output
  of::

    $ python -m phasorpy versions
    Python-3.13.3
    phasorpy-0.5
    numpy-2.2.6
    ...

Contribute code or documentation
--------------------------------

The source code for the PhasorPy library and documentation is hosted in
a GitHub repository at
`https://github.com/phasorpy/phasorpy <https://github.com/phasorpy/phasorpy>`_.

The repository is based on `git <https://git-scm.com/>`_, a distributed
version control software for tracking changes in the source code files and for
coordinating work among programmers who are collaboratively developing.

PhasorPy uses GitHub's `fork and pull collaborative development model
<https://docs.github.com/en/pull-requests/collaborating-with-pull-requests>`_.
All contributions to the PhasorPy source code and documentation should
be developed in personal forks/copies of the code, and then submitted as
`pull requests <https://github.com/phasorpy/phasorpy/pulls>`_ (PRs).

The PhasorPy project loosely follows the
`Scientific Python <https://learn.scientific-python.org/development/>`_ and
`pyOpenSci <https://www.pyopensci.org/python-package-guide/>`_
development guidelines.

Fork the repository
...................

To work on the PhasorPy source code, fork the repository by pressing the
"Fork" button at
`https://github.com/phasorpy/phasorpy <https://github.com/phasorpy/phasorpy>`_.

Then clone the personal fork to the local machine::

    $ git clone https://github.com/your-user-name/phasorpy.git
    $ cd phasorpy
    $ git remote add upstream https://github.com/phasorpy/phasorpy.git

There are now two remote repositories:
``upstream``, which refers to the PhasorPy repository, and
``origin``, which refers to the personal fork.

Instead of using the git command line application, you may find
`GitHub Desktop <https://desktop.github.com>`_ easier to use.

Create a development environment
................................

A Python compatible C compiler is required for developing the PhasorPy
library. For example, `Visual Studio <https://visualstudio.microsoft.com/>`_,
`Xcode <https://developer.apple.com/xcode/>`_, or gcc.

To work with the PhasorPy source code, it is recommended to set up a Python
virtual environment and install all PhasorPy dependencies in it.
For example, to create a `venv <https://docs.python.org/3/library/venv.html>`_
environment for an existing Python interpreter on POSIX systems, run the
following from within the local phasorpy repository::

    $ mkdir -p ~/pyenv/phasorpy-dev
    $ python -m venv ~/pyenv/phasorpy-dev
    $ source ~/pyenv/phasorpy-dev/bin/activate
    $ pip install -r requirements_dev.txt
    $ pip install -e .

Verify that the development environment is working by running the tests::

    $ python -m pytest -v

Create a branch
...............

Before implementing any changes or submitting a pull request, consider
opening a `GitHub issue <https://github.com/phasorpy/phasorpy/issues>`_
to report the bug fix or feature being worked on.

Synchronize the personal fork with the upstream repository, then create a
new, separate branch for each bug fix or new feature being worked on.
For example::

    $ git checkout main
    $ git fetch upstream
    $ git rebase upstream/main
    $ git push
    $ git checkout -b new-feature-branch
    $ git push -u origin new-feature-branch

This changes the local repository to the "new-feature-branch" branch.
Keep any changes in this branch specific to one bug or feature.

To update this branch with latest code from the PhasorPy repository,
retrieve the changes from the main branch, make a backup of the feature
branch, and perform a rebase::

    $ git fetch upstream
    $ git checkout new-feature-branch
    $ git branch new-feature-branch-backup new-feature-branch
    $ git rebase upstream/main

This replays local commits at the "new-feature-branch" branch on top
of the latest PhasorPy upstream main branch.
Merge-conflicts need to be resolved before submitting a pull request.

Tests
.....

PhasorPy includes a `pytest <https://docs.pytest.org/>`_ based suite of
unit tests in the ``tests`` folder. All classes and functions must be tested
thorougly.

Run the unit tests in the development environment::

    $ python -m pytest -v --cov=phasorpy --cov-report=html tests

All tests must pass.

PhasorPy strives to maintain near complete code coverage. A coverage report
is generated during testing in the ``_htmlcov`` folder.

Configuration settings for pytest and other tools are in the
``pyproject.toml`` file.

Code standards
..............

All the PhasorPy source code, including tutorials and docstring examples,
must conform to certain styles, which can be and applied with
`pre-commit <https://pre-commit.com/>`_::

    $ python -m pre_commit run --all-files

User-facing classes and functions must use
`type hints <https://peps.python.org/pep-0484/>`_
and pass verification with the
`MyPy <https://mypy.readthedocs.io>`_
static type checker::

    $ python -m mypy


The PhasorPy project follows the Scientific Python
`SPEC 0 — Minimum Supported Dependencies
<https://scientific-python.org/specs/spec-0000/>`_ recommendation
for Python, NumPy, and other dependencies.

Documentation
.............

User-facing classes and functions must contain
`docstrings <https://peps.python.org/pep-0257/>`_
following the `numpydoc
<https://numpydoc.readthedocs.io/en/stable/format.html#docstring-standard>`_
standard.

Examples in docstrings must run and pass as
`doctests <https://docs.python.org/3/library/doctest.html>`_ ::

    $ python -m pytest -v src/phasorpy

Examples in docstrings are meant to illustrate mere usage, not to
provide a testing framework.

PhasorPy uses `Sphinx <https://www.sphinx-doc.org>`_
to generate the documentation in HTML format published at
`phasorpy.org <https://www.phasorpy.org>`_.

Sphinx documentation is written in the
`reStructuredText <https://docutils.sourceforge.io/rst.html>`_
markup language in the .rst files in the ``docs`` and ``tutorials`` folders.

All user-facing classes and functions should be included in the
``docs/api/*.rst`` files.

Any changes should be mentioned in the release notes (``docs/release.rst``).

New features or important usage information should be covered in the
tutorials. Tutorials are included in the documentation via the
`Sphinx-Gallery <https://sphinx-gallery.github.io>`_
extension, which builds an HTML gallery of examples from the set of Python
scripts in the ``tutorials`` folder.

Examples in the .rst files must run and pass as doctests::

    $ python -m pytest -v docs

Documentation in HTML format can be built from the docstrings, .rst,
and tutorial files by running::

    $ cd docs
    $ make clean
    $ make html
    $ open _build/html/index.html

Commit the changes
..................

Commit changed and new files to the local repository::

    $ git add phasorpy/new_file.py
    $ git commit -a -m "Summarize changes in 50 characters or less"

Please do not include binary data, or any files other than source code,
documentation, or project settings in the repository.

Create a pull request
.....................

Push the changes from the local repository back to the personal fork
on GitHub::

    $ git push origin new-feature-branch

Open the personal fork on GitHub::

    $ open https://github.com/your-user-name/phasorpy.git

Click the green "pull request" button on the "new-feature-branch" branch.

Review the pull request checklist for recommendations.

All tests are automatically run via
`GitHub Actions <https://github.com/features/actions>`_ for every pull request
and must pass before code or documentation can be accepted.

Other PhasorPy developers will review the pull request to check and help
to improve its implementation, documentation, and style.

Pull requests must be approved by a
`PhasorPy organization member <https://github.com/orgs/phasorpy/people>`_
before merging.
