# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# import os
# import sys
# sys.path.insert(0, os.path.abspath('../src/'))

# remove the examples header from HTML tutorials
import sphinx_gallery.gen_rst

sphinx_gallery.gen_rst.EXAMPLE_HEADER = (
    sphinx_gallery.gen_rst.EXAMPLE_HEADER.replace(
        '.. only:: html', '.. only:: xml'
    )
)

# project information

project = 'PhasorPy'
copyright = '2022-2023 PhasorPy Contributors'
author = 'PhasorPy Contributors'

import phasorpy

version = phasorpy.__version__
release = phasorpy.__version__

# general configuration

extensions = [
    'sphinx.ext.napoleon',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.viewcode',
    'sphinx.ext.todo',
    # 'sphinx.ext.intersphinx',
    # 'numpydoc',
    'sphinx_copybutton',
    'sphinx_click',
    'sphinx_issues',
    'sphinx_gallery.gen_gallery',
    'pytest_doctestplus.sphinx.doctestplus',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# options for HTML output

# html_theme = 'alabaster'
# html_theme = 'sphinx_rtd_theme'
html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
html_show_sourcelink = False

html_logo = '_static/logo.png'
# html_favicon = ''

pygments_style = 'sphinx'
intersphinx_mapping = {'https://docs.python.org/': None}

# extension configurations

napoleon_google_docstring = False
napoleon_numpy_docstring = True
todo_include_todos = True

html_theme_options = {
    'logo': {
        'text': 'PhasorPy',
        'alt_text': 'PhasorPy',
    },
}

sphinx_gallery_conf = {
    'filename_pattern': 'phasorpy_',
    'examples_dirs': '../tutorials',
    'gallery_dirs': 'tutorials',
}

copybutton_prompt_text = (
    r'>>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: '
)
copybutton_prompt_is_regexp = True
