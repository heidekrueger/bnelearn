# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys
sys.path.insert(0, os.path.abspath('../bnelearn/'))

print(f"abs.path = {os.path.abspath('.')}")

print(sys.path)

# -- Project information -----------------------------------------------------

project = 'bnelearn'
copyright = '2022, Institute for Decision Sciences and Systems, Technical University of Munich'
author = 'Chair for Decision Sciences and Systems, TUM'

# The full version, including alpha/beta/rc tags
release = '1.0.0'

# -- General configuration ---------------------------------------------------

import bnelearn

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

extensions = [
        "sphinx.ext.autodoc"
        ]

#source_suffix = {
#    '.rst': 'restructuredtext',
#    '.txt': 'restructuredtext',
#    '.py': 'restructuredtext'
#}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_logo = "bnelearn-gray.png"


def run_apidoc(_):
    modules = [os.path.abspath('../bnelearn/')]
    for module in modules:
        cur_dir = os.path.abspath(os.path.dirname(__file__))
        output_path = os.path.join(cur_dir, module, 'doc')
        cmd_path = 'sphinx-apidoc'
        if hasattr(sys, 'real_prefix'):  # Check to see if we are in a virtualenv
            # If we are, assemble the path manually
            cmd_path = os.path.abspath(os.path.join(sys.prefix, 'bin', 'sphinx-apidoc'))
        subprocess.check_call([cmd_path, '-e', '-o', output_path, module, '--force'])

def setup(app):
    app.connect('builder-inited', run_apidoc)

#display private members 
#autodoc_default_options = {     "members": True,     "undoc-members": True, "private-members": True  }

def setup(app):
    app.connect('builder-inited', run_apidoc)
    app.add_css_file('css/modify.css')
