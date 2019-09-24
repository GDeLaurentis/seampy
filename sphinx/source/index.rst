.. seampy documentation master file, created by
   sphinx-quickstart on Tue Sep 24 12:11:24 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

seampy
==================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

seampy (Scattering Equations AMplitudes with PYthon) is a Python package to solve the scattering equations and compute scattering amplitudes.
The scattering equations are solved to high floating-point precision by means of elimination theory.
The solutions are then used to build amplitudes in a variety of theories.


Installation
=================

Installation is easy with pip::

  pip install seampy

it also requires the phase space package lips::

  pip install lips

alternatively the package can be cloned from github at https://github.com/GDeLaurentis/seampy.


Quick start
=================

To get started computing amplitudes open an interactive python session and follow this simple example::

	$ ipython

	In [1]: import seampy, lips

	In [2]: oParticles = lips.Particles(6)

	In [3]: oParticles.fix_mom_cons()

	In [4]: oNumAmp = seampy.NumericalAmplitude(theory="DF2", helconf="pppmmm")

	In [5]: oNumAmp(oParticles)

the output will be a complex number with 300 digits of precision by default.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
