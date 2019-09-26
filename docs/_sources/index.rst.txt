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

To get started computing amplitudes open an (interactive) python session and follow this simple example:

  .. code-block:: python
		  :linenos:
		   
		     import seampy, lips

		     # generate phase space point
		     oParticles = lips.Particles(6)
		     oParticles.fix_mom_cons()

		     # compute gauge or gravity amplitude: give helconf
		     oYMAmp = NumericalAmplitude(theory="YM", helconf="pmpmpm")
		     oYMAmp(oParticles)  # returns a complex number

		     # compute scalar amplitude: give multiplicity of phase space
		     oBSAmp = NumericalAmplitude(theory="BS", multiplicity=6)
		     oBSAmp(oParticles)  # returns a complex number


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
