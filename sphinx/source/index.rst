.. seampy documentation master file, created by
   sphinx-quickstart on Tue Sep 24 12:11:24 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to seampy's documentation!
==================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

seampy (Scattering Equations AMplitudes with PYthon)


Installing seampy
=================

Installation is easy with pip::

  pip install seampy

it also requires the phase space package lips::

  pip install lips-lite

alternatively the package can be cloned from github.


seampy QuickStart
=================

To get started computing amplitudes open an interactive python session and follow this simple example::

	$ ipython

	In [1]: import seampy, lips

	In [2]: oParticles = lips.Particles(6)

	In [3]: oParticles.fix_mom_cons()

	In [4]: oNumAmp = seampy.NumericalAmplitude(theory="DF2", helconf="pppmmm")

	In [5]: oNumAmp(oParticles)

the output will be a complex number	


Then you should get::

	Finished: An initial directory structure has been created.

	You should now populate your master file .\source\index.rst and create other documentation
	source files. Use the sphinx-build command to build the docs, like so:
	   sphinx-build -b builder .\source .\build
	where "builder" is one of the supported builders, e.g. html, latex or linkcheck.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
