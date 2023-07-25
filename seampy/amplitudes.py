#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Giuseppe

from __future__ import unicode_literals
from __future__ import print_function

import numpy
import os
import re
import multiprocessing
import sympy
import mpmath
import functools

from .tools import MyShelf, pfaffian
from .solver import solve_scattering_equations, mandelstams
from .integrands import A, Psi, Cyc, Phi, W1


mpmath.mp.dps = 300


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


theories = ["YM", "EG", "BS", "BI", "NLSM", "Galileon", "CG", "DF2"]
scalar_theories = ["BS", "NLSM", "Galileon"]


class NumericalAmplitude(object):
    """
    | NumericalAmplitude provides a callable object to compute scattering amplitudes.
    | To call provide a phase space point from the lips package.

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

    """

    def __init__(self, theory, helconf=None, multiplicity=None):
        if theory not in theories:
            raise Exception("Theory not understood.")
        self.theory = theory
        if theory in scalar_theories:
            if helconf is not None:
                raise Exception("Scalar theories don't need an helicity configuration.")
            self.multiplicity = int(multiplicity)
            self.__name__ = self.theory + "/" + str(self.multiplicity) + "pt"
            self.process_name = self.theory + "_" + str(self.multiplicity) + "pt"
        else:
            if helconf is None:
                raise Exception("Gauge and gravity theories require an helicity configuration.")
            assert multiplicity is None or multiplicity == len(helconf)
            self.multiplicity = len(helconf)
            self.helconf = helconf.replace("+", "p").replace("-", "m")
            self.__name__ = self.theory + "/" + str(self.multiplicity) + "pt/" + self.helconf
            self.process_name = self.theory + "_" + self.helconf

        self.zs = sympy.symbols('z1:{}'.format(self.multiplicity + 1))
        self.ks = sympy.symbols('k1:{}'.format(self.multiplicity + 1))
        self.es = sympy.symbols('e1:{}'.format(self.multiplicity + 1))
        self.ss = mandelstams(self.multiplicity)
        self.get_call_cache()

        # A
        self.A = A(self.multiplicity)
        self.sA = self.A.tolist()
        self.sA = [[_ee_sub(_ke_sub(_ek_sub(_kk_sub(_zz_sub(str(entry)))))) for entry in line] for line in self.sA]

        # Psi
        self.Psi = Psi(self.multiplicity)
        self.sPsi = self.Psi.tolist()
        self.sPsi = [[_ee_sub(_ke_sub(_ek_sub(_kk_sub(_zz_sub(str(entry)))))) for entry in line] for line in self.sPsi]

        # Cyc
        self.Cyc = Cyc(self.multiplicity)
        self.sCyc = _zz_sub(str(self.Cyc))

        # Phi
        self.Phi = Phi(self.multiplicity)
        self.sPhi = self.Phi.tolist()
        self.sPhi = [[(_kk_sub(_zz_sub(str(entry)))) for entry in line] for line in self.sPhi]

        # W1
        self.W1 = W1(self.multiplicity)
        self.sW1 = _ke_sub(_ek_sub(_zz_sub(str(self.W1))))

    def solve_se(self, oParticles):
        """Interface to seampy.solver.solve_scattering_equations. Takes a phase space point as input."""
        dict_ss = {str(s): oParticles.compute(str(s)) for s in self.ss}
        return solve_scattering_equations(self.multiplicity, dict_ss)

    def nPfPsi(self, sol, oParticles):
        """Numerical pfaffian of reduced Psi."""
        locs = locals()
        nPsi = numpy.array([[eval(entry, None, locs) for entry in line] for line in self.sPsi])
        Pf = pfaffian(nPsi)
        return Pf / 2

    def nPfA(self, sol, oParticles):
        """Numerical pfaffian of reduced A."""
        locs = locals()
        nPfA = numpy.array([[eval(entry, None, locs) for entry in line] for line in self.sA])
        Pf = pfaffian(nPfA)
        return Pf / 2

    def nCyc(self, sol):
        """Numerical cyclic Parke-Taylor-like factor."""
        locs = locals()
        return mpmath.mpc(eval(self.sCyc, None, locs))

    def nW1(self, sol, oParticles):
        """Numerical W1 (integrand for DF2 and CG)."""
        locs = locals()
        return mpmath.mpc(eval(self.sW1, None, locs))

    def detJ(self, sol, oParticles):
        """Numerical determinant of reduced Jacobian matrix Phi."""
        locs = locals()
        return mpmath.det(mpmath.matrix([[eval(entry, None, locs) for entry in line] for line in self.sPhi])) if self.sPhi != [] else 1

    def _evaluate(self, oParticles):
        num_sols = self.solve_se(oParticles)
        if hasattr(self, "helconf"):
            oParticles.helconf = self.helconf
        if self.theory == "YM":
            nor = 1j / (mpmath.sqrt(2) ** (self.multiplicity - 2))
            res = sum([self.nCyc(num_sol) * self.nPfPsi(num_sol, oParticles) / self.detJ(num_sol, oParticles) for num_sol in num_sols])
        elif self.theory == "EG":
            nor = 1j / 2 ** (self.multiplicity - 2)
            res = sum([(self.nPfPsi(num_sol, oParticles) ** 2) / self.detJ(num_sol, oParticles) for num_sol in num_sols])
        elif self.theory == "BS":
            nor = 1j
            res = sum([(self.nCyc(num_sol) ** 2) / self.detJ(num_sol, oParticles) for num_sol in num_sols])
        elif self.theory == "BI":
            nor = 1j / mpmath.sqrt(2) ** (self.multiplicity - 6)
            res = sum([(self.nPfPsi(num_sol, oParticles) * self.nPfA(num_sol, oParticles) ** 2) / self.detJ(num_sol, oParticles) for num_sol in num_sols])
        elif self.theory == "NLSM":
            nor = 1j * 4
            res = sum([(self.nCyc(num_sol) * self.nPfA(num_sol, oParticles) ** 2) / self.detJ(num_sol, oParticles) for num_sol in num_sols])
        elif self.theory == "Galileon":
            nor = 1j * 16
            res = sum([(self.nPfA(num_sol, oParticles) ** 4) / self.detJ(num_sol, oParticles) for num_sol in num_sols])
        elif self.theory == "CG":
            nor = 1j * 2
            res = sum([(self.nW1(num_sol, oParticles) * self.nPfPsi(num_sol, oParticles)) / self.detJ(num_sol, oParticles) for num_sol in num_sols])
        elif self.theory == "DF2":
            nor = 1j * mpmath.sqrt(2) ** self.multiplicity
            res = sum([(self.nW1(num_sol, oParticles) * self.nCyc(num_sol)) / self.detJ(num_sol, oParticles) for num_sol in num_sols])
        return nor * res

    def __call__(self, oParticles):
        """__call__ documentation"""
        # look up in call cache
        if str(hash(oParticles)) in self.dCallCache:
            return self.dCallCache[str(str(hash(oParticles)))]
        # else compute and save to cache
        res = self._evaluate(oParticles)
        self.dCallCache[str(str(hash(oParticles)))] = res
        return res

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    @property
    def call_cache_path(self):
        return os.path.dirname(__file__) + "/.cache/call_caches"

    def get_call_cache(self):
        if "dCallCache_" + self.process_name not in dir(NumericalAmplitude):
            self.reload_call_cache()
        else:
            self.dCallCache = getattr(NumericalAmplitude, "dCallCache_" + self.process_name)
        self.len_keys_call_cache_at_last_save = len(self.dCallCache.keys())

    def reload_call_cache(self):
        if not os.path.exists(self.call_cache_path):
            os.makedirs(self.call_cache_path)
        with MyShelf(self.call_cache_path + "/" + self.process_name, 'c') as persistentCallCache:
            dCallCache = multiprocessing.Manager().dict(dict(persistentCallCache))
        setattr(NumericalAmplitude, "dCallCache_" + self.process_name, dCallCache)
        self.dCallCache = dCallCache

    def save_call_cache(self):
        if len(self.dCallCache.keys()) > self.len_keys_call_cache_at_last_save:
            self.len_keys_call_cache_at_last_save = len(self.dCallCache.keys())
            with MyShelf(self.call_cache_path + "/" + self.process_name, 'c') as persistentCallCache:
                persistentCallCache.update(self.dCallCache)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


ps_ij = re.compile(r"(s_\d*)")

pattern_kk = re.compile(r"k(\d)\*k(\d)")
_kk_sub = functools.partial(pattern_kk.sub, r"oParticles.compute('s_\1\2') / 2")

pattern_ek = re.compile(r"e(\d)\*k(\d)")
_ek_sub = functools.partial(pattern_ek.sub, r"oParticles.ep(\1, \2)")

pattern_ke = re.compile(r"k(\d)\*e(\d)")
_ke_sub = functools.partial(pattern_ke.sub, r"oParticles.pe(\1, \2)")

pattern_ee = re.compile(r"e(\d)\*e(\d)")
_ee_sub = functools.partial(pattern_ee.sub, r"oParticles.ee(\1, \2)")

pattern_zz = re.compile(r"(z\d)")
_zz_sub = functools.partial(pattern_zz.sub, r"sol[str(sympy.symbols('\1'))]")
