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

from tools import MyShelf, pfaffian
from solver import solve_scattering_equations, mandelstams
from integrands import A, Psi, Cyc, Phi, W1


mpmath.mp.dps = 300


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


class NumericalAmplitude(object):

    def __init__(self, theory, helconf):
        self.theory = theory
        self.helconf = helconf
        self.__name__ = self.theory + "/" + self.helconf.replace("+", "p").replace("-", "m")
        self.process_name = self.theory + "_" + self.helconf.replace("+", "p").replace("-", "m")

        self.multiplicity = len(helconf)
        self.zs = sympy.symbols('z1:{}'.format(self.multiplicity + 1))
        self.ks = sympy.symbols('k1:{}'.format(self.multiplicity + 1))
        self.es = sympy.symbols('e1:{}'.format(self.multiplicity + 1))
        self.ss = mandelstams(self.multiplicity)
        self.get_call_cache()

        # A
        self.A = A(self.multiplicity)
        self.sA = self.A.tolist()
        self.sA = [[ee_sub(ke_sub(ek_sub(kk_sub(zz_sub(str(entry)))))) for entry in line] for line in self.sA]
        # Psi
        self.Psi = Psi(self.multiplicity)
        self.sPsi = self.Psi.tolist()
        self.sPsi = [[ee_sub(ke_sub(ek_sub(kk_sub(zz_sub(str(entry)))))) for entry in line] for line in self.sPsi]
        # Cyc
        self.Cyc = Cyc(self.multiplicity)
        self.sCyc = zz_sub(str(self.Cyc))
        # Phi
        self.Phi = Phi(self.multiplicity)
        self.sPhi = self.Phi.tolist()
        self.sPhi = [[(kk_sub(zz_sub(str(entry)))) for entry in line] for line in self.sPhi]
        # W1
        self.W1 = W1(self.multiplicity)
        self.sW1 = ke_sub(ek_sub(zz_sub(str(self.W1))))

    def solve_SE(self, oParticles):
        num_ss = map(oParticles.compute, map(str, self.ss))
        dict_ss = {str(self.ss[i]): num_ss[i] for i in range(len(self.ss))}
        return solve_scattering_equations(self.multiplicity, dict_ss)

    def nPfPsi(self, sol, oParticles):
        nPsi = numpy.array([[eval(entry, None) for entry in line] for line in self.sPsi])
        Pf = pfaffian(nPsi)
        return Pf / 2

    def nPfA(self, sol, oParticles):
        nPfA = numpy.array([[eval(entry, None) for entry in line] for line in self.sA])
        Pf = pfaffian(nPfA)
        return Pf / 2

    def nCyc(self, sol):
        return mpmath.mpc(eval(self.sCyc, None))

    def nW1(self, sol, oParticles):
        return mpmath.mpc(eval(self.sW1, None))

    def detJ(self, sol, oParticles):
        return mpmath.det(mpmath.matrix([[eval(entry, None) for entry in line] for line in self.sPhi])) if self.sPhi != [] else 1

    def _evaluate(self, oParticles):
        num_sols = self.solve_SE(oParticles)
        oParticles.helconf = self.helconf
        if self.theory == "YM":
            res = sum([self.nCyc(num_sol) * self.nPfPsi(num_sol, oParticles) / self.detJ(num_sol, oParticles) for num_sol in num_sols])
        elif self.theory == "GR":
            res = sum([(self.nPfPsi(num_sol, oParticles) ** 2) / self.detJ(num_sol, oParticles) for num_sol in num_sols])
        elif self.theory == "BS":
            res = sum([(self.nCyc(num_sol) ** 2) / self.detJ(num_sol, oParticles) for num_sol in num_sols])
        elif self.theory == "BI":
            res = sum([(self.nPfPsi(num_sol, oParticles) * self.nPfA(num_sol, oParticles) ** 2) / self.detJ(num_sol, oParticles) for num_sol in num_sols])
        elif self.theory == "NLSM":
            res = sum([(self.nCyc(num_sol) * self.nPfA(num_sol, oParticles) ** 2) / self.detJ(num_sol, oParticles) for num_sol in num_sols])
        elif self.theory == "Galileon":
            res = sum([(self.nPfA(num_sol, oParticles) ** 4) / self.detJ(num_sol, oParticles) for num_sol in num_sols])
        elif self.theory == "CG":
            res = sum([mpmath.sqrt(2) ** self.multiplicity * (
                self.nW1(num_sol, oParticles) * self.nPfPsi(num_sol, oParticles)) / self.detJ(num_sol, oParticles) for num_sol in num_sols])
        elif self.theory == "DF2":
            res = sum([(self.nW1(num_sol, oParticles) * self.nCyc(num_sol)) / self.detJ(num_sol, oParticles) for num_sol in num_sols])
        else:
            raise Exception("Theory not understood")
        res = res / (1j * mpmath.sqrt(2) ** (self.multiplicity - 2))
        return res

    def __call__(self, oParticles):
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
        if "dCallCache_" + self.process_name not in dir(CHYUnknown):
            self.reload_call_cache()
        else:
            self.dCallCache = getattr(CHYUnknown, "dCallCache_" + self.process_name)
        self.len_keys_call_cache_at_last_save = len(self.dCallCache.keys())

    def reload_call_cache(self):
        if not os.path.exists(self.call_cache_path):
            os.makedirs(self.call_cache_path)
        with MyShelf(self.call_cache_path + "/" + self.process_name, 'c') as persistentCallCache:
            dCallCache = multiprocessing.Manager().dict(dict(persistentCallCache))
        setattr(CHYUnknown, "dCallCache_" + self.process_name, dCallCache)
        self.dCallCache = dCallCache

    def save_call_cache(self):
        if len(self.dCallCache.keys()) > self.len_keys_call_cache_at_last_save:
            self.len_keys_call_cache_at_last_save = len(self.dCallCache.keys())
            with MyShelf(self.call_cache_path + "/" + self.process_name, 'c') as persistentCallCache:
                persistentCallCache.update(self.dCallCache)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


ps_ij = re.compile("(s_\d*)")

pattern_kk = re.compile(r"k(\d)\*k(\d)")
kk_sub = functools.partial(pattern_kk.sub, r"oParticles.compute('s_\1\2') / 2")

pattern_ek = re.compile(r"e(\d)\*k(\d)")
ek_sub = functools.partial(pattern_ek.sub, r"oParticles.ep(\1, \2)")

pattern_ke = re.compile(r"k(\d)\*e(\d)")
ke_sub = functools.partial(pattern_ke.sub, r"oParticles.pe(\1, \2)")

pattern_ee = re.compile(r"e(\d)\*e(\d)")
ee_sub = functools.partial(pattern_ee.sub, r"oParticles.ee(\1, \2)")

pattern_zz = re.compile(r"(z\d)")
zz_sub = functools.partial(pattern_zz.sub, r"sol[str(sympy.symbols('\1'))]")
