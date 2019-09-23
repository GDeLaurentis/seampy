#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Giuseppe

from __future__ import unicode_literals
from __future__ import print_function

import numpy
import os
import re
import multiprocessing
import itertools
import sympy
import mpmath
import functools
import operator

from tools import MyShelf
from scattering_equations_solver import solve_scattering_equations


mpmath.mp.dps = 300


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


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def A(n):
    """A matrix (reduced)"""
    ks = sympy.symbols('k1:{}'.format(n + 1))
    zs = sympy.symbols('z1:{}'.format(n + 1))

    A = sympy.zeros(n, n)
    for a in range(n):
        for b in range(a + 1, n):
            A[a, b] = 2 * ks[a] * ks[b] / (zs[a] - zs[b])
            A[b, a] = - A[a, b]

    A = A[range(2, n), range(2, n)]

    # Mobius fix
    A = A.subs(((zs[0], sympy.oo), (zs[1], 1), (zs[n - 1], 0)))

    return A


def Psi(n):
    """Psi anti-symmetric matrix (argument of Pfaffian)"""
    ks = sympy.symbols('k1:{}'.format(n + 1))
    es = sympy.symbols('e1:{}'.format(n + 1))
    zs = sympy.symbols('z1:{}'.format(n + 1))

    # A, B, C & Psi Matrices

    # A Matrix
    A = sympy.zeros(n, n)
    for a in range(n):
        for b in range(a + 1, n):
            A[a, b] = 2 * ks[a] * ks[b] / (zs[a] - zs[b])
            A[b, a] = - A[a, b]
    # print("A:")
    # pprint(A)

    # B Matrix
    B = sympy.zeros(n, n)
    for a in range(n):
        for b in range(a + 1, n):
            B[a, b] = 2 * es[a] * es[b] / (zs[a] - zs[b])
            B[b, a] = - B[a, b]
    # print("\nB:")
    # pprint(B)

    # C Matrix
    C = sympy.zeros(n, n)
    for a in range(n):
        for b in range(n):
            if a == b:
                C[a, b] = - sum([2 * es[a] * ks[j] / (zs[a] - zs[j]) for j in range(n) if j != a])
            else:
                C[a, b] = 2 * es[a] * ks[b] / (zs[a] - zs[b])
    # print("\nC:")
    # pprint(C)

    # Psi - from block matrix
    Psi = sympy.Matrix(sympy.BlockMatrix(((A, -C.T), (C, B))))

    # drop rows & cols 1 and 2 (i.e. 0 and 1)
    Psi = Psi[range(2, 2 * n), range(2, 2 * n)]

    for i in range(1, n):
        Psi = Psi.subs(((zs[0] - zs[i], zs[0]), (zs[i] - zs[0], - zs[0]), (zs[i] + zs[0], zs[0]), (- zs[i] - zs[0], - zs[0])))
    # pull out z1 ** -2  ---  Pfaffian goes as root of this
    for i in range(2 * n - 2):
        if Psi[-n, i] != 0:
            Psi[-n, i] = Psi[-n, i] * zs[0]
        if Psi[i, -n] != 0:
            Psi[i, -n] = Psi[i, -n] * zs[0]
    # Mobius fix
    Psi = Psi.subs(((zs[0], sympy.oo), (zs[1], 1), (zs[n - 1], 0)))
    # simplify
    for i in range(2, n):
        Psi = Psi.subs(((sympy.oo - zs[i], sympy.oo), ))
        Psi = Psi.subs(((- sympy.oo + zs[i], - sympy.oo), ))
        Psi = Psi.subs(((sympy.oo + zs[i], sympy.oo), ))

    return Psi


def Cyc(n):
    """Parke-Taylor-like cyclic factor"""
    zs = sympy.symbols('z1:{}'.format(n + 1))

    Cyc = -1 / reduce(operator.mul, [(zs[i] - zs[i + 1]) for i in range(1, n - 1)])
    # Mobius fix
    Cyc = Cyc.subs(((zs[0], sympy.oo), (zs[1], 1), (zs[n - 1], 0)))
    # simplify
    for i in range(1, n):
        Cyc = Cyc.subs(((sympy.oo - zs[i], sympy.oo), ))
        Cyc = Cyc.subs(((- sympy.oo + zs[i], - sympy.oo), ))
        Cyc = Cyc.subs(((sympy.oo + zs[i], sympy.oo), ))

    return Cyc


def Phi(n):
    """SE Jacobian matrix"""
    ks = sympy.symbols('k1:{}'.format(n + 1))
    zs = sympy.symbols('z1:{}'.format(n + 1))

    Phi = sympy.zeros(n, n)
    for a in range(n):
        for b in range(a, n):
            if a == b:
                Phi[a, b] = - sum([2 * ks[a] * ks[j] / (zs[a] - zs[j]) ** 2 for j in range(n) if j != a])
            else:
                Phi[a, b] = 2 * ks[a] * ks[b] / (zs[a] - zs[b]) ** 2
                Phi[b, a] = Phi[a, b]
    # drop rows and columns 1, 2, and n (i.e. 0, 1, n - 1)
    Phi = Phi[range(2, n - 1), range(2, n - 1)]
    # Mobius fix
    Phi = Phi.subs(((zs[0], sympy.oo), (zs[1], 1), (zs[n - 1], 0)))
    # simplify
    for i in range(2, n):
        Phi = Phi.subs(((sympy.oo - zs[i], sympy.oo), ))
        Phi = Phi.subs(((- sympy.oo + zs[i], - sympy.oo), ))
        Phi = Phi.subs(((sympy.oo + zs[i], sympy.oo), ))

    return Phi


def W1(n):
    ks = sympy.symbols('k1:{}'.format(n + 1), commutative=False)
    es = sympy.symbols('e1:{}'.format(n + 1), commutative=False)
    zs = sympy.symbols('z1:{}'.format(n + 1), commutative=True)

    # W_11...11
    def omega(i):
        r = i + 1 if i != n - 1 else 0
        return - sum([es[i] * ks[j] * (zs[j] - zs[r]) / ((zs[r] - zs[i]) * (zs[i] - zs[j])) for j in range(n) if j != i])

    W1 = reduce(operator.mul, [omega(i) for i in range(n)])

    # Mobius fix:
    # simplify infinities
    for i in range(1, n):
        W1 = W1.subs(((zs[0] - zs[i], zs[0]), ))
        W1 = W1.subs(((-zs[0] + zs[i], - zs[0]), ))
        W1 = W1.subs(((zs[0] + zs[i], zs[0]), ))
        # pull out a factor of z1 ** -2 ( oo ** -2 )
    W1 = W1.subs(((1 / zs[0] ** 2, 1), ))
    # remaining two punctures
    W1 = W1.subs(((zs[1], 1), (zs[n - 1], 0)))

    return W1


def pfaffian(matrix):
    rows, cols = matrix.shape
    if rows == 0 and cols == 0:
        return 1
    else:
        return sum((-1) ** i * matrix[0, i - 1] *
                   pfaffian(matrix[numpy.ix_([j for j in range(cols) if j != 0 and j != i - 1],
                                             [j for j in range(cols) if j != 0 and j != i - 1])]) for i in range(2, cols + 1))


assert(pfaffian(numpy.array([[0, 1], [-1, 0]])) == 1)
assert(pfaffian(numpy.array([[0, 1, 2, 3], [-1, 0, 4, 5], [-2, -4, 0, 6], [-3, -5, -6, 0]])) == 1 * 6 - 2 * 5 + 4 * 3)


class CHYUnknown(object):

    def __init__(self, theory, helconf):
        self.theory = theory
        self.helconf = helconf
        self.__name__ = self.theory + "/" + self.helconf.replace("+", "p").replace("-", "m")
        self.process_name = self.theory + "_" + self.helconf.replace("+", "p").replace("-", "m")

        self.multiplicity = len(helconf)
        self.zs = sympy.symbols('z1:{}'.format(self.multiplicity + 1))
        self.ks = sympy.symbols('k1:{}'.format(self.multiplicity + 1))
        self.es = sympy.symbols('e1:{}'.format(self.multiplicity + 1))
        self.ss = tuple(map(sympy.symbols, ['s_{}'.format("".join(map(str, (1,) + subset))) for i in range(1, self.multiplicity + 1 - 3)
                                            for subset in itertools.combinations(range(2, self.multiplicity), i)]))
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
