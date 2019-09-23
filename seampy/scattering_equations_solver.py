#!/usr/bin/env python
# -*- coding: utf-8 -*-

#   ___          _   _           _             ___                _   _               ___      _
#  / __| __ __ _| |_| |_ ___ _ _(_)_ _  __ _  | __|__ _ _  _ __ _| |_(_)___ _ _  ___ / __| ___| |_ _____ _ _
#  \__ \/ _/ _` |  _|  _/ -_) '_| | ' \/ _` | | _|/ _` | || / _` |  _| / _ \ ' \(_-< \__ \/ _ \ \ V / -_) '_|
#  |___/\__\__,_|\__|\__\___|_| |_|_||_\__, | |___\__, |\_,_\__,_|\__|_\___/_||_/__/ |___/\___/_|\_/\___|_|
#                                      |___/         |_|

# Author: Giuseppe
# Date: 11 September 2019


import sympy
import itertools
import operator
import re
import functools
import mpmath
import math
import copy

from tools import flatten

mpmath.mp.dps = 300


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def mandelstams(n):
    """Mandelstam variables appearing in the polynomial scattering equations."""
    return tuple(map(sympy.symbols, ['s_{}'.format("".join(map(str, (1,) + subset))) for i in range(1, n + 1 - 3) for subset in itertools.combinations(range(2, n), i)]))


def punctures(n):
    """Punctures of the Riemann sphere."""
    return sympy.symbols('z1:{}'.format(n + 1))


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def hms(n):
    """Scattering equations in polynomial form."""
    ss = mandelstams(n)
    zs = punctures(n)
    return [sum(ss[map(str, ss).index("s_{}".format("".join(map(str, (1,) + subset))))] * reduce(operator.mul, [zs[j - 1] for j in subset])
                for subset in itertools.combinations(range(2, n), i)) for i in range(1, n + 1 - 3)]


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def M(i, n=None):
    """Elimination theory matrix: i^th matrix in the recursion with n external legs."""
    if n is None:
        n = i
    zs = punctures(n)
    if i < 3:
        raise Exception("Elimination theory matrix: invalid argument.")
    elif i == 3:
        return sympy.Matrix([])
    elif i == 4:
        return sympy.Matrix(hms(n))
    else:
        M_im1 = M(i - 1, n)
        M_im1_diff = sympy.diff(M_im1, zs[i - 3 - 1])
        M_im1 = M_im1.subs(zs[i - 3 - 1], 0)  # just a trick to simplify calculation, nothing to do with Mobius
        rows, columns = M_im1.shape
        M_i = sympy.Matrix(sympy.BlockMatrix(tuple(
            tuple([M_im1 if _i == _j else M_im1_diff if _i == _j + 1 else sympy.zeros(rows, columns) for _i in range(i - 3)])
            for _j in range(i - 4))))
        return M_i


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def V(n):
    """Elimination theory vector of variables."""
    zs = punctures(n)[1:n - 3]
    elimination_vector = flatten(sympy.Matrix([1, zs[0]]))
    for i in range(1, len(zs)):
        elimination_vector = flatten(sympy.tensorproduct([zs[i] ** j for j in range(i + 2)], elimination_vector))
    return elimination_vector


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def numerical_coeffs(Mn, n, dict_ss):
    """Numerical coefficients of polynomial of order (n - 3)! obtianed from determinant of elimination theory matrix."""
    Mn = Mn.tolist()
    Mn = [[zs_sub(ss_sub(str(entry))) for entry in line] for line in Mn]
    zs = punctures(n)

    values = [mpmath.e ** (2 * mpmath.pi * 1j * j / (math.factorial(n - 3) + 1)) for j in range(math.factorial(n - 3) + 1)]

    A = [[value ** exponent for exponent in range(math.factorial(n - 3) + 1)[::-1]] for value in values]
    b = []

    for i, value in enumerate(values):
        dict_zs = {str(zs[-2]): value, str(zs[-3]): 1}  # noqa --- used in eval function
        nMn = mpmath.matrix([[eval(entry, None) for entry in line] for line in Mn])
        b += [mpmath.det(nMn)]

    return mpmath.lu_solve(A, b).T.tolist()[0]


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def solve_scattering_equations(n, dict_ss):
    if n == 3:
        return [{}]

    Mn = M(n)
    zs = punctures(n)

    num_coeffs = numerical_coeffs(Mn, n, dict_ss)
    roots = mpmath.polyroots(num_coeffs, maxsteps=10000, extraprec=300)
    sols = [{str(zs[-2]): root * zs[-3]} for root in roots]

    if n == 4:
        sols = [{str(zs[-2]): mpmath.mpc(sympy.simplify(sols[0][str(zs[-2])].subs({zs[1]: 1})))}]
    else:
        Mnew = copy.deepcopy(Mn)
        Mnew[:, 0] += Mnew[:, 1] * zs[1]
        Mnew.col_del(1)
        Mnew.row_del(-1)

        # subs
        sol = sols[0]
        Mnew = Mnew.tolist()
        Mnew = [[zs_sub(ss_sub(str(entry))).replace("dict_zs['z{}']".format(n - 1),
                                                    "dict_zs['z{}'] * mpmath.mpc(sol[str(zs[-2])] / zs[-3])".format(n - 2))
                 for entry in line] for line in Mnew]

        # get scaling
        dict_zs = {str(zs[-3]): 10 ** -10, str(zs[1]): 1}
        nMn = mpmath.matrix([[eval(entry, None) for entry in line] for line in Mnew])
        a = mpmath.det(nMn)
        dict_zs = {str(zs[-3]): 10 ** -11, str(zs[1]): 1}
        nMn = mpmath.matrix([[eval(entry, None) for entry in line] for line in Mnew])
        b = mpmath.det(nMn)
        assert(abs(round(mpmath.log(abs(b) / abs(a)) / mpmath.log(10)) - mpmath.log(abs(b) / abs(a)) / mpmath.log(10)) < 10 ** - 5)
        scaling = - round(mpmath.log(abs(b) / abs(a)) / mpmath.log(10))

        # solve the linear equations
        for i in range(1, n - 3):
            Mnew = copy.deepcopy(Mn)
            index = V(n).index(zs[i])
            Mnew[:, 0] += Mnew[:, index] * zs[i]
            Mnew.col_del(index)
            Mnew.row_del(-1)
            Mnew = Mnew.tolist()
            if i == 1:
                Mnew = [[zs_sub(ss_sub(str(entry))).replace("dict_zs['z{}']".format(n - 1),
                                                            "dict_zs['z{}'] * mpmath.mpc(sol[str(zs[-2])] / zs[-3])".format(n - 2))
                         for entry in line] for line in Mnew]
                for sol in sols:
                    A = [[value ** exponent for exponent in [1, 0]] for value in [-1, 1]]
                    b = []
                    for value in [-1, 1]:
                        dict_zs = {str(zs[-3]): value, str(zs[1]): 1}
                        nMn = mpmath.matrix([[eval(entry, None) for entry in line] for line in Mnew])
                        b += [mpmath.det(nMn) / (value ** scaling)]
                    coeffs = mpmath.lu_solve(A, b).T.tolist()[0]
                    sol[str(zs[-3])] = - coeffs[1] / coeffs[0]
                    sol[str(zs[-2])] = mpmath.mpc((sympy.simplify(sol[str(zs[-2])].subs({zs[-3]: sol[str(zs[-3])]}))))
            else:
                Mnew = [[zs_sub(ss_sub(str(entry))) for entry in line] for line in Mnew]

                for sol in sols:
                    A = [[value ** exponent for exponent in [1, 0]] for value in [-1, 1]]
                    b = []
                    for value in [-1, 1]:
                        dict_zs = {str(zs[i]): value, str(zs[-3]): sol[str(zs[-3])], str(zs[-2]): sol[str(zs[-2])]}  # noqa --- used in eval function
                        nMn = mpmath.matrix([[eval(entry, None) for entry in line] for line in Mnew])
                        b += [mpmath.det(nMn)]
                    coeffs = mpmath.lu_solve(A, b).T.tolist()[0]
                    sol[str(zs[i])] = - coeffs[1] / coeffs[0]

    return sols


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


pattern_ss = re.compile(r"s_(\d*)")
ss_sub = functools.partial(pattern_ss.sub, r"dict_ss['s_\1']")
pattern_zs = re.compile(r"z(\d*)")
zs_sub = functools.partial(pattern_zs.sub, r"dict_zs['z\1']")


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
