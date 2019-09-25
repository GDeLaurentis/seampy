#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Giuseppe


from __future__ import unicode_literals
from __future__ import print_function


import sympy
import operator


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def A(n):
    """A anti-symmetric matrix (reduced)."""
    ks = sympy.symbols('k1:{}'.format(n + 1))
    zs = sympy.symbols('z1:{}'.format(n + 1))

    A = sympy.zeros(n, n)
    for a in range(n):
        for b in range(a + 1, n):
            A[a, b] = 2 * ks[a] * ks[b] / (zs[a] - zs[b])
            A[b, a] = - A[a, b]

    # drop rows & cols 1 and 2 (i.e. 0 and 1)
    A = A[range(2, n), range(2, n)]

    # Mobius fix
    A = A.subs(((zs[0], sympy.oo), (zs[1], 1), (zs[n - 1], 0)))

    return A


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def Psi(n):
    """Psi anti-symmetric matrix (reduced)."""
    ks = sympy.symbols('k1:{}'.format(n + 1))
    es = sympy.symbols('e1:{}'.format(n + 1))
    zs = sympy.symbols('z1:{}'.format(n + 1))

    # A Matrix
    A = sympy.zeros(n, n)
    for a in range(n):
        for b in range(a + 1, n):
            A[a, b] = 2 * ks[a] * ks[b] / (zs[a] - zs[b])
            A[b, a] = - A[a, b]

    # B Matrix
    B = sympy.zeros(n, n)
    for a in range(n):
        for b in range(a + 1, n):
            B[a, b] = 2 * es[a] * es[b] / (zs[a] - zs[b])
            B[b, a] = - B[a, b]

    # C Matrix
    C = sympy.zeros(n, n)
    for a in range(n):
        for b in range(n):
            if a == b:
                C[a, b] = - sum([2 * es[a] * ks[j] / (zs[a] - zs[j]) for j in range(n) if j != a])
            else:
                C[a, b] = 2 * es[a] * ks[b] / (zs[a] - zs[b])

    # Psi - from block matrix
    Psi = sympy.Matrix(sympy.BlockMatrix(((A, -C.T), (C, B))))

    # drop rows & cols 1 and 2 (i.e. 0 and 1)
    Psi = Psi[range(2, 2 * n), range(2, 2 * n)]

    # simplify operations involving infinity
    for i in range(1, n):
        Psi = Psi.subs(((zs[0] - zs[i], zs[0]), (zs[i] - zs[0], - zs[0]), (zs[i] + zs[0], zs[0]), (- zs[i] - zs[0], - zs[0])))

    # pull out z1 ** -2  ---  Pfaffian goes aas root of this
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


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def Cyc(n):
    """Parke-Taylor-like cyclic factor (reduced)."""
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


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def Phi(n):
    """SE Jacobian matrix (reduced)."""
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


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def W1(n):
    """DF2 and CG integrand (reduced)."""
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
