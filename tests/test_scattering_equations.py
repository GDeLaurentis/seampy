#

import math

from seampy import V, M, solve_scattering_equations


def test_size_M():
    n = 6
    assert((M(n).shape) == (math.factorial(n - 3), math.factorial(n - 3)))
