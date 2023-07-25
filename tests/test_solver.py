import math
import sympy
import pytest

import lips
import seampy


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def test_size_M():
    n = 6
    assert (seampy.M(n).shape) == (math.factorial(n - 3), math.factorial(n - 3))


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##


@pytest.mark.parametrize(
    "n", [3, 4, 5, 6, 7]
)
def test_scattering_equations_n_point(n):
    oParticles = lips.Particles(n)
    oParticles.fix_mom_cons()
    num_ss = {str(s): oParticles.compute(str(s)) for s in seampy.mandelstams(n)}
    sols = seampy.solve_scattering_equations(n, num_ss)
    assert all([entry < 10 ** -270 for entry in map(float, map(abs, sympy.simplify(seampy.hms(n).subs(sols[0]).subs(num_ss).subs({seampy.punctures(n)[1]: 1}))))])
