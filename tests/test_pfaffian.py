import numpy
import seampy


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def test_pfaffian_one():
    assert seampy.tools.pfaffian(numpy.array([[0, 1], [-1, 0]])) == 1


def test_pfaffian_two():
    assert seampy.tools.pfaffian(numpy.array([[0, 1, 2, 3], [-1, 0, 4, 5], [-2, -4, 0, 6], [-3, -5, -6, 0]])) == 1 * 6 - 2 * 5 + 4 * 3
