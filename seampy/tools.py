#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Giuseppe


import shelve
import numpy


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def flatten(temp_list, recursion_level=0, treat_list_subclasses_as_list=True, treat_tuples_as_lists=False, max_recursion=None):
    from sympy.matrices.dense import MutableDenseMatrix
    from numpy import ndarray
    flat_list = []
    for entry in temp_list:
        if type(entry) == list and (max_recursion is None or recursion_level < max_recursion):
            flat_list += flatten(entry, recursion_level=recursion_level + 1, treat_list_subclasses_as_list=treat_list_subclasses_as_list,
                                 treat_tuples_as_lists=treat_tuples_as_lists, max_recursion=max_recursion)
        elif ((issubclass(type(entry), list) or type(entry) in [MutableDenseMatrix, ndarray]) and
              treat_list_subclasses_as_list is True and (max_recursion is None or recursion_level < max_recursion)):
            flat_list += flatten(entry, recursion_level=recursion_level + 1, treat_list_subclasses_as_list=treat_list_subclasses_as_list,
                                 treat_tuples_as_lists=treat_tuples_as_lists, max_recursion=max_recursion)
        elif (type(entry) == tuple and treat_tuples_as_lists is True and (max_recursion is None or recursion_level < max_recursion)):
            flat_list += flatten(entry, recursion_level=recursion_level + 1, treat_list_subclasses_as_list=treat_list_subclasses_as_list,
                                 treat_tuples_as_lists=treat_tuples_as_lists, max_recursion=max_recursion)
        else:
            flat_list += [entry]
    return flat_list


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


class MyShelf(object):     # context manager shelf

    def __init__(self, filename, mode):
        self.filename = filename
        self.mode = mode

    def __enter__(self):
        self.obj = shelve.open(self.filename, self.mode)
        return self.obj

    def __exit__(self, exc_type, exc_value, traceback):
        self.obj.close()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def pfaffian(matrix):
    rows, cols = matrix.shape
    if rows == 0 and cols == 0:
        return 1
    else:
        return sum((-1) ** i * matrix[0, i - 1] *
                   pfaffian(matrix[numpy.ix_([j for j in range(cols) if j != 0 and j != i - 1],
                                             [j for j in range(cols) if j != 0 and j != i - 1])]) for i in range(2, cols + 1))
