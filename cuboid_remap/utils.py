"""
utility functions for cuboid_remap
"""

from __future__ import print_function, division
import sys
import numpy as np
from jax.typing import ArrayLike


__all__ = ['triple_scalar_product', 'gcd', 'coprime_triples']


def triple_scalar_product(u: ArrayLike, v: ArrayLike, w: ArrayLike) -> float:
    """Triple scalar product of three vectors"""
    return u[0]*(v[1]*w[2] - v[2]*w[1]) +\
           u[1]*(v[2]*w[0] - v[0]*w[2]) +\
           u[2]*(v[0]*w[1] - v[1]*w[0])


def gcd(*args: int) -> int:
    """Return the greatest common integer divisor"""

    # return self if a single number is passed
    if len(args) == 1:
        return args[0]
    # pairwise case
    elif len(args) == 2:
        a = args[0]
        b = args[1]

        if (a < 0):
            a = -a

        if (b < 0):
            b = -b

        while (b != 0):
            tmp = b
            b = a % b
            a = tmp
        return a
    # if greater than two arguments, recurse
    else:
        a = args[0]
        b = gcd(*args[1:])
        return gcd(a, b)


def coprime_triples(
    max_int: int,
    min_int: int = 0,
    method: str = 'efficient'
) -> dict:
    """
    Return all integer coprime triples within a range

    Args:
        max_int (int): maximum integer in the range
        min_int (int, optional): minimum integer in the range.  Default is 0
        method (str, optional): method to use.  Default is 'efficient'

    Returns:
        d (dict): dictionary of coprime triples.  The keys of the dictionary
            are the sorted integers stored in a tuple. The associated values
            are the number of times this triple was encountered in the
            algorithm
    """

    d = {}

    if method == 'brute_force':
        # loop through all possible integer combinations
        for i in range(min_int, max_int+1):
            for j in range(min_int, max_int+1):
                for k in range(min_int, max_int+1):
                    if gcd(i, j, k) == 1:
                        x = min(i, j, k)  # smallest
                        z = max(i, j, k)  # largest
                        y = (i + j + k) - (x + z)  # middle
                        key = (x, y, z)
                        try:
                            d[key] += 1
                        except KeyError:
                            d[key] = 1
    elif method == 'efficient':
        # break loop when encountering a coprime double
        for i in range(min_int, max_int+1):
            for j in range(min_int, max_int+1):
                # if a pair is coprime, a triple must be coprime
                if gcd(i, j) == 1:
                    for k in range(min_int, max_int+1):
                        x = min(i, j, k)  # smallest
                        z = max(i, j, k)  # largest
                        y = (i + j + k) - (x + z)  # middle
                        key = (x, y, z)
                        try:
                            d[key] += 1
                        except KeyError:
                            d[key] = 1
                # if not, check to see if triple is coprime
                else:
                    for k in range(min_int, max_int+1):
                        if gcd(i, j, k) == 1:
                            x = min(i, j, k)  # smallest
                            z = max(i, j, k)  # largest
                            y = (i + j + k) - (x + z)  # middle
                            key = (x, y, z)
                            try:
                                d[key] += 1
                            except KeyError:
                                d[key] = 1

    else:
        msg = ('method not recognized.')
        raise ValueError(msg)

    return d
