"""
"""

from __future__ import print_function, division
import numpy as np
import sys
from copy import deepcopy
from itertools import combinations_with_replacement, permutations
from cuboid_remap.cuboid import Cuboid
from cuboid_remap.utils import triple_scalar_product, coprime_triples
import jax


__all__ = ['remap', 'remap_Lbox', 'generate_lattice_vectors']
__author__ = ['Duncan Campbell', ]


def remap(coords, Lbox, u1=[1, 0, 0], u2=[0, 1, 0], u3=[0, 0, 1]):
    """
    Re-map coordinates in a periodic box.

    Parameters
    ----------
    coords : array_like
        numpy of array of shape (3,npts)

    Lbox : float

    Returns
    -------
    new_coords : numpy.array
        remaped coordinates
    """

    coords = deepcopy(coords)
    coords = np.atleast_1d(coords)/Lbox

    assert np.shape(coords)[1] == (3)

    C = Cuboid(u1, u2, u3)

    # loop through each coordinate
    Npts = np.shape(coords)[0]
    transform = jax.vmap(C.Transform)
    coords = transform(coords[:, 0], coords[:, 1], coords[:, 2])
    # for i in range(Npts):
    #     coords[i, :] = C.Transform(coords[i, 0], coords[i, 1], coords[i, 2])

    return coords*Lbox


def remap_Lbox(u1=[1, 0, 0], u2=[0, 1, 0], u3=[0, 0, 1], Lbox=[1, 1, 1]):
    """
    Return re-mapped cuboid sides given a set of lattive vectors.

    Parameters
    ----------
    u1 : array_like

    u2 : array_like

    u3 : array_like

    Returns
    -------
    Lbox : numpy.array
        length 3 array containing sorted cuboid side lengths

    """

    u1 = np.atleast_1d(u1)
    u2 = np.atleast_1d(u2)
    u3 = np.atleast_1d(u3)

    assert np.shape(u1) == (3,)
    assert np.shape(u2) == (3,)
    assert np.shape(u3) == (3,)

    s1 = np.dot(u1, u1)
    s2 = np.dot(u2, u2)

    d12 = np.dot(u1, u2)
    d23 = np.dot(u2, u3)
    d13 = np.dot(u1, u3)

    alpha = -1.0*d12/s1
    gamma = -1.0*(alpha*d13 + d23)/(alpha*d12 + s2)
    beta = -1.0*(d13 + gamma*d12)/s1

    e1 = u1
    e2 = u2 + alpha*u1
    e3 = u3 + beta*u1 + gamma*u2

    L1 = np.sqrt(np.sum(e1**2))
    L2 = np.sqrt(np.sum(e2**2))
    L3 = np.sqrt(np.sum(e3**2))

    return L1, L2, L3


def generate_lattice_vectors(max_int=5):
    """
    Generate all invertible, integer-valued, 3x3 unimodular matrices.
    These are used as shear matrices in the cuboid remapping alogorithm.

    Parameters
    ----------
    max_int : int
        maximum value for a matrix element

    Returns
    -------
    d : dictionary
        dictionary of shear matrices.  The keys of the dictionary
        are the sorted cuboid side lengths. The associated values
        are invertible, integer-valued, 3x3 unimodular matrices.

    Notes
    -----
    Right now this uses a brute-force search method.  However, only
    matrices with co-prime triplet rows are conisdered.  I think
    this follows from the Quillen-Suslin Theorem.
    """

    # find all coprime triples in the range [-max_int, max_int]
    d = coprime_triples(max_int, -1*max_int)
    triplets = d.keys()

    # examine permuations of coprime triplets
    triplets = [[perm for perm in permutations(
        triplets[i])] for i in range(len(d.keys()))]

    # flatten list
    triplets = [item for sublist in triplets for item in sublist]
    num_triplets = len(triplets)

    # create dictionary to store shear matrices
    m = {}
    num_remappings = 0

    for i in range(num_triplets):
        u1 = np.array(triplets[i])

        for j in range(num_triplets):
            u2 = np.array(triplets[j])

            for k in range(num_triplets):
                u3 = np.array(triplets[k])

                # check to see if matrix is unimodular
                if triple_scalar_product(u1, u2, u3) == 1:

                    # calculate cuboid side lengths
                    L1, L2, L3 = np.sort(remap_Lbox(u1, u2, u3))[::-1]
                    # use these lengths as the dictionary key
                    key = (L1, L2, L3)

                    # add this shear matrix to the dictionary
                    try:
                        m[key].append([u1, u2, u3])
                    # add if this geometry has not been found
                    except KeyError:
                        num_remappings += 1
                        m[key] = [[u1, u2, u3]]

    return m
