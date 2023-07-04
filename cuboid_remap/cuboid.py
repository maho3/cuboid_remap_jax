"""
cuboid class and related objects

This code is a simple rewrite of Jordan Carlson and Martin White's code.

I have removed the dependence on the vec3 class in the original implementation
in favor of off-the-shelf numpy vector objects and functions.
"""

from __future__ import print_function, division
from math import floor, ceil, fmod
import numpy as np
from cuboid_remap.utils import triple_scalar_product
import jax
import jax.numpy as jnp


__all__ = ['Cuboid', 'Plane', 'Cell', 'unitCubeTest']
__author__ = ['Duncan Campbell', 'see doc string note']


class Cuboid:
    """
    cuboid remapping class.
    """

    def __init__(self, u1=[1, 0, 0], u2=[0, 1, 0], u3=[0, 0, 1]):
        """
        Parameters
        ----------
        u1 : array_like
            lattice vector, integer array of length 3

        u2 : array_like
            lattice vector, integer array of length 3

        u3 : array_like
            lattice vector, integer array of length 3

        Notes
        -----
        u1,u2,u3 form an unimodular invertable 3x3 integer matrix.
        See the functions the remap.py to select lattive vectors
        """
        u1 = np.atleast_1d(u1).astype('float64')
        u2 = np.atleast_1d(u2).astype('float64')
        u3 = np.atleast_1d(u3).astype('float64')

        if triple_scalar_product(u1, u2, u3) != 1:
            msg = ("Invalid lattice vectors: u1 = %s, u2 = %s, u3 = %s" %
                   (u1, u2, u3))
            raise ValueError(msg)
        else:
            s1 = np.dot(u1, u1)
            s2 = np.dot(u2, u2)
            d12 = np.dot(u1, u2)
            d23 = np.dot(u2, u3)
            d13 = np.dot(u1, u3)
            alpha = -d12/s1
            gamma = -(alpha*d13 + d23)/(alpha*d12 + s2)
            beta = -(d13 + gamma*d12)/s1
            self.e1 = u1
            self.e2 = u2 + alpha*u1
            self.e3 = u3 + beta*u1 + gamma*u2

        self.L1 = np.linalg.norm(self.e1)
        self.L2 = np.linalg.norm(self.e2)
        self.L3 = np.linalg.norm(self.e3)
        self.n1 = self.e1/self.L1
        self.n2 = self.e2/self.L2
        self.n3 = self.e3/self.L3
        self.cells = []

        v0 = np.array([0.0, 0.0, 0.0])
        self.v = [v0,
                  v0 + self.e3,
                  v0 + self.e2,
                  v0 + self.e2 + self.e3,
                  v0 + self.e1,
                  v0 + self.e1 + self.e3,
                  v0 + self.e1 + self.e2,
                  v0 + self.e1 + self.e2 + self.e3]

        # Compute bounding box of cuboid
        xs = [vk[0] for vk in self.v]
        ys = [vk[1] for vk in self.v]
        zs = [vk[2] for vk in self.v]
        vmin = np.array([min(xs), min(ys), min(zs)])
        vmax = np.array([max(xs), max(ys), max(zs)])

        # Extend to nearest integer coordinates
        ixmin = int(floor(vmin[0]))
        ixmax = int(ceil(vmax[0]))
        iymin = int(floor(vmin[1]))
        iymax = int(ceil(vmax[1]))
        izmin = int(floor(vmin[2]))
        izmax = int(ceil(vmax[2]))

        # Determine which cells (and which faces within those cells) are non-trivial
        for ix in range(ixmin, ixmax):
            for iy in range(iymin, iymax):
                for iz in range(izmin, izmax):
                    shift = np.array([-1.0*ix, -1.0*iy, -1.0*iz])
                    faces = [Plane(self.v[0] + shift, +1.0*self.n1),
                             Plane(self.v[4] + shift, -1.0*self.n1),
                             Plane(self.v[0] + shift, +1.0*self.n2),
                             Plane(self.v[2] + shift, -1.0*self.n2),
                             Plane(self.v[0] + shift, +1.0*self.n3),
                             Plane(self.v[1] + shift, -1.0*self.n3)]

                    c = Cell(ix, iy, iz)
                    skipcell = False
                    for f in faces:
                        r = unitCubeTest(f)
                        if r == +1:
                            # unit cube is completely above this plane--this cell is empty
                            continue
                        elif r == 0:
                            # unit cube intersects this plane--keep track of it
                            c.faces.append(f)
                        elif r == -1:
                            skipcell = True
                            break

                    if skipcell or len(c.faces) == 0:
                        continue
                    else:
                        self.cells.append(c)

        # for the identity remapping, use exactly one cell
        if len(self.cells) == 0:
            self.cells.append(Cell())

        # store all unique faces of data
        unique_faces = {}
        for i, c in enumerate(self.cells):
            for j, f in enumerate(c.faces):
                n = np.array((f.a, f.b, f.c, f.d))
                if tuple(n) in unique_faces:
                    unique_faces[tuple(n)].append((i, 1))
                elif tuple(-1*n) in unique_faces:
                    unique_faces[tuple(-1*n)].append((i, -1))
                else:
                    unique_faces[tuple(n)] = [(i, 1)]

        Nface = len(unique_faces)
        Ncell = len(self.cells)

        self.normals = np.zeros(shape=(Nface, 3))
        self.d = np.zeros(shape=(Nface, 1))
        self.select = np.zeros(shape=(Ncell, Nface))

        for i, (key, value) in enumerate(unique_faces.items()):
            self.normals[i] = key[:3]
            self.d[i] = key[-1]
            for c, f in value:
                self.select[c, i] = f

        self.cell_cens = np.array([[c.ix, c.iy, c.iz] for c in self.cells])
        self.nmat = np.stack([self.n1, self.n2, self.n3])

    def Transform(self, x, y, z):
        """
        Transform coordinates
        """
        data = jnp.stack([x, y, z])[:, None]
        bools = ~jnp.all(
            (self.select * (self.normals@data+self.d).T) > 0, axis=1)
        offset = jnp.sum(jnp.where(bools[:, None], 0, self.cell_cens), axis=0)
        return jnp.dot(self.nmat, (data[:, 0] + offset))

    def InverseTransform(self, r1, r2, r3):
        """
        Inverse transform coordinates
        """
        p = r1*self.n1 + r2*self.n2 + r3*self.n3
        x1 = fmod(p[0], 1) + (p[0] < 0)
        x2 = fmod(p[1], 1) + (p[1] < 0)
        x3 = fmod(p[2], 1) + (p[2] < 0)
        return np.array([x1, x2, x3])


class Plane:
    """
    class to represent a plane in 3-D cartesian space
    """

    def __init__(self, p, n):
        """
        Parameters
        ----------
        p : array_like
            a point in a plane

        n : array_like
            a vector normal to the plane
        """
        self.normal = n
        self.a = n[0]
        self.b = n[1]
        self.c = n[2]
        self.d = -1.0*np.dot(p, n)

    def above(self, x, y, z):
        """
        Compare a point to a plane.

        Parameters
        ----------
        x, y, z : float
             coordinates of a point

        Returns
        -------
        above : float
            value is positive, negative, or zero depending on whether
            the point lies above, below, or on the plane.
        """
        return self.a*x + self.b*y + self.c*z + self.d


class Cell:
    """
    class to represent a cell
    """

    def __init__(self, ix=0, iy=0, iz=0):
        """
        """
        self.ix = ix
        self.iy = iy
        self.iz = iz
        # collection of planes that define the faces of the cell
        self.faces = []

    def contains(self, x, y, z):
        """
        determine if the cell contains a point
        """

        out = jnp.array([jnp.less(-f.above(x, y, z), 0) for f in self.faces])

        return jnp.all(out)


def unitCubeTest(P):
    """
    Detemrine if a unit cube is above, below, or intersecting a plane

    Parameters
    ----------
    P : plane object

    Returns
    -------
    u : int
        [+1, 0, -1] if the unit cube is above, below, or intersecting the plane.
    """

    above = 0
    below = 0

    corners = [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1),
               (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]
    for (a, b, c) in corners:
        s = P.above(a, b, c)
        if s > 0:
            above = 1
        elif s < 0:
            below = 1
    return above - below
