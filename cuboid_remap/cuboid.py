"""
Cuboid class and related objects

This code is a rewrite of Duncan Campbell's code, which itself was a rewrite
of Jordan Carlson and Martin White's code for cuboid remapping. Almost all of
the code structure remains exactly the same as Campbell's implementation.

The algorithmic details of most of the code remains exactly the same as the
original Carlson-White implementation. This includes the construction of the
Cuboid tesselation. The primary difference now is in the implementation of
Cuboid.Transform, which has been made significantly more efficient through
vectorization with Jax.
"""

from __future__ import print_function, division
from math import floor, ceil
import numpy as np
from jax import Array
import jax.numpy as jnp
from jax.typing import ArrayLike
from cuboid_remap.utils import triple_scalar_product


__all__ = ['Cuboid', 'Plane', 'Cell', 'unitCubeTest']


class Cuboid:
    def __init__(
            self,
            u1: ArrayLike = [1, 0, 0],
            u2: ArrayLike = [0, 1, 0],
            u3: ArrayLike = [0, 0, 1]
    ):
        """
        Cuboid remapping class.

        Args:
            u1: lattice vector, integer array of length 3
            u2: lattice vector, integer array of length 3
            u3: lattice vector, integer array of length 3

        Notes:
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

        # Determine which cells (and which faces) are non-trivial
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

                    c = Cell((ix, iy, iz))
                    skipcell = False
                    for f in faces:
                        r = unitCubeTest(f)
                        if r == +1:
                            # unit cube is completely above this plane
                            # this cell is empty
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

        # Store positions and orientations of all faces as matrices
        # These are used to speed up Transform computation

        # store all unique faces of data
        unique_faces = {}
        for i, c in enumerate(self.cells):
            for j, f in enumerate(c.faces):
                n = np.array((*f.n, f.d))
                if tuple(n) in unique_faces:
                    unique_faces[tuple(n)].append((i, 1))
                elif tuple(-n) in unique_faces:
                    unique_faces[tuple(-n)].append((i, -1))
                else:
                    unique_faces[tuple(n)] = [(i, 1)]

        Nface = len(unique_faces)
        Ncell = len(self.cells)

        # normals of all faces
        self.normals = np.zeros(shape=(Nface, 3))
        # offset of all faces
        self.d = np.zeros(shape=(Nface, 1))
        # selection matrix which specifies which side of faces are
        # in which cells
        self.select = np.zeros(shape=(Ncell, Nface))

        for i, (key, value) in enumerate(unique_faces.items()):
            self.normals[i] = key[:3]
            self.d[i] = key[-1]
            for c, f in value:
                self.select[c, i] = f

        # center positions of all cells
        self.cell_cens = jnp.array([c.pos for c in self.cells])
        # matrix of transformation vectors
        self.nmat = jnp.stack([self.n1, self.n2, self.n3])

    def Transform(self, x: ArrayLike) -> Array:
        """Transform coordinates to the cuboid domain

        Args:
            x (ArrayLike): Point or array of points to transform to the
                cuboid. Can be of shape (3,) or (N, 3), where N is the
                number of points.

        Returns:
            Array: Transformed points of equal shape to input x
        """
        x = x.astype(jnp.float64)
        x = jnp.atleast_2d(x).T
        bools = jnp.all(
            (self.select * (self.normals@x+self.d).T) >= 0, axis=1)
        offset = jnp.sum(jnp.where(bools[:, None], self.cell_cens, 0), axis=0)
        output = jnp.squeeze(jnp.dot(self.nmat, (x[:, 0] + offset)))
        output %= jnp.array([self.L1, self.L2, self.L3])
        return output

        # todo: the % is a hack to make sure that the transformed
        # coordinates are in the unit cube. This is necessary because
        # something weird happens when the coordinate falls exactly on
        # the boundary of the Plane. This should be fixed.

    def InverseTransform(self, r: ArrayLike) -> Array:
        """Transform coordinates from the cuboid domain to the unit cube

        Args:
            r (ArrayLike): Point or array of points to transform to the
                unit cube. Can be of shape (3,) or (N, 3), where N is the
                number of points.

        Returns:
            Array: Transformed points of equal shape to input r
        """
        r = jnp.atleast_2d(r).T
        p = jnp.dot(self.nmat, r)
        x = jnp.fmod(p, 1) + (p < 0)
        return jnp.squeeze(x)

    def TransformVelocity(self, v: ArrayLike) -> Array:
        """Transform velocity from the unit cube to the cuboid domain

        Args:
            v (ArrayLike): Velocity or array of velocities to transform to the
                cuboid. Can be of shape (3,) or (N, 3), where N is the
                number of points.

        Returns:
            Array: Transformed velocities of equal shape to input v
        """
        v = jnp.atleast_2d(v).T
        return jnp.squeeze(jnp.dot(self.nmat, v))


class Plane:
    def __init__(self, p: ArrayLike, n: ArrayLike):
        """Class to represent a plane in 3-D cartesian space

        Args:
            p (ArrayLike): A point on the plane
            n (ArrayLike): The normal vector to the plane
        """
        self.n = jnp.array(n)
        self.d = -1.0*jnp.dot(p, n)

    def above(self, x: ArrayLike) -> float:
        """Compare a point to a plane.

        Args:
            x (ArrayLike): coordinates of a point

        Returns:
            float: positive if point is above the plane, negative if below
        """
        x = jnp.atleast_1d(x)
        return jnp.dot(x, self.n) + self.d


class Cell:
    def __init__(self, pos: ArrayLike):
        """Class to represent a cell

        Args:
            pos (ArrayLike): position of the cell
        """
        self.pos = jnp.array(pos)
        # collection of planes that define the faces of the cell
        self.faces = []

    def contains(self, x) -> bool:
        """
        Determine if a point is inside the cell

        Args:
            x (ArrayLike): coordinates of a point

        Returns:
            bool: True if point is inside the cell, False otherwise
        """
        out = jnp.array([jnp.less(-f.above(x), 0) for f in self.faces])
        return jnp.all(out)


def unitCubeTest(P: Plane) -> int:
    """
    Determine if a unit cube is above, below, or intersecting a plane

    Args:
        P (Plane): Plane to test

    Returns:
        int: [+1, 0, -1] if the unit cube is above, below, or intersecting
            the plane.
    """

    above = 0
    below = 0

    corners = [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1),
               (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]
    for corner in corners:
        s = P.above(corner)
        if s > 0:
            above = 1
        elif s < 0:
            below = 1
    return above - below
