"""
    tensor.py

    This module implements a basic spatial three-vector and rank-two tensor.
    These types are derived from `numpy.ndarray` for complete access to the
    vectorized linear algebra operations in `scipy`.

        * Can be created as a single vector/tensor or as a grid of vectors/
        tensors.  This structure was chosen to facilitate vectorized
        computations at each grid cell.

        * Named properties for components are added to the `ndarray`
        sub-classes.

    Usage
    -----
    * Calculate the tensor-vector contraction `u_j = v_i T^ij`:
        ```
            v = ThreeVector(x,y,z)
            T = Tensor.Symmetric(xx, xy, xz, yy, yz, zz)

            u = T.vector_contract(v)
        ```
    * Calculate the tensor-vector contraction `u_i = T^ij v_j`:
        ```
            v = ThreeVector(x,y,z)
            T = Tensor.Symmetric(xx, xy, xz, yy, yz, zz)

            u = T.vector_contract(v, slot=1)
        ```

    * Calculate the tensor-vector-vector contraction `S = T^ij u_i v_j`:
        ```
            u = ThreeVector(x,y,z)
            v = ThreeVector(x,y,z)
            T = Tensor.Symmetric(xx, xy, xz, yy, yz, zz)

            S = T.contract_with_vectors(u, v)
        ```

    * Calculate the tensor-tensor contraction `A^ik = T^i_j L^jk`:
        ```
            T = Tensor.Symmetric(xx, xy, xz, yy, yz, zz)
            L = Tensor.Symmetric(xx, xy, xz, yy, yz, zz)

            A = T.tensor_contract(L, slots=(1,0))
        ```

    * Calculate full tensor-tensor contraction `S = T^ij L_ij`:
        ```
            T = Tensor.Symmetric(xx, xy, xz, yy, yz, zz)
            L = Tensor.Symmetric(xx, xy, xz, yy, yz, zz)

            S = T.tensor_full_contract(L)
        ```
"""

import numpy as np


class ThreeVector(np.ndarray):
    """Spatial 3-vector"""

    def __new__(cls, x, y, z):
        """Create a new ThreeVector.

        Note
        ----
        `__new__` is required for deriving from `ndarray` so that casting and views work properly.

        Parameters
        ----------
        x : Scalar
            First component, e.g. x, r, etc.
        y : Scalar
            Second component, e.g. y, theta, etc.
        z : Scalar
            Third component, e.g. z, phi, etc.

        Returns
        -------
        ThreeVector
            Instantiated ThreeVector object
        """
        # Generally components will be stored in contiguous arrays, i.e. all
        # `x` in one array, all `y` in a second array, etc.  This reshapes the
        # vector to a grid of vectors (if necessary), allowing for use of
        # `np.linalg` routines to be applied to a grid as a whole.
        vec = np.stack([x, y, z], axis=-1).view(cls)

        return vec

    def __array_finalize__(self, obj):
        """Required for deriving from `ndarray`."""
        if obj is None:
            return

    ###################################################
    #   Named properties for quick component access   #
    ###################################################
    @property
    def x(self):
        return self[..., 0]

    @x.setter
    def x(self, value):
        self[..., 0] = value

    @property
    def y(self):
        return self[..., 1]

    @y.setter
    def y(self, value):
        self[..., 1] = value

    @property
    def z(self):
        return self[..., 2]

    @z.setter
    def z(self, value):
        self[..., 2] = value


class Tensor(np.ndarray):
    """Rank-two spatial tensor"""

    # For easier selection of `einsum` parameter strings
    _vector_contract_slots = np.array(["...ij,...i", "...ij,...j"])
    _tensor_contract_slots = \
        np.array([["...ij,...ik->...jk", "...ij,...ki->...jk"],
                  ["...ij,...jk->...ik", "...ik,...jk->...ij"]])

    def __new__(cls, xx, xy, xz, yx, yy, yz, zx, zy, zz):
        """Create a new Tensor

        Parameters
        ----------
        xx,xy,xz,yx,yy,yz,zx,zy,zz : Scalar
            Components of the tensor

        Returns
        -------
        Tensor
            Instantiated Tensor object.
        """
        # Reshape to a grid of tensors (if needed) to facilitate `np.linalg`
        T = np.stack([xx, xy, xz, yx, yy, yz, zx, zy, zz], axis=-1)
        T = np.reshape(T, (*np.asarray(xx).shape, 3, 3)).view(cls)

        return T

    def __array_finalize__(self, obj):
        """Required for deriving from `ndarray`."""
        if obj is None:
            return

    @classmethod
    def Symmetric(cls, xx, xy, xz, yy, yz, zz):
        """Create a new symmetric Tensor

        Parameters
        ----------
        xx,xy,xz,yy,yz,zz : Scalar
            Components of the tensor

        Returns
        -------
        Tensor
            Instantiated Tensor object.
        """
        return cls(xx, xy, xz, xy, yy, yz, xz, yz, zz)

    def vector_contract(self, A, slot=0):
        """Contract a vector with the tensor.

        Parameters
        ----------
        A : ThreeVector
            Vector to contract with tensor.
        slot : int, optional
            Which tensor slot to contract (the default is 0, which the first
            slot.)

        Returns
        -------
        ThreeVector
            3-vector resulting from contraction.
        """
        return np.einsum(self._vector_contract_slots[slot], self, A,
                         optimize='greedy')

    def contract_with_vectors(self, A, B):
        """Contract two vectors with the tensor.

        Parameters
        ----------
        A,B : ThreeVector
            Vectors to contract with tensor.

        Returns
        -------
        Scalar
            Scalar resulting from contraction.
        """
        return np.einsum("...ij,...i,...j", self, A, B, optimize='greedy')

    def tensor_contract(self, T, slots=[0, 0]):
        """Contract a tensor with this tensor.

        Parameters
        ----------
        T : Tensor
            Tensor to contract with this tensor.
        slots : (int,int), optional
            Which tensor slots to contract (the default is (0,0), which the 
            first slot of each tensor.)

        Returns
        -------
        Tensor
            Tensor resulting from contraction.
        """
        return np.einsum(self._tensor_contract_slots[slots], self, T,
                         optimize='greedy')

    def tensor_full_contract(self, T):
        """Fully contract a tensor with this tensor.

        Parameters
        ----------
        T : Tensor
            Tensor to contract with this tensor.

        Returns
        -------
        Scalar
            Scalar resulting from contraction.
        """
        return np.einsum("...ij,...ij", self, T, optimize='greedy')

    ###################################################
    #   Named properties for quick component access   #
    ###################################################

    @property
    def xx(self):
        return self[..., 0, 0]

    @xx.setter
    def xx(self, value):
        self[..., 0, 0] = value

    @property
    def xy(self):
        return self[..., 0, 1]

    @xy.setter
    def xy(self, value):
        self[..., 0, 1] = value

    @property
    def xz(self):
        return self[..., 0, 2]

    @xz.setter
    def xz(self, value):
        self[..., 0, 2] = value

    @property
    def yx(self):
        return self[..., 1, 0]

    @yx.setter
    def yx(self, value):
        self[..., 1, 0] = value

    @property
    def yy(self):
        return self[..., 1, 1]

    @yy.setter
    def yy(self, value):
        self[..., 1, 1] = value

    @property
    def yz(self):
        return self[..., 1, 2]

    @yz.setter
    def yz(self, value):
        self[..., 1, 2] = value

    @property
    def zx(self):
        return self[..., 2, 0]

    @zx.setter
    def zx(self, value):
        self[..., 2, 0] = value

    @property
    def zy(self):
        return self[..., 2, 1]

    @zy.setter
    def zy(self, value):
        self[..., 2, 1] = value

    @property
    def zz(self):
        return self[..., 2, 2]

    @zz.setter
    def zz(self, value):
        self[..., 2, 2] = value
