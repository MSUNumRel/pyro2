"""
    metric.py

    Implmentation of a generic 3+1 metric and sample derived classes.  The
    base `Metric` class contains the spatial metric, lapse, shift vector, and
    extrinsic curvature.  These quantities are represented as rank-two tensors
    and three-vectors, whether as single vector/tensors or as a grid of
    vectors/tensors.  Additional functionality includes routines for lowering/
    raising indices of vectors and tensors, and contractions/scalar products.

    Three sample derived metrics are included:

        * Minkowski: Cartesian flat metric.  As this metric does not change in 
        space, it is not implemented as a grid function.

        * Spherical: Spherical coordinate metric; can be instantiated at one 
        grid location or over an entire grid.

        * Schwarzschild: Schwarzschild metric in Schwarzschild coordinates, 
        with appropriate lapse function and vanishing shift vector and 
        extrinsic curvature; can be instantiated at one grid location or over 
        an entire grid.
"""

import numpy as np

from gr.tensor import ThreeVector, Tensor

# For convenience
PI_OVER_TWO = 0.5*np.pi


class Metric(object):
    """Base 3+1 metric object.

    Attributes
    ----------
    g : Rank-2 Tensor
        The spatial metric (covariant components)
    inv_g : Rank-2 Tensor
        Inverse spatial metric (contravariant components)
    det_g : Scalar
        Determinant of the spatial metric

    alpha : Scalar
        The lapse function
    beta : 3-Vector
        The shift vector
    K : Rank-2 Tensor
        The extrinsic curvature
    """

    def __init__(self, g: Tensor, alpha, beta: ThreeVector, K: Tensor):
        """Instantiate the metric

        Parameters
        ----------
        g : Tensor
            Covariant spatial metric
        alpha : Scalar
            Lapse function
        beta : ThreeVector
            Shift vector
        K : Tensor
            Extrinsic curvature
        """
        # Save the inputs
        self.g = g
        self.alpha = alpha
        self.beta = beta
        self.K = K

        # Calculate these once
        self.inv_g = np.linalg.inv(g)
        self.det_g = np.linalg.det(g)

    def raise_vector(self, A):
        """Raise the index of a covariant vector.

        Parameters
        ----------
        A : ThreeVector
            Covariant 3-vector

        Returns
        -------
        ThreeVector
            3-Vector with raised components
        """
        return self.inv_g.vector_contract(A)

    def lower_vector(self, A):
        """Lower the index of a contravariant vector.

        Parameters
        ----------
        A : ThreeVector
            Contravariant 3-vector

        Returns
        -------
        ThreeVector
            3-Vector with lowered components
        """
        return self.g.vector_contract(A)

    def raise_tensor(self, T, slot=0):
        """Raise the specified component of a rank-two tensor

        Parameters
        ----------
        T : Tensor
            A rank-two tensor with covariant components to raise
        slot : int, optional
            The slot to raise (the default is 0, which corresponds to the first slot)

        Returns
        -------
        Tensor
            Rank-2 tensor with raised components in specified slot.
        """
        return self.inv_g.tensor_contract(T, slots=(0, slot))

    def lower_tensor(self, T, slot=0):
        """Lower the specified component of a rank-two tensor

        Parameters
        ----------
        T : Tensor
            A rank-two tensor with contravariant components to lower
        slot : int, optional
            The slot to lower (the default is 0, which corresponds to the first slot)

        Returns
        -------
        Tensor
            Rank-2 tensor with lowered components in specified slot.
        """
        return self.g.tensor_contract(T, slots=(0, slot))

    def raise_tensor_all(self, T):
        """Raise all components of a rank-two tensor

        Parameters
        ----------
        T : Tensor
            A rank-two tensor with covariant components to raise

        Returns
        -------
        Tensor
            Rank-2 tensor with raised components.
        """
        return np.einsum("...ik,...jl,...kl->...ij", self.inv_g, self.inv_g, T,
                         optimize='greedy')

    def lower_tensor_all(self, T):
        """Lower all components of a rank-two tensor

        Parameters
        ----------
        T : Tensor
            A rank-two tensor with contravariant components to lower

        Returns
        -------
        Tensor
            Rank-2 tensor with lowered components.
        """
        return np.einsum("...ik,...jl,...kl->...ij", self.g, self.g, T,
                         optimize='greedy')

    def scalar_product(self, A, B):
        """Scalar product of two 3-vectors.

        Parameters
        ----------
        A, B : ThreeVector
            Vectors to contract

        Returns
        -------
        Scalar
            The scalar prodcut of the two provided vectors.
        """
        return self.g.contract_with_vectors(A, B)


class MinkowskiMetric(Metric):
    """Minkowski flat metric in Cartesian coordinates"""

    def __init__(self):
        """Instantiate a non-grid function Minkowski metric."""
        # Fill spatial metric components
        g = np.eye(3, dtype=float).view(Tensor)

        # Vanishing shift and extrinsic curvature
        K = np.zeros_like(g)
        beta = ThreeVector(0.0, 0.0, 0.0)

        # Required time component of full metric: -alpha**2 dt**2
        alpha = 1.0

        super().__init__(g, alpha, beta, K)


class SphericalMetric(Metric):
    """Spherical coordinate metric"""

    def __init__(self, r, theta=PI_OVER_TWO):
        """Instantiate a spherical coordinate metric.

        Parameters
        ----------
        r : Scalar
            Radial coordinate
        theta : Scalar 
            Polar angle
        """
        # Even if these are scalars use as arrays
        r = np.asanyarray(r)
        theta = np.asanyarray(theta)

        # Fill spatial metric components
        g = np.zeros((*r.shape, 3, 3), dtype=float).view(Tensor)
        g.yy = r**2
        g.zz = r**2*np.sin(theta)**2

        # Vanishing shift and extrinsic curvature
        K = np.zeros_like(g)
        beta = ThreeVector(0.0, 0.0, 0.0)

        # Required time component of full metric: -alpha**2 dt**2
        alpha = 1.0

        super().__init__(g, alpha, beta, K)


class SchwarzschildMetric(Metric):

    def __init__(self, M, r, theta=PI_OVER_TWO):
        # Even if these are scalars use as arrays
        r = np.asanyarray(r)
        theta = np.asanyarray(theta)

        # Schwarzschild lapse
        alpha = np.sqrt(1.0 - 2.0*M/r)

        # Vanishing shift
        beta = np.zeros((*r.shape, 3), dtype=float).view(ThreeVector)

        # Mass associated with this metric
        self.M = M

        # Fill spatial metric components
        g = np.zeros((*r.shape, 3, 3), dtype=float).view(Tensor)
        g.xx = 1.0/(1.0 - 2.0*M/r)
        g.yy = r**2
        g.zz = r**2*np.sin(theta)**2

        # Vanishing extrinsic curvature
        K = np.zeros_like(g)

        super().__init__(g, alpha, beta, K)


class RGPSMetric(Metric):
    """Implementation of the Radial-Gauge, Polar Slicing metric found in
    O'Connor & Ott (2010)"""

    def __init__(self, m, Phi, r, theta=PI_OVER_TWO):
        # Even if these are scalars use as arrays
        m = np.asanyarray(m)
        Phi = np.asarray(Phi)
        r = np.asanyarray(r)
        theta = np.asanyarray(theta)

        # RGPS lapse
        alpha = np.exp(Phi)

        # RGPS X parameter
        self.X = 1.0/np.sqrt(1.0 - 2.0*m/r)

        # Mass associated with this metric
        self.m = m

        self.Phi = Phi

        # Vanishing shift
        beta = np.zeros((*r.shape, 3), dtype=float).view(ThreeVector)

        # Fill spatial metric components
        g = np.zeros((*r.shape, 3, 3), dtype=float).view(Tensor)
        g.xx = 1.0/(1.0 - 2.0*m/r)
        g.yy = r**2
        g.zz = r**2*np.sin(theta)**2

        # This is not strictly true; the extrinsic curvature in this case is
        # defined as K_ij = -(1/alpha)(dX/dt), but the curvature is never
        # explicity used in the RGPS formulation.
        K = np.zeros_like(g)

        super().__init__(g, alpha, beta, K)
