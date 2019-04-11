# unit tests for the metric

from gr.metric import Metric, MinkowskiMetric, SchwarzschildMetric
from gr.tensor import ThreeVector, Tensor
from mesh.patch import Grid1d

import numpy as np

from numpy.testing import assert_array_equal


class TestMetric(object):
    @classmethod
    def setup_class(cls):
        """This is run once for each class before any tests"""
        pass

    @classmethod
    def teardown_class(cls):
        """This is run once for each class after any tests"""
        pass

    def setup_method(self):
        """This is run before each test"""
        self.M = 1.0
        self.R = 2.0*self.M
        self.grid = Grid1d(10, xmin=1.5*self.R, xmax=2.0*self.R)

    def teardown_method(self):
        """This is run after each test"""
        self.M = None
        self.R = None
        self.grid = None

    def test_create_minkowski(self):
        """Test creating a Minkowski metric"""
        metric = MinkowskiMetric()

        eta = np.eye(3).view(Tensor)

        assert np.allclose(metric.g, eta)
        assert np.allclose(metric.inv_g, eta)
        assert np.allclose(metric.det_g, 1.0)

        assert np.allclose(metric.K, 0.0)
        assert np.allclose(metric.beta, 0.0)
        assert np.allclose(metric.alpha, 1.0)

    def test_raise_lower_vector(self):
        """Test creating and using a Schwarzschild metric to raise/lower 
        vector indices."""
        r = self.grid.x

        metric = SchwarzschildMetric(self.M, r, 0.5*np.pi)

        a2 = 1.0-2.0*self.M/r

        # x = 0.0*r + 1.0
        x = 1.0
        y = 2.0*x
        z = 3.0*x

        A = ThreeVector(x, y, z)

        Au = metric.raise_vector(A)
        Al = metric.lower_vector(Au)

        actual_Au = ThreeVector(x*a2, y/(r**2), z/(r**2))

        assert np.allclose(Au, actual_Au)
        assert np.allclose(Al, A)

    def test_raise_lower_tensor(self):
        """Test creating and using a Schwarzschild metric to raise/lower 
        tensor indices."""

        r = self.grid.x

        metric = SchwarzschildMetric(self.M, r, 0.5*np.pi)

        a2 = 1.0-2.0*self.M/r

        # x = np.ones(self.grid.qx)
        x = 1.0
        y = 2.0*x
        z = 3.0*x
        zs = np.zeros(self.grid.qx)

        T = Tensor.Symmetric(x, 0.0, 0.0, y, 0.0, z)

        Tuu = metric.raise_tensor_all(T)
        Tll = metric.lower_tensor_all(Tuu)

        actual_Tuu = Tensor.Symmetric(x*a2*a2, zs, zs, y/(r**4), zs, z/(r**4))

        assert np.allclose(Tuu, actual_Tuu)
        assert np.allclose(Tll, T)
