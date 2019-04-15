import numpy as np
import abc

from mesh import patch


def minmod(alpha, beta):
    sign = np.sign(alpha)
    return sign*np.maximum(0.0, np.minimum(np.abs(alpha), beta*sign))


class Reconstruct1D(abc.ABC):

    @abc.abstractmethod
    def interface_states(self, grid: patch.Grid1d, U, U_l, U_r):
        pass


class MinmodReconstruct1D(Reconstruct1D):

    def interface_states(self, grid: patch.Grid1d, U, U_l, U_r):
        U_l[:, 0] = U_r[:, 0] = U[:, 0]
        U_l[:, -1] = U_r[:, -1] = U[:, -1]

        dx = grid.dx

        s = np.diff(U)/dx

        slope = minmod(s[:, :-1], s[:, 1:])

        U_l[:, 1:-1] = U[:, 1:-1] - 0.5*slope*dx
        U_r[:, 1:-1] = U[:, 1:-1] + 0.5*slope*dx
