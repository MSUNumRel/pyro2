"""
    reconstruct.py

    This script contains the base class for recosntructing cell-face data.  A 
    sample implementation of a piecewise-linear reconstruction with a minmod 
    slope limiter is included.
"""
import numpy as np
import abc

from mesh import patch


def minmod(alpha, beta):
    """Simple minmod implementation

    Parameters
    ----------
    alpha : float(s)
        First value(s) to compare
    beta : float(s)
        Second values(s) to compare

    Returns
    -------
    float(s)
        Minmod result for two values
    """
    sign = np.sign(alpha)
    return sign*np.maximum(0.0, np.minimum(np.abs(alpha), beta*sign))


class Reconstruct1D(abc.ABC):
    """Base class for reconstruction methods."""

    @abc.abstractmethod
    def interface_states(self, grid: patch.Grid1d, U, U_l, U_r):
        """Calculate the values of values at the cell faces

        Parameters
        ----------
        grid : patch.Grid1d
            The grid this data lives on
        U : array of floats
            The values in the cells
        U_l : array of floats
            [out] The reconstructed left edge values
        U_r : array of floats
            [out] The reconstructed right edge values
        """
        pass

    @abc.abstractmethod
    def interface_state(self, grid: patch.Grid1d, U, U_l, U_r):
        """Calculate the values of values at the cell faces

        Parameters
        ----------
        grid : patch.Grid1d
            The grid this data lives on
        U : array of floats
            The values in the cells
        U_l : array of floats
            [out] The reconstructed left edge values
        U_r : array of floats
            [out] The reconstructed right edge values
        """
        pass


class MinmodReconstruct1D(Reconstruct1D):
    """Piecewise-linear minmod reconstruction."""

    def interface_states(self, grid: patch.Grid1d, U, U_l, U_r):
        """Calculate the values of values at the cell faces

        Parameters
        ----------
        grid : patch.Grid1d
            The grid this data lives on
        U : array of floats
            The values in the cells
        U_l : array of floats
            [out] The reconstructed left edge values
        U_r : array of floats
            [out] The reconstructed right edge values
        """
        # Set outer faces to outer cell values since there are no neighboring
        # states to compute the reconstruction from
        U_l[:, 0] = U_r[:, 0] = U[:, 0]
        U_l[:, -1] = U_r[:, -1] = U[:, -1]

        # Compute the base slopes
        dx = grid.dx
        s = np.diff(U)/dx

        # Compute TVD slopes with minmod
        slope = minmod(s[:, :-1], s[:, 1:])

        # Use TVD slopes to reconstruct interfaces
        U_l[:, 1:-1] = U[:, 1:-1] - 0.5*slope*dx
        U_r[:, 1:-1] = U[:, 1:-1] + 0.5*slope*dx

    def interface_state(self, grid: patch.Grid1d, U, U_l, U_r):
        """Calculate the values of values at the cell faces

        Parameters
        ----------
        grid : patch.Grid1d
            The grid this data lives on
        U : array of floats
            The values in the cells
        U_l : array of floats
            [out] The reconstructed left edge values
        U_r : array of floats
            [out] The reconstructed right edge values
        """
        # Set outer faces to outer cell values since there are no neighboring
        # states to compute the reconstruction from
        U_l[0] = U_r[0] = U[0]
        U_l[-1] = U_r[-1] = U[-1]

        # Compute the base slopes
        dx = grid.dx
        s = np.diff(U)/dx

        # Compute TVD slopes with minmod
        slope = minmod(s[:-1], s[1:])

        # Use TVD slopes to reconstruct interfaces
        U_l[1:-1] = U[1:-1] - 0.5*slope*dx
        U_r[1:-1] = U[1:-1] + 0.5*slope*dx
