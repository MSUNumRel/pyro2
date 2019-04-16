"""
    riemann.py

    Interface for Riemann solvers and two sample solvers:

        * HLLE: Two-wave approximate Riemann solver
        * Rusanov: More diffusive version of HLLE with the assumption that the 
        characteristics lambda_+ == lambda_-
"""
import numpy as np
import abc

from hydro_base.variables import ConservedVector1D, PrimitiveVector1D, \
    FluxVector1D, CharacteristicVector1D

from hydro_base.equations import Equations1D


class RiemannSolver1D(abc.ABC):
    """Base class for Riemann solvers"""

    @abc.abstractmethod
    def fluxes(self, U_l: ConservedVector1D, U_r: ConservedVector1D,
               V_l: PrimitiveVector1D, V_r: PrimitiveVector1D,
               F_l: FluxVector1D, F_r: FluxVector1D,
               char_l: CharacteristicVector1D, char_r: CharacteristicVector1D,
               F: FluxVector1D):
        """Compute the fluxes from solving the Riemann problem at the cell 
        interfaces.

        Parameters
        ----------
        U_l : ConservedVector1D
            Conserved variables on the left side of interface
        U_r : ConservedVector1D
            Conserved variables on the right side of interface
        V_l : PrimitiveVector1D
            Primitive variables on the left side of interface
        V_r : PrimitiveVector1D
            Primitive variables on the right side of interface
        F_l : FluxVector1D
            Fluxes calculated from left side values
        F_r : FluxVector1D
            Fluxes calculated from right side values
        char_l : CharacteristicVector1D
            Characteristics from left side values
        char_r : CharacteristicVector1D
            Characteristics from right side values
        F : FluxVector1D
            [out] Fluxes from solving the local Riemann problems
        """
        pass


class HLLE1D(RiemannSolver1D):
    """Two-wave approximate Riemann solver."""

    def __init__(self, equations: Equations1D):
        """Initialize the HLLE Riemann solver

        Parameters
        ----------
        equations : Equations1D
            The hydrodynamic equations used by this solver
        """
        self.eqns = equations

    def fluxes(self, U_l: ConservedVector1D, U_r: ConservedVector1D,
               V_l: PrimitiveVector1D, V_r: PrimitiveVector1D,
               F_l: FluxVector1D, F_r: FluxVector1D,
               char_l: CharacteristicVector1D, char_r: CharacteristicVector1D,
               F: FluxVector1D):
        """Compute the fluxes from solving the Riemann problem at the cell 
        interfaces.

        Parameters
        ----------
        U_l : ConservedVector1D
            Conserved variables on the left side of interface
        U_r : ConservedVector1D
            Conserved variables on the right side of interface
        V_l : PrimitiveVector1D
            Primitive variables on the left side of interface
        V_r : PrimitiveVector1D
            Primitive variables on the right side of interface
        F_l : FluxVector1D
            Fluxes calculated from left side values
        F_r : FluxVector1D
            Fluxes calculated from right side values
        char_l : CharacteristicVector1D
            Characteristics from left side values
        char_r : CharacteristicVector1D
            Characteristics from right side values
        F : FluxVector1D
            [out] Fluxes from solving the local Riemann problems
        """
        s_l = np.minimum(0.0, np.minimum(char_l.minus, char_r.minus))
        s_r = np.maximum(0.0, np.minimum(char_l.plus, char_r.plus))

        F[:] = (s_r*F_l - s_l*F_r + s_r*s_l*(U_r - U_l))/(s_r - s_l)


class Rusanov1D(RiemannSolver1D):
    """More diffusive version of the HLLE solver."""

    def __init__(self, equations: Equations1D):
        """Initialize the Rusanov Riemann solver

        Parameters
        ----------
        equations : Equations1D
            The hydrodynamic equations used by this solver
        """
        self.eqns = equations

    def fluxes(self, U_l: ConservedVector1D, U_r: ConservedVector1D,
               V_l: PrimitiveVector1D, V_r: PrimitiveVector1D,
               F_l: FluxVector1D, F_r: FluxVector1D,
               char_l: CharacteristicVector1D, char_r: CharacteristicVector1D,
               F: FluxVector1D):
        """Compute the fluxes from solving the Riemann problem at the cell 
        interfaces.

        Parameters
        ----------
        U_l : ConservedVector1D
            Conserved variables on the left side of interface
        U_r : ConservedVector1D
            Conserved variables on the right side of interface
        V_l : PrimitiveVector1D
            Primitive variables on the left side of interface
        V_r : PrimitiveVector1D
            Primitive variables on the right side of interface
        F_l : FluxVector1D
            Fluxes calculated from left side values
        F_r : FluxVector1D
            Fluxes calculated from right side values
        char_l : CharacteristicVector1D
            Characteristics from left side values
        char_r : CharacteristicVector1D
            Characteristics from right side values
        F : FluxVector1D
            [out] Fluxes from solving the local Riemann problems
        """
        s_l = np.minimum(0.0, np.minimum(char_l.minus, char_r.minus))
        s_r = np.maximum(0.0, np.minimum(char_l.plus, char_r.plus))

        s = np.maximum(np.abs(s_l), np.abs(s_r))

        F[:] = 0.5*(F_l + F_r) - 0.5*s*(U_r - U_l)
