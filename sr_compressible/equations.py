"""
    equations.py

    This script implements the 1D special-relativistic Euler equations of compressible hydrodynamics.  The formulation of these equations and code follows closely to:

        * Rezzolla & Zanotti "Relativistic Hydrodynamics" (2013)
        * David Radice, Lecture+Notes+Code
          JINA Neutron Star Merger Summer School (2018)
          https://github.com/dradice/JINA_MSU_School_2018/tree/master/Radice
"""
import numpy as np
import abc
from scipy.optimize import brentq

from hydro_base.variables import ConservedVector1D, PrimitiveVector1D, \
    FluxVector1D, SourceVector1D, CharacteristicVector1D

from hydro_base.equations import Equations1D

from hydro_base.eos import EOS


def lorentz_and_vel(u):
    """Calculate the Lorentz factor and spatial velocity

    Parameters
    ----------
    u : float(s)
        4-velocity

    Returns
    -------
    (float(s), float(s))
        The Lorentz factor and spatial velocity
    """
    W = np.sqrt(1.0 + u*u)
    v = u/W

    return W, v


class CompressibleSR1D(Equations1D):
    """Relativistic compressible Euler equations (1D)"""

    def __init__(self, eos: EOS, source_func=None):
        """Initialize the system of equations

        Parameters
        ----------
        eos : EOS
            The equation of state used in these equations
        source_func : function pointer
            An externally defined source function with same call signature as 
            `sources()` below
        """
        self.eos = eos
        self.external_source_func = source_func

    def fluxes(self, U: ConservedVector1D, V: PrimitiveVector1D,
               F: FluxVector1D):
        """Calculate the fluxes

        Parameters
        ----------
        U : ConservedVector1D
            Conserved variables
        V : PrimitiveVector1D
            Primitive variables
        F : FluxVector1D
            [out] Flux result vector
        """
        D, S, E = U
        rho0, u, eps = V

        W, v = lorentz_and_vel(u)

        P = self.eos.pressure(rho0, eps)

        F.density = D*v
        F.momentum = S*v + P
        F.energy = (E + P)*v

    def sources(self, U: ConservedVector1D, V: PrimitiveVector1D,
                S: SourceVector1D):
        """Calculate the source terms

        Parameters
        ----------
        U : ConservedVector1D
            Conserved variables
        V : PrimitiveVector1D
            Primitive variables
        S : SourceVector1D
            [out] Source result vector
        """
        S[:] = 0.0
        if self.external_source_func != None:
            self.external_source_func(U, V, S)

    def speeds(self, U: ConservedVector1D, V: PrimitiveVector1D,
               chars: CharacteristicVector1D):
        """Calculate the characteristic speeds

        Parameters
        ----------
        U : ConservedVector1D
            Conserved variables
        V : PrimitiveVector1D
            Primitive variables
        chars : CharacteristicVector1D
            [out] Characteristic speeds result vector
        """
        rho0, u, eps = V
        W, v = lorentz_and_vel(u)
        a = self.eos.sound_speed(rho0, eps)

        chars.minus = (v - a)/(1.0 - v*a)
        chars.center = v
        chars.plus = (v + a)/(1.0 + v*a)

    def prim2con(self, V: PrimitiveVector1D, U: ConservedVector1D):
        """Convert the primitive variables to conserved quantities

        Parameters
        ----------
        V : PrimitiveVector1D
            Primitive variables
        U : ConservedVector1D
            [out] Conserved variables
        """
        rho0, u, eps = V

        W, v = lorentz_and_vel(u)

        P = self.eos.pressure(rho0, eps)
        rho = self.eos.energy(rho0, eps)

        factor = (P + rho)*W*W
        U.density = rho0*W
        U.momentum = factor*v
        U.energy = factor - P

    def con2prim(self, U: ConservedVector1D, V: PrimitiveVector1D):
        """Convert the conserved variables to primitives

        Parameters
        ----------
        U : ConservedVector1D
            Conserved variables
        V : PrimitiveVector1D
            [out] Primitive variables
        """
        def guess(P, D, S, E):
            """Guess the conserved variables given `P`"""
            try:
                v = S/(E + P)
                W = 1.0/np.sqrt(1.0 - v*v)
                u = W*v
                rho0 = D/W
                eps = (E - D*W + P*(1.0 - W*W))/(D*W)
            except ValueError:
                print((P, D, S, E))

            return rho0, u, eps

        def root_func(P, D, S, E):
            """Function for root-finder"""
            rho0, u, eps = guess(P, D, S, E)

            return self.eos.pressure(rho0, eps) - P

        D, S, E = U

        N = D.shape[0]

        # No nice way to vectorize this - do one cell at a time
        for i in range(N):
            rho0_max = D[i]
            eps_max = E[i]/D[i]

            P_min = 1e-6
            P_max = self.eos.pressure(rho0_max, eps_max)

            P = brentq(root_func, P_min, P_max, args=(D[i], S[i], E[i]))
            rho0, u, eps = guess(P, D[i], S[i], E[i])

            V.density[i] = rho0
            V.velocity[i] = u
            V.specific_energy[i] = eps
