"""
    polytrop.py

    Interface for a system of general-relativistic hydrodynamic equations 
    describing a polytropic star.
"""
import numpy as np
from scipy.optimize import brentq, newton

from hydro_base.variables import ConservedVector1D, PrimitiveVector1D, \
    FluxVector1D, SourceVector1D, CharacteristicVector1D

from hydro_base.eos import IsentropicEOS

from mesh.patch import RadialGrid

from gr.metric import Metric, RGPSMetric
from gr.hydro.equations import GREquations1D


class Polytrope(GREquations1D):
    """System of equations for a polytropic star"""

    def __init__(self, eos: IsentropicEOS, grid: RadialGrid, atm_rho=1e-10,
                 atm_eps=1e-10, source_func=None):
        """Initialize the system of equations

        Parameters
        ----------
        eos : EOS
            The equation of state used in these equations
        grid : RadialGrid
            The grid these equations apply to.
        atm_rho : float
            Atmosphere density
        source_func : function pointer
            An externally defined source function with the call signature:
            `source_func(U, V, g, grid, S)`
        """
        self.eos = eos
        self.grid = grid
        self.atm_rho = atm_rho
        self.atm_eps = atm_eps
        self.external_source_func = source_func

    def fluxes(self, U: ConservedVector1D, V: PrimitiveVector1D,
               g: Metric, F: FluxVector1D):
        """Calculate the fluxes

        Parameters
        ----------
        U : ConservedVector1D
            Conserved variables
        V : PrimitiveVector1D
            Primitive variables
        g : Metric
            Metric containing lapse, shift, spatial metric, and extrinsic
            curvature
        F : FluxVector1D
            [out] Flux result vector
        """
        # Extract and calculate the required quantities
        D, S, tau = U
        rho0, v, eps = V

        P = self.eos.pressure(rho0)

        # Fill the flux vector
        F.density[:] = D*v
        F.momentum[:] = S*v + P
        F.energy[:] = (tau - P)*v

    def sources(self, U: ConservedVector1D, V: PrimitiveVector1D,
                g: RGPSMetric, source: SourceVector1D):
        """Calculate the source terms

        Parameters
        ----------
        U : ConservedVector1D
            Conserved variables
        V : PrimitiveVector1D
            Primitive variables
        g : Metric
            Metric containing lapse, shift, spatial metric, and extrinsic
            curvature
        S : SourceVector1D
            [out] Source result vector
        """
        # Extract and calculate the required quantities
        D, S, tau = U
        rho0, v, eps = V

        P = self.eos.pressure(rho0)

        # Sources are on cell centers so these values are fine
        X = g.X
        m = g.m
        alpha = g.alpha

        r = self.grid.x
        alpha_X = alpha*X
        m_r2 = m/(r*r)

        # The density and energy sources are zero in the RGPS formulation
        source.density[:] = 0.0
        source.energy[:] = 0.0

        # Build up momentum flux in parts
        source.momentum[:] = (S*v - tau - D)*alpha_X*(8.0*np.pi*r*P + m_r2)
        source.momentum[:] += alpha_X*P*m_r2
        source.momentum[:] += 2.0*alpha*P/(X*r)

        # Update any external sources
        if self.external_source_func is not None:
            self.external_source_func(U, V, g, self.grid, source)

    def speeds(self, U: ConservedVector1D, V: PrimitiveVector1D,
               g: Metric, chars: CharacteristicVector1D):
        """Calculate the characteristic speeds

        Parameters
        ----------
        U : ConservedVector1D
            Conserved variables
        V : PrimitiveVector1D
            Primitive variables
        g : Metric
            Metric containing lapse, shift, spatial metric, and extrinsic
            curvature
        chars : CharacteristicVector1D
            [out] Characteristic speeds result vector
        """
        rho0, v, eps = V
        a = self.eos.sound_speed(rho0, eps)

        chars.minus = (v - a)/(1.0 - v*a)
        chars.center = v
        chars.plus = (v + a)/(1.0 + v*a)

    def prim2con(self, V: PrimitiveVector1D, g: Metric, U: ConservedVector1D):
        """Convert the primitive variables to conserved quantities

        Parameters
        ----------
        V : PrimitiveVector1D
            Primitive variables
        g : Metric
            Metric containing lapse, shift, spatial metric, and extrinsic
            curvature
        U : ConservedVector1D
            [out] Conserved variables
        """
        # Extract and calculate the required quantities
        rho0, v, eps = V

        W = 1.0/np.sqrt(1.0 - v*v)
        P = self.eos.pressure(rho0)

        # Enthalpy
        h = 1.0 + eps + P/rho0

        # Used more than once below
        rho_W = rho0*W
        rho_h_W2 = rho_W*W*h
        # We might be doing this at the cell faces and/or on a smaller range than the full grid so calculate this in place
        X = rho_h_W2 - P

        # Fill the conserved vector
        U.density = X*rho_W
        U.momentum = rho_h_W2*v
        U.energy = rho_h_W2 - P - U.density

    def con2prim(self, U: ConservedVector1D, g: Metric, V: PrimitiveVector1D):
        """Convert the conserved variables to primitives

        Parameters
        ----------
        U : ConservedVector1D
            Conserved variables
        g : Metric
            Metric containing lapse, shift, spatial metric, and extrinsic
            curvature
        V : PrimitiveVector1D
            [out] Primitive variables
        """
        def guess(P, D, S, tau):
            """Guess the conserved variables given `P`"""
            try:
                X = tau + D
                v = S/(tau + D + P)
                W = 1.0/np.sqrt(1.0 - v*v)
                rho0 = D/(X*W)
                eps = (tau + D + P*(1.0 - W*W))/(rho0*W*W) - 1.0
            except ValueError:
                print((P, D, S, tau))

            return rho0, v, eps

        def root_func(P, D, S, tau):
            """Function for root-finder"""
            rho0, v, eps = guess(P, D, S, tau)

            return self.eos.pressure(rho0) - P

        # def from_rho(rho, D, S, tau):
        #     P = self.eos.pressure(rho)
        #     eps = self.eos.energy(rho)
        #     v = S/(tau + D + P)

        #     return v, eps

        # def func(rho, D, S, tau):
        #     v, eps = from_rho(rho, D, S, tau)
        #     W = 1.0/np.sqrt(1.0 - v*v)
        #     X = tau + D
        #     return rho0*X*W - D

        # def dfunc(rho, D, S, tau):
        #     pass

        D, S, tau = U

        N = D.shape[0]

        X = D + tau

        # No nice way to vectorize this - do one cell at a time
        for i in range(N):
            if D[i] == 0.0:
                V.density[i] = 0.0
                V.velocity[i] = 0.0
                V.specific_energy[i] = 0.0
            else:
                # Are we in the atmosphere?
                if V.density[i] <= self.atm_rho:
                    V.density[i] = self.atm_rho
                    U.density[i] = self.atm_rho
                    U.momentum[i] = 0.0
                    U.energy[i] = V.density[i]*V.specific_energy[i]

                if U.momentum[i] == 0.0:
                    V.velocity[i] = 0.0
                    V.density[i] = D[i]/X[i]
                    V.specific_energy[i] = \
                        (tau[i] + D[i] - V.density[i])/V.density[i]

                    if V.density[i] < self.atm_eps:
                        V.density[i] = self.atm_eps
                else:
                    P_guess = self.eos.pressure(V.density[i])

                    P = newton(root_func, P_guess, args=(D[i], S[i], tau[i]),
                               maxiter=1000, tol=1e-6)
                    rho0, v, eps = guess(P, D[i], S[i], tau[i])

                    V.density[i] = rho0
                    V.velocity[i] = v
                    V.specific_energy[i] = eps
