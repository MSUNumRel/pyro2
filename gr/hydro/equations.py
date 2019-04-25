"""
    equations.py

    Interface for a system of general-relativistic hydrodynamic equations.
"""
import numpy as np
from scipy.optimize import brentq, newton
import abc

from hydro_base.variables import ConservedVector1D, PrimitiveVector1D, \
    FluxVector1D, SourceVector1D, CharacteristicVector1D

from hydro_base.eos import IsentropicEOS

from mesh.patch import Grid1d, RadialGrid

from gr.metric import Metric, RGPSMetric


class GREquations1D(abc.ABC):
    """Base class for a system of general-relativistic hydrodynamic
    equations"""

    @abc.abstractmethod
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
        pass

    @abc.abstractmethod
    def sources(self, U: ConservedVector1D, V: PrimitiveVector1D,
                g: Metric, S: SourceVector1D):
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
        pass

    @abc.abstractmethod
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
        pass

    @abc.abstractmethod
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
        pass

    @abc.abstractmethod
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
        pass


class RPGSEquations(GREquations1D):
    """Hydrodynamic equations from the Radial-Gauge, Polar Slicing
    (RPGS) metric"""

    def __init__(self, eos: IsentropicEOS, grid: Grid1d, source_func=None):
        """Initialize the system of equations

        Parameters
        ----------
        eos : EOS
            The equation of state used in these equations
        grid : Grid1d
            The grid these equations apply to.
        source_func : function pointer
            An externally defined source function with the call signature:
            `source_func(U, V, g, grid, S)`
        """
        self.eos = eos
        self.grid = grid
        self.external_source_func = source_func

    def lorentz_and_vel(self, u, X):
        """Calculate the Lorentz factor and spatial velocity

        Parameters
        ----------
        u : float(s)
            4-velocity radial component
        X : float(s)
            Metric g_rr component

        Returns
        -------
        tuple of floats
            (Lorentz factor, spatial velocity)
        """
        W = np.sqrt(1.0 + X*X*u*u)
        v = X*u/W

        return W, v

    def fluxes(self, U: ConservedVector1D, V: PrimitiveVector1D,
               g: RGPSMetric, F: FluxVector1D):
        """Calculate the fluxes

        Parameters
        ----------
        U : ConservedVector1D
            Conserved variables
        V : PrimitiveVector1D
            Primitive variables
        g : RGPSMetric
            Metric containing lapse, shift, spatial metric, and extrinsic
            curvature
        F : FluxVector1D
            [out] Flux result vector
        """
        # Extract and calculate the required quantities
        D, S, _ = U
        rho0, v, eps = V

        # _, v = self.lorentz_and_vel(u, g.X)
        W = 1.0/np.sqrt(1.0 - v*v)
        P = self.eos.pressure(rho0)

        # Used repeatedly below
        D_v = D*v

        # Fill the flux vector
        F.density[:] = D_v
        F.momentum[:] = S*v + P
        F.energy[:] = S - D_v

    def sources(self, U: ConservedVector1D, V: PrimitiveVector1D,
                g: RGPSMetric, source: SourceVector1D):
        """Calculate the source terms

        Parameters
        ----------
        U : ConservedVector1D
            Conserved variables
        V : PrimitiveVector1D
            Primitive variables
        g : RGPSMetric
            Metric containing lapse, shift, spatial metric, and extrinsic
            curvature
        S : SourceVector1D
            [out] Source result vector
        """
        # Extract and calculate the required quantities
        D, S, tau = U
        rho0, v, eps = V

        # _, v = self.lorentz_and_vel(u, g.X)

        P = self.eos.pressure(rho0)

        # Used more than once below
        r = self.grid.x
        X = D + tau
        alpha_X = g.alpha*X
        m_r2 = g.m/(r*r)

        # The density and energy sources are zero in the RGPS formulation
        source.density = 0.0
        source.energy = 0.0

        # Build up momentum flux in parts
        source.momentum = (S*v - tau - D)*alpha_X*(8.0*np.pi*r*P + m_r2)
        source.momentum += alpha_X*P*m_r2
        source.momentum += 2*g.alpha*P / (X*r)

        source.momentum[X == 0.0] = 0.0

        # Update any external sources
        if self.external_source_func is not None:
            self.external_source_func(U, V, g, self.grid, source)

    def speeds(self, U: ConservedVector1D, V: PrimitiveVector1D,
               g: RGPSMetric, chars: CharacteristicVector1D):
        """Calculate the characteristic speeds

        Parameters
        ----------
        U : ConservedVector1D
            Conserved variables
        V : PrimitiveVector1D
            Primitive variables
        g : RGPSMetric
            Metric containing lapse, shift, spatial metric, and extrinsic
            curvature
        chars : CharacteristicVector1D
            [out] Characteristic speeds result vector
        """
        rho0, v, eps = V
        # _, v = self.lorentz_and_vel(u, g.X)
        a = self.eos.sound_speed(rho0, eps)

        chars.minus = (v - a)/(1.0 - v*a)
        chars.center = v
        chars.plus = (v + a)/(1.0 + v*a)

    def prim2con(self, V: PrimitiveVector1D, g: RGPSMetric,
                 U: ConservedVector1D):
        """Convert the primitive variables to conserved quantities

        Parameters
        ----------
        V : PrimitiveVector1D
            Primitive variables
        g : RGPSMetric
            Metric containing lapse, shift, spatial metric, and extrinsic
            curvature
        U : ConservedVector1D
            [out] Conserved variables
        """
        # Extract and calculate the required quantities
        rho0, v, eps = V

        # W, v = self.lorentz_and_vel(u, g.X)
        W = 1.0/np.sqrt(1.0 - v*v)
        P = self.eos.pressure(rho0)

        # Enthalpy
        # h = np.ones_like(P) + eps
        # mask = rho0 != 0.0
        # h[mask] = 1.0 + eps[mask] + P[mask]/rho0[mask]
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

    def con2prim(self, U: ConservedVector1D, g: RGPSMetric,
                 V: PrimitiveVector1D):
        """Convert the conserved variables to primitives

        Parameters
        ----------
        U : ConservedVector1D
            Conserved variables
        g : RGPSMetric
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

        def from_rho(rho, D, S, tau):
            P = self.eos.pressure(rho)
            eps = self.eos.energy(rho)
            v = S/(tau + D + P)

            return v, eps

        def func(rho, D, S, tau):
            v, eps = from_rho(rho, D, S, tau)
            W = 1.0/np.sqrt(1.0 - v*v)
            X = tau + D
            return rho0*X*W - D

        def dfunc(rho, D, S, tau):
            pass

        D, S, tau = U

        N = D.shape[0]

        X = D + tau

        Didx = np.argmax(D < 1e-8)
        # Didx = N

        rho0_max = 2e-3
        rho0_min = 1e-8

        P_min = self.eos.pressure(rho0_min)
        P_max = self.eos.pressure(rho0_max)

        # No nice way to vectorize this - do one cell at a time
        for i in range(N):
            # rho0_max = D[i]/X[i]
            # eps_max = X[i]/D[i]
            P_max = self.eos.pressure(V.density[i])
            # print(i, P_max)

            # P = brentq(root_func, P_min, P_max, args=(D[i], S[i], tau[i]))
            P = newton(root_func, P_max, args=(D[i], S[i], tau[i]), maxiter=1000, tol=1e-6)
            rho0, v, eps = guess(P, D[i], S[i], tau[i])

            V.density[i] = rho0
            V.velocity[i] = v
            V.specific_energy[i] = eps

        # V[:, Didx:] = 0.0
