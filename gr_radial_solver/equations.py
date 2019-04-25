"""
    equations.py

    Interface for a system of general-relativistic hydrodynamic equations
    describing a polytropic star.
"""
import numpy as np
from scipy.optimize import brentq, newton

from hydro_base.variables import ConservedVector1D, PrimitiveVector1D, \
    FluxVector1D, SourceVector1D, CharacteristicVector1D

from hydro_base.eos import IsentropicEOS

from mesh.patch import RadialGrid
from gr_radial_solver.custom_grid import CustomGrid1D

from gr.metric import Metric, RGPSMetric


class Polytrope(object):
    """System of equations for a polytropic star"""

    def __init__(self, eos: IsentropicEOS, grid: CustomGrid1D, atm_rho=1e-10,
                atm_eps=1e-10, source_func=None):
        """Initialize the system of equations

        Parameters
        ----------
        eos : EOS
            The equation of state used in these equations
        grid : CustomGrid
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
        # Extract and calculate the required quantities
        D, S, tau = U
        rho0, v, eps = V

        P = self.eos.pressure(rho0)

        # Fill the flux vector
        F.density[:] = D*v
        F.momentum[:] = S*v + P
        # F.energy[:] = (tau + P)*v
        F.energy[:] = S - D*v

    def sources(self, U: ConservedVector1D, V: PrimitiveVector1D,
                alpha, X, m, source: SourceVector1D):
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
        ilo = self.grid.ilo
        ihi = self.grid.ihi

        D, S, tau = U[:, ilo:ihi+1]
        rho0, v, eps = V[:, ilo:ihi+1]

        P = self.eos.pressure(rho0)

        r = self.grid.x[ilo:ihi+1]
        alpha_X = alpha[ilo:ihi+1]*X[ilo:ihi+1]
        m_r2 = m[ilo:ihi+1]/(r*r)

        # The density and energy sources are zero in the RGPS formulation
        source.density[:] = 0.0
        source.energy[:] = 0.0

        # Build up momentum flux in parts
        source.momentum[ilo:ihi+1] = 0.0
        source.momentum[ilo:ihi+1] += (S*v - tau - D)*alpha_X*(8.0*np.pi*r*P + m_r2)
        source.momentum[ilo:ihi+1] += alpha_X*P*m_r2
        source.momentum[ilo:ihi+1] += 2.0*alpha[ilo:ihi+1]*P/(X[ilo:ihi+1]*r)

        # Update any external sources
        if self.external_source_func is not None:
            self.external_source_func(U, V, alpha, X, m, self.grid, source)

    def speeds(self, V: PrimitiveVector1D, chars: CharacteristicVector1D,
               abs=False):
        """Calculate the characteristic speeds

        Parameters
        ----------
        V : PrimitiveVector1D
            Primitive variables
        chars : CharacteristicVector1D
            [out] Characteristic speeds result vector
        """
        rho0, v, eps = V
        a = self.eos.sound_speed(rho0, eps)

        if abs:
            v = np.abs(v)

        chars.minus = (v - a)/(1.0 - v*a)
        chars.center = v
        chars.plus = (v + a)/(1.0 + v*a)

    def apply_atmosphere(self, V: PrimitiveVector1D, X):
        rho0, v, eps = V

        mask = rho0 < self.atm_rho
        rho0[mask] = self.atm_rho
        v[mask] = 0.0
        eps[mask] = self.eos.energy(rho0)

    def prim2con(self, V: PrimitiveVector1D, X, U: ConservedVector1D):
        """Convert the primitive variables to conserved quantities

        Parameters
        ----------
        V : PrimitiveVector1D
            Primitive variables
        U : ConservedVector1D
            [out] Conserved variables
        """
        # Extract and calculate the required quantities
        rho0, v, eps = V
        D, S, tau = U

        # Pressure & Enthalpy
        P = self.eos.pressure(rho0)
        h = self.eos.enthalpy(rho0)

        # Lorentz factor
        W = 1.0/np.sqrt(1.0 - v*v)

        # Fill the conserved vector
        D[:] = X*W*rho0
        S[:] = rho0*h*W*W*v
        tau[:] = rho0*h*W*W - P - D

    def con2prim(self, U: ConservedVector1D, X, rho_old, V: PrimitiveVector1D):
        """Convert the conserved variables to primitives

        Parameters
        ----------
        U : ConservedVector1D
            Conserved variables
        V : PrimitiveVector1D
            [out] Primitive variables
        """
        D, S, tau = U
        rho0, v, eps = V

        # rho0 = rho_old

        def guess(P, i):
            """Guess the conserved variables given `P`"""
            try:
                vv = S[i]/(tau[i] + D[i] + P)
                xx = tau[i] + D[i]
                W = 1.0/np.sqrt(1.0 - vv*vv)
                rrho0 = D[i]/(X[i]*W)
                eeps = (tau[i] + D[i] + P*(1.0 - W*W))/(rrho0*W*W) - 1.0
                # eeps = self.eos.energy(rrho0)

                if rho0[i] == self.atm_rho:
                    # rho0[i] = self.atm_rho
                    D[i] = rho0[i]
                    S[i] = 0.0
                    tau[i] = rho0[i]*eps[i]

                # if rho_old[i] == self.atm_rho:
                #     # rho0[i] = self.atm_rho
                #     D[i] = rho0[i]
                #     S[i] = 0.0
                #     tau[i] = rho0[i]*eps[i]

            except ValueError:
                print((P, D[i], S[i], tau[i]))

            return rrho0, vv, eeps

        def root_func(P, i):
            """Function for root-finder"""
            rrho0, vv, eeps = guess(P, i)

            return self.eos.pressure(rrho0) - P

        def droot_func(P, i):
            rrho0, vv, eeps = guess(P, i)
            T = (tau[i] + D[i] + P)**2 - S[i]**2
            dP_drho = self.eos.gamma*P/rrho0
            drho_dP = D[i]*S[i]**2/(np.sqrt(T)*(tau[i]+D[i]+P)**2)
            de_dP = P*S[i]**2/(rrho0*(D[i] + tau[i] + P)*T)
            dP_de = 0.0  # self.eos.gamma*rrho0

            return dP_drho*drho_dP + dP_de*de_dP - 1.0

        N = D.shape[0]

        # No nice way to vectorize this - do one cell at a time
        for i in range(N):
            # if D[i] == 0.0:
            #     rho0[i] = 0.0
            #     v[i] = 0.0
            #     eps[i] = 0.0
            if D[i] == self.atm_rho:
                rho0[i] = self.atm_rho
                v[i] = 0.0
                eps[i] = (tau[i] + D[i] - D[i]/X[i])/rho0[i]
                S[i] = 0.0
                tau[i] = rho0[i]*eps[i]
            else:
                # Are we in the atmosphere?
                if rho0[i] == self.atm_rho:
                    # rho0[i] = self.atm_rho
                    D[i] = rho0[i]
                    S[i] = 0.0
                    tau[i] = rho0[i]*eps[i]

                # if rho0[i] <= self.atm_rho:
                #     v[i] = 0.0
                #     S[i] = 0.0
                # if rho_old[i] == self.atm_rho:
                #     # rho0[i] = self.atm_rho
                #     D[i] = rho0[i]
                #     S[i] = 0.0
                #     tau[i] = rho0[i]*eps[i]

                if S[i] == 0.0:
                    v[i] = 0.0
                    rho0[i] = D[i]/X[i]
                    eps[i] = (tau[i] + D[i] - D[i]/X[i])/rho0[i]

                    if eps[i] < self.atm_eps:
                        eps[i] = self.atm_eps
                else:
                    P_guess = self.eos.pressure(rho_old[i])

                    P = newton(root_func, P_guess,  fprime=droot_func,
                       args=(i,), maxiter=1000, tol=1e-10)
                    rho0[i], v[i], eps[i] = guess(P, i)
