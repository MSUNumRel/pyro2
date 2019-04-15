import numpy as np
from scipy.optimize import minimize_scalar
from scipy.integrate import solve_ivp, solve_bvp
from scipy.interpolate import CubicSpline

from hydro_base.eos import IsentropicEOS
from gr.init_data import InitialData1D
from gr.init_data.init_data import _default_var_names
from mesh import patch


class TOVInitialData(InitialData1D):

    def initialize(self, eos: IsentropicEOS):
        self.eos = eos

        self.r = self.grid.x
        self.m = np.zeros_like(self.r)
        self.phi = np.zeros_like(self.r)
        self.P = np.zeros_like(self.r)

        self.rho0 = np.zeros_like(self.r)
        self.P = np.zeros_like(self.r)
        self.eps = np.zeros_like(self.r)

        self.u = np.zeros_like(self.r)

        self.vars = {
            "density": self.rho0,
            "velocity": self.u,
            "specific energy": self.eps,
            "mass": self.m,
            "potential": self.phi,
        }

    def center_state(self, rho0_c):
        # Use first grid cell as central radius
        r_c = self.grid.x[self.grid.ng]

        # Setup central conditions
        _, eps_c = self.eos.from_density(rho0_c)

        V_c = 4.0/3.0*np.pi*r_c**3
        m_c = V_c*rho0_c*(1.0 + eps_c)

        phi_c = 0.0

        return [phi_c, r_c, m_c]

    def rhs_pressure(self, lP, U):
        P = np.exp(lP)

        rho0, eps = self.eos.from_pressure(P)
        _, r, m = U

        A = 4.0*np.pi*r**2

        E = rho0*(1.0 + eps)

        dphi_dP = -1.0/(E + P)
        dr_dP = r*(r - 2.0*m)/(m + A*P*r)*dphi_dP
        dm_dP = A*E*dr_dP

        return [dphi_dP*P, dr_dP*P, dm_dP*P]

    def initial_guess(self, rho0_c, rho0_end, tol=1e-8):
        P_start, _ = self.eos.from_density(rho0_c)
        lP_start = np.log(P_start)

        P_end, _ = self.eos.from_density(rho0_end)
        lP_end = np.log(P_end)

        U0 = self.center_state(rho0_c)

        sol = solve_ivp(self.rhs_pressure, [lP_start, lP_end], U0,
                        atol=tol, rtol=tol)

        phi, r, m = sol.y
        P = np.exp(sol.t)
        rho0, eps = self.eos.from_pressure(P)

        # Match to Schwarzschild solution at outer radius
        R = r[-1]
        M = m[-1]
        phi_R = 0.5*np.log(1.0 - 2.0*M/R)

        offset = phi_R - phi[-1]
        phi += offset

        return [phi, r, m, P, rho0, eps]

    def radius_derivs(self, r, U):
        phi, m, P = U

        rho0, eps = self.eos.from_pressure(P)

        E = rho0*(1.0 + eps)

        A = 4.0*np.pi*r**2

        dm_dr = A*E
        dP_dr = -(E + P)*(m + A*P*r)/(r*(r - 2.0*m))
        dphi_dr = -dP_dr/(E + P)

        return [dphi_dr, dm_dr, dP_dr]

    def solve_to_grid(self, rho0_c, rho0_end, tol=1e-8, max_iter=10000):

        # Generate an initial guess
        U_i = self.initial_guess(rho0_c, rho0_end, tol)
        phi_i, r_i, m_i, P_i, rho0_i, eps_i = U_i
        U0 = [phi_i[0], m_i[0], P_i[0]]
        print(U0)
        # Make sure that the outer radius is contained within the grid with at
        # least one ghost zone past the outer radius
        R = r_i[-1]
        assert R < self.r[-1]

        # Find the first grid cell past the outer radius
        ridx = np.argmax(self.r > R)
        # Generate radial grid
        ng = self.grid.ng
        r = self.r[ng:ridx]

        sol = solve_ivp(self.radius_derivs, [r[0], r[-1]], U0,
                        atol=tol, rtol=tol, t_eval=r)

        phi, m, P = sol.y
        rho0, eps = self.eos.from_pressure(P)

        # Match to Schwarzschild solution at outer radius
        R = r[-1]
        M = m[-1]
        phi_R = 0.5*np.log(1.0 - 2.0*M/R)

        offset = phi_R - phi[-1]
        phi += offset

        # Store the solution
        self.phi[ng:ridx] = phi
        self.m[ng:ridx] = m
        self.P[ng:ridx] = P
        self.rho0[ng:ridx] = rho0
        self.eps[ng:ridx] = eps

        # Fill in enclosed mass for outer radii
        M = m[-1]
        self.m[ridx:] = M

        # Fill in Schwarzschild solution for outer radius
        self.phi[ridx:] = 0.5*np.log(1.0 - 2.0*M/self.r[ridx:])

    def fill_patch(self, cc_patch: patch.CellCenterData1d,
                   var_names=_default_var_names):

        # Make sure the grids match
        assert np.allclose(self.grid.x, cc_patch.grid.x)

        for key, var in self.vars.items():
            if key in var_names.keys():
                cc_patch.get_var(var_names[key])[:] = var[:]
