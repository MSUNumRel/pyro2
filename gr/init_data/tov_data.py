"""
    tov_data.py

    This script implements the initial data class for a TOV star.  Provided an 
    EOS and central density, this class will generate the initial primitive 
    and metric variables for a TOV star in hydrostatic equilibrium.  An 
    adapative step integration over the log space of the pressure will be 
    performed first to determine the outer radius of the star.  Once this 
    boundary is determined, a fixed step integration over the provided grid 
    will determine the initial state of the star.  This state can then be 
    loaded into a CellCenterData1d patch.
"""

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.integrate import solve_ivp, solve_bvp
from scipy.interpolate import CubicSpline

from hydro_base.eos import IsentropicEOS
from gr.init_data import InitialData1D
from gr.init_data.init_data import _default_var_names
from mesh import patch


class TOVInitialData(InitialData1D):
    """Initial data structure for a TOV star."""

    def initialize(self, eos: IsentropicEOS):
        """Initialize the data structure.

        Parameters
        ----------
        eos : IsentropicEOS
            A subclassed isentropic EOS, e.g. `PolytropicEOS`.
        """
        self.eos = eos

        # Allocate storage based on the grid size
        self.r = self.grid.x
        self.m = np.zeros_like(self.r)
        self.phi = np.zeros_like(self.r)
        self.P = np.zeros_like(self.r)

        self.rho0 = np.zeros_like(self.r)
        self.P = np.zeros_like(self.r)
        self.eps = np.zeros_like(self.r)

        # This will always be zero - just here for convenience
        self.v = np.zeros_like(self.r)

        # For mapping to a data patch
        self.vars = {
            "density": self.rho0,
            "velocity": self.v,
            "specific energy": self.eps,
            "mass": self.m,
            "potential": self.phi,
        }

    def center_state(self, rho0_c):
        """Calculate the central zone of the star.

        Parameters
        ----------
        rho0_c : float
            Density at the center of the star (relativistic units)

        Returns
        -------
        tuple of floats
            The central zone values of the potential, radius, and mass
        """

        # Use first grid cell as central radius
        # r_c = self.grid.x[self.grid.ng]
        r_c = self.grid.x[0]

        # Setup central conditions
        _, eps_c = self.eos.from_density(rho0_c)

        V_c = 4.0/3.0*np.pi*r_c**3
        m_c = V_c*rho0_c*(1.0 + eps_c)

        # Can be any arbitrary value - this will get offset later to match the
        # Schwarzschild solution at the maximum radius of the star.
        phi_c = 0.0

        return [phi_c, r_c, m_c]

    def rhs_pressure(self, lP, U):
        """Calculate the RHS of the TOV equations in regards to pressure
        derivatives.

        Parameters
        ----------
        lP : float
            Natural log of the pressure
        U : array of floats
            The current solution vector at `lP`

        Returns
        -------
        array of floats
            The evaluated RHS of the TOV equations.
        """
        # Calculate RHS in terms of P derivatives
        P = np.exp(lP)

        # Get needed values from EOS and solution vector
        rho0, eps = self.eos.from_pressure(P)
        _, r, m = U

        # Area of the shell this zone represents
        A = 4.0*np.pi*r**2

        # Mass-energy density
        E = rho0*(1.0 + eps)

        # RHS of TOV equations in regards to pressure
        dphi_dP = -1.0/(E + P)
        dr_dP = r*(r - 2.0*m)/(m + A*P*r)*dphi_dP
        dm_dP = A*E*dr_dP

        # Factor of `P` converts these back to derivatives in log space
        return [dphi_dP*P, dr_dP*P, dm_dP*P]

    def initial_guess(self, rho0_c, rho0_end, tol=1e-8):
        """Generate an initial guess at the TOV solution and determine the 
        location of the surface and its propertiers.

        Parameters
        ----------
        rho0_c : float
            Central density to begin intergration at.
        rho0_end : float
            A small density to determine when the outer radius of the star is 
            found.
        tol : float, optional
            Error tolerance for integration

        Returns
        -------
        grid of floats
            The solution from the adaptive IVP solver for the potential,
            radius, mass, pressure, density, and specific energy.
        """
        # Determine starting and ending pressure for the integration
        P_start, _ = self.eos.from_density(rho0_c)
        lP_start = np.log(P_start)

        P_end, _ = self.eos.from_density(rho0_end)
        lP_end = np.log(P_end)

        # Calculate the central zone state
        U0 = self.center_state(rho0_c)

        # Let scipy do the heavy lifting
        sol = solve_ivp(self.rhs_pressure, [lP_start, lP_end], U0,
                        atol=tol, rtol=tol)

        print("Made initial TOV guess")
        # Extract what we need from the integration result
        phi, r, m = sol.y
        P = np.exp(sol.t)
        rho0, eps = self.eos.from_pressure(P)

        print("Matching to Schwarzschild")
        # Match to Schwarzschild solution at outer radius
        R = r[-1]
        M = m[-1]
        phi_R = 0.5*np.log(1.0 - 2.0*M/R)

        offset = phi_R - phi[-1]
        phi += offset

        return [phi, r, m, P, rho0, eps]

    def radius_derivs(self, r, U):
        """RHS of the TOV equations in terms of radial derivatives.

        Parameters
        ----------
        r : float
            Radius to evaluate the RHS
        U : array of floats
            Previous solution vector

        Returns
        -------
        array of floats
            The RHS of the TOV equations
        """
        # Extract the values needed to evaluate the RHS
        phi, m, P = U
        rho0, eps = self.eos.from_pressure(P)

        # Mass-energy density
        E = rho0*(1.0 + eps)

        # Surface area of the zone
        A = 4.0*np.pi*r**2

        # Derivatives with respect to the radius
        dm_dr = A*E
        dP_dr = -(E + P)*(m + A*P*r)/(r*(r - 2.0*m))
        dphi_dr = -dP_dr/(E + P)

        return [dphi_dr, dm_dr, dP_dr]

    def solve_to_grid(self, rho0_c, rho0_end, tol=1e-8):
        """Determine the structure of a TOV star and fill the grid with the 
        results.

        Parameters
        ----------
        rho0_c : float
            The central density
        rho0_end : float
            Some small final density to determine the edge of the star
        tol : float, optional
            Error tolerance (the default is 1e-8)
        """
        # Generate an initial guess
        U_i = self.initial_guess(rho0_c, rho0_end, tol)
        phi_i, r_i, m_i, P_i, rho0_i, eps_i = U_i
        U0 = [phi_i[0], m_i[0], P_i[0]]

        # Make sure that the outer radius is contained within the grid with at
        # least one ghost zone past the outer radius
        R = r_i[-1]
        assert R < self.r[-1]

        # Find the first grid cell past the outer radius
        ridx = np.argmax(self.r > R)
        # Generate radial grid
        ng = self.grid.ng
        # r = self.r[ng:ridx]
        r = self.r[:ridx]

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
        self.phi[:ridx] = phi
        self.m[:ridx] = m
        self.P[:ridx] = P
        self.rho0[:ridx] = rho0
        self.eps[:ridx] = eps

        # Fill in enclosed mass for outer radii
        M = m[-1]
        self.m[ridx:] = M

        # Fill in Schwarzschild solution for outer radius
        self.phi[ridx:] = 0.5*np.log(1.0 - 2.0*M/self.r[ridx:])

    def fill_patch(self, cc_patch: patch.CellCenterData1d,
                   var_names=_default_var_names):
        """Fill a CellCenterData1d patch with the solution.

        Parameters
        ----------
        cc_patch : patch.CellCenterData1d
            The patch to fill
        var_names : dict, optional
            Which variables to fill.
        """
        # Make sure the grids match
        assert np.allclose(self.grid.x, cc_patch.grid.x)

        # Fill only the requested variables
        for key, var in self.vars.items():
            if key in var_names.keys():
                cc_patch.get_var(var_names[key])[:] = var[:]
