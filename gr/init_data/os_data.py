"""
    os_data.py

    This script implements the initial data class for an Oppenheimer-Snyder 
    star. Provided an initial stellar density and radius, this class will 
    generate the initial primitive and metric variables for an OS star. With 
    the initial radius given, a fixed step integration over the provided grid 
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


class OSInitialData(InitialData1D):
    """Initial data structure for OS star."""

    def initialize(self, eos: IsentropicEOS):
        """
        Initialize the data structure.
        """
        self.eos = eos

        # Allocate storage based on the grid size
        self.r = self.grid.x
        self.m = np.zeros_like(self.r)
        self.phi = np.zeros_like(self.r)
        self.P = np.zeros_like(self.r)

        self.rho0 = np.zeros_like(self.r)
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
            # "radMass_ratio": self.a,
        }

    def center_state(self, rho0):
        """Calculate the central zone of the star.

        Parameters
        ----------
        rho0 : float
            Density of the star (relativistic units)

        Returns
        -------
        tuple of floats
            The central zone values of the potential, radius, and mass
        """

        # Use first grid cell as central radius
        r_c = self.grid.x_i[self.grid.ilo+1]
        print("Center radius: {:.2e}".format(r_c))

        # Setup central conditions
        V_c = 4.0/3.0*np.pi*r_c**3
        m_c = V_c*rho0

        # Can be any arbitrary value - this will get offset later to match the
        # Schwarzschild solution at the maximum radius of the star.
        phi_c = 0.0

        return [phi_c, r_c, m_c]

    def radius_derivs(self, r, U, rho0):
        """RHS of the OS equations in terms of radial derivatives.

        Parameters
        ----------
        r : float
            Radius to evaluate the RHS
        U : array of floats
            Previous solution vector
        rho0 : float
            The initial density of the star

        Returns
        -------
        array of floats
            The RHS of the OS equations
        """
        # Extract the values needed to evaluate the RHS
        phi, r, m = U
        P = 0.0
        eps = 0.0

        # Surface area of the zone
        A = 4.0*np.pi*r**2

        # Derivatives with respect to the radius
        dm_dr = A*rho0
        dP_dr = 0.0
        dphi_dr = m/(r**2) - dm_dr/r

        return [dphi_dr, dm_dr, dP_dr]

    def solve_to_grid(self, rho0, r_end, tol=1e-8):
        """Determine the structure of a OS star and fill the grid with the 
        results.

        Parameters
        ----------
        rho0 : float
            The initial density of the star (homogenous)
        r_end : float
            The initial radius of the edge of the star
        tol : float, optional
            Error tolerance (the default is 1e-8)
        """

        # Generate a the central state
        U0 = self.center_state(rho0)

        # Make sure that the outer radius is contained within the grid with at
        # least one ghost zone past the outer radius
        R = r_end
        assert R < self.r[-1], "Max radius is {:.2e} is not in grid with (max r = {:.2e}".format(r_end, self.r[-1])

        # Find the first grid cell past the outer radius
        ridx = np.argmax(self.r > R)
        # Generate radial grid
        ng = self.grid.ng
        r = self.r[ng:ridx]
        r_pts = self.grid.x_i[ng+1:ridx+1]

        # sol = solve_ivp(self.radius_derivs, [r_pts[0], r_pts[-1]], U0,
        #                 atol=tol, rtol=tol, t_eval=r_pts)
        sol = solve_ivp(lambda r,U: self.radius_derivs(r, U, rho0), 
                        [r_pts[0], r_pts[-1]], U0,
                        atol=tol, rtol=tol, t_eval=r_pts)

        print("Solved to grid")
        phi, m, P = sol.y
        r_sol = sol.t
        eps = 0.0

        # Match to Schwarzschild solution at outer radius
        R = r_pts[-1]
        M = m[-1]
        phi_R = 0.5*np.log(1.0 - 2.0*M/R)

        offset = phi_R - phi[-1]
        phi += offset

        # Interpolators
        phi_func = CubicSpline(r_sol, phi, extrapolate=True)
        m_func = CubicSpline(r_sol, m, extrapolate=True)

        # Store the solution
        self.phi[ng:ridx] = phi_func(r_pts)
        self.m[ng:ridx] = m_func(r_pts)
        self.P[ng:ridx] = 0.0
        self.rho0[ng:ridx] = rho0
        self.eps[ng:ridx] = 0.0

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
