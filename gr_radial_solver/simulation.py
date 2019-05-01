"""
    simulation.py

    This script implements a method of lines (MOL) solver for the general-relativistic compressible Euler equations using the radial-gauge, polar sliced metric (RGPS).  The scheme below uses a strong stability preserving Runge-Kutta 2nd-order integration.  The formulation of this solver and code follows closely to:

        * O'Connor & Ott (2010)
        * O'Connor (2015)
        * GR1D https://github.com/evanoconnor/GR1D

    NOTE
    ----
    There are still alot of legacy code/comments from previous versions of 
    this code, e.g. not all parameter lists for functions are updated/present.
"""
import numpy as np
import matplotlib.pyplot as plt
import importlib

from mesh import patch

from hydro_base.grid import Grid1D
from gr_radial_solver.custom_grid import CustomGrid1D
from gr_radial_solver.equations import Polytrope

from hydro_base.eos import GammaLawEOS, PolytropicEOS, NoPressureEOS
from hydro_base.reconstruct import Reconstruct1D, minmod
from hydro_base.riemann import HLLE1D, Rusanov1D
import hydro_base.variables as vars

from gr.metric import RGPSMetric
from gr import units

from simulation_null import NullSimulation, bc_setup_1d

from util import msg


def grid_setup_1d(rp):
    # Get mesh parameters
    nzones = rp.get_param("mesh.nzones")
    ncenter = rp.get_param("mesh.ncenter")
    ng = rp.get_param("mesh.nghost")

    dr_center = rp.get_param("mesh.dr_center")
    dr = rp.get_param("mesh.dr")

    r_const = rp.get_param("mesh.r_const")
    r_max = rp.get_param("mesh.r_max")

    xmin = rp.get_param('mesh.xmin')
    xmax = rp.get_param('mesh.xmax')

    # Convert to geometric units
    dr_center = units.convert(dr_center, units.CGS.length, units.REL.length)
    dr = units.convert(dr, units.CGS.length, units.REL.length)
    r_const = units.convert(r_const, units.CGS.length, units.REL.length)
    r_max = units.convert(r_max, units.CGS.length, units.REL.length)

    grid = CustomGrid1D(nzones, ncenter, dr_center, dr, r_const, r_max, ng)
    # grid = Grid1D()

    return grid


class MinmodReconstruct1D:
    """Piecewise-linear minmod reconstruction."""

    def __init__(self, grid: CustomGrid1D):
        self.grid = grid

    def interface_states(self, ilo, ihi, U, U_l, U_r):
        """Calculate the values of values at the cell faces

        Parameters
        ----------
        U : array of floats
            The values in the cells
        U_l : array of floats
            [out] The reconstructed left edge values
        U_r : array of floats
            [out] The reconstructed right edge values
        """
        # Set outer faces to outer cell values since there are no neighboring
        # states to compute the reconstruction from
        # U_l[:, ilo] = U_r[:, ilo] = U[:, ilo]
        # U_l[:, ihi] = U_r[:, ihi] = U[:, ihi]

        # Compute the base slopes
        dx = np.diff(self.grid.x[ilo:ihi+1])
        s = np.diff(U[:, ilo:ihi+1])/dx

        dxl = self.grid.x[ilo:ihi+1] - self.grid.x_i[ilo:ihi+1]
        dxr = self.grid.x_i[ilo+1:ihi+2] - self.grid.x[ilo:ihi+1]

        # Compute TVD slopes with minmod
        slope = minmod(s[:, :-1], s[:, 1:])

        # Use TVD slopes to reconstruct interfaces
        U_l[:, ilo+1:ihi] = U[:, ilo+1:ihi] - 0.5*slope*dxl[1:-1]
        U_r[:, ilo+1:ihi] = U[:, ilo+1:ihi] + 0.5*slope*dxr[1:-1]

    def interface_state(self, ilo, ihi, U, U_l, U_r):
        """Calculate the values of values at the cell faces

        Parameters
        ----------
        U : array of floats
            The values in the cells
        U_l : array of floats
            [out] The reconstructed left edge values
        U_r : array of floats
            [out] The reconstructed right edge values
        """
        # Compute the base slopes
        dx = np.diff(self.grid.x[ilo:ihi+1])
        s = np.diff(U[ilo:ihi+1])/dx

        dxl = self.grid.x[ilo:ihi+1] - self.grid.xi[ilo:ihi+1]
        dxr = self.grid.xi[ilo+1:ihi+2] - self.grid.x[ilo:ihi+1]

        # Compute TVD slopes with minmod
        slope = minmod(s[:-1], s[1:])

        # Use TVD slopes to reconstruct interfaces
        U_l[ilo+1:ihi] = U[ilo+1:ihi] - 0.5*slope*dxl[1:-1]
        U_r[ilo+1:ihi] = U[ilo+1:ihi] + 0.5*slope*dxr[1:-1]


class Simulation(NullSimulation):

    def __init__(self, solver_name, problem_name, rp, timers=None,
                 data_class=patch.CellCenterData1d):
        """Initialize the general-relativistic RPGS hydro simulation.
        Pass through parameters to `NullSimulation` while forcing the data
        type to be CellCenterData1d.
        """
        super().__init__(solver_name, problem_name, rp, timers, data_class)

    def initialize(self):
        """Initialize the simulation."""

        grid = grid_setup_1d(self.rp)
        self.grid = grid

        # create the variables
        my_data = patch.CellCenterData1d(grid)
        bc = bc_setup_1d(self.rp)[0]

        # Primitives
        my_data.register_var("rho0", bc)
        my_data.register_var("v", bc)
        my_data.register_var("eps", bc)

        # Conserved
        my_data.register_var("D", bc)
        my_data.register_var("S", bc)
        my_data.register_var("tau", bc)

        # Metric stuff
        my_data.register_var("m", bc)
        my_data.register_var("X", bc)
        my_data.register_var("phi", bc)

        # store the EOS gamma as an auxillary quantity so we can have a
        # self-contained object stored in output files to make plots.
        gamma = self.rp.get_param("eos.gamma")
        K = self.rp.get_param("eos.k")
        my_data.set_aux("gamma", gamma)
        my_data.set_aux("k", K)

        # Are we evolving the metric?
        self.evolve_metric = self.rp.get_param("gr.evolve_metric")
        my_data.set_aux("evolve_metric", self.evolve_metric)
        print(self.evolve_metric)

        # Get and store the atmosphere value used in the simulation
        self.atm_rho = self.rp.get_param("atmosphere.rho0")
        self.atm_eps = self.rp.get_param("atmosphere.eps")
        my_data.set_aux("atm_rho", self.atm_rho)
        my_data.set_aux("atm_eps", self.atm_eps)

        my_data.create()

        self.cc_data = my_data

        # Setup solver stuff
        self.eos = PolytropicEOS(K, gamma) # for tov
        # self.eos = NoPressureEOS(0.0, 1.67) # for os
        self.eqns = Polytrope(self.eos, grid, self.atm_rho, self.atm_eps)
        self.reconstruct = MinmodReconstruct1D(grid)

        riemann_solvers = {"hlle": HLLE1D, "rusanov": Rusanov1D}
        rsolver = self.rp.get_param("gr_radial_solver.riemann").lower()
        print("Using", rsolver, "Riemann solver")
        self.riemann = riemann_solvers[rsolver](self.eqns)

        # Setup temp storage for calculations
        shape = (grid.qx,)

        print(shape)
        # Cell-centered values
        self.U = [vars.ConservedVector1D(shape) for i in range(3)]
        self.V = vars.PrimitiveVector1D(shape)
        self.char = vars.CharacteristicVector1D(shape)
        self.source = vars.SourceVector1D(shape)

        self.F = vars.FluxVector1D((grid.qx+1,))

        self.dU = vars.ConservedVector1D(shape)

        self.rho0_old = np.zeros(shape, dtype=float)

        # Left-face values
        self.U_l = vars.ConservedVector1D(shape)
        self.V_l = vars.PrimitiveVector1D(shape)
        self.F_l = vars.FluxVector1D(shape)
        self.char_l = vars.CharacteristicVector1D(shape)

        # Right-face values
        self.U_r = vars.ConservedVector1D(shape)
        self.V_r = vars.PrimitiveVector1D(shape)
        self.F_r = vars.FluxVector1D(shape)
        self.char_r = vars.CharacteristicVector1D(shape)

        # Metric stuff
        self.m = np.zeros(shape)
        self.m_i = np.zeros(shape)
        self.m_zone = np.zeros(shape)

        self.phi = np.zeros(shape)
        self.phi_i = np.zeros(shape)
        self.dphi_dr = np.zeros(shape)

        self.X = np.zeros(shape)
        self.X_l = np.zeros(shape)
        self.X_r = np.zeros(shape)

        self.alpha = np.zeros(shape)
        self.alpha_l = np.zeros(shape)
        self.alpha_r = np.zeros(shape)

        # now set the initial conditions for the problem
        problem = importlib.import_module("gr_radial_solver.problems.{}".format(self.problem_name))
        problem.init_data(self.cc_data, self.rp, self.eqns)

        self.load_from_patch(0)
        self.apply_bcs(0)
        self.update_metric_vars(0)
        # self.eqns.prim2con(self.V, self.X, self.U[0])
        # self.apply_bcs(0)
        self.apply_bcs_metric()

    def load_from_patch(self, level):
        """Load internal data from the data patch.

        Parameters
        ----------
        level : int
            Which time level to load the patch to
        """
        ng = self.cc_data.grid.ng

        # "Pointers" to the variables in the patch
        D = self.cc_data.get_var("D")
        S = self.cc_data.get_var("S")
        tau = self.cc_data.get_var("tau")

        rho0 = self.cc_data.get_var("rho0")
        v = self.cc_data.get_var("v")
        eps = self.cc_data.get_var("eps")

        m = self.cc_data.get_var("m")
        phi = self.cc_data.get_var("phi")

        # Copy patch tointernal data
        self.U[level].density = D[:]
        self.U[level].momentum = S[:]
        self.U[level].energy = tau[:]

        self.V.density = rho0[:]
        self.V.velocity = v[:]
        self.V.specific_energy = eps[:]

        self.m[:] = m[:]
        self.phi[:] = phi[:]

    def save_to_patch(self, level):
        """Save internal data to the data patch.

        Parameters
        ----------
        level : int
            Which time level to save to the patch
        """
        # "Pointers" to the variables in the patch
        D = self.cc_data.get_var("D")
        S = self.cc_data.get_var("S")
        tau = self.cc_data.get_var("tau")

        rho0 = self.cc_data.get_var("rho0")
        v = self.cc_data.get_var("v")
        eps = self.cc_data.get_var("eps")

        m = self.cc_data.get_var("m")
        phi = self.cc_data.get_var("phi")

        # Copy internal data to the patch
        D[:] = self.U[level].density
        S[:] = self.U[level].momentum
        tau[:] = self.U[level].energy

        rho0[:] = self.V.density
        v[:] = self.V.velocity
        eps[:] = self.V.specific_energy

        m[:] = self.m
        phi[:] = self.phi

    def update_metric_vars(self, level):

        # For convenience below
        def calc_X(M, R):
            discrim = 1.0 - 2.0*M/R

            # Check if a black hole has formed
            assert discrim > 0.0, "Black hole formed!"

            return 1.0/np.sqrt(discrim)

        # For easier reference
        D, S, tau = self.U[level]
        rho0, v, eps = self.V

        # E = tau + D
        P = self.eos.pressure(rho0)

        h = 1.0 + eps + P/rho0
        W2 = 1.0/(1.0 - v*v)
        E = rho0*h*W2 - P

        r = self.grid.x
        r_i = self.grid.x_i
        vol = self.grid.volume

        vol_fac = 4.0/3.0*np.pi

        # Start and end points (note end point is a ghost cell)
        ilo = self.grid.ilo
        ihi = self.grid.ihi + 1  # self.grid.qx - 2

        # Calculate the mass of each zone and total enclosed masses

        # Inner zone
        self.m_zone[ilo] = E[ilo]*vol[ilo]
        self.m[ilo] = \
            self.m_zone[ilo] - vol_fac*E[ilo]*(r_i[ilo+1]**3 - r[ilo]**3)
        self.m_i[ilo] = 0.0

        self.X[ilo] = calc_X(self.m[ilo], r[ilo])

        # Calculate remaining zone and enclosed masses
        for i in range(ilo + 1, ihi + 1):
            self.m_zone[i] = E[i]*vol[i]

            self.m[i] = self.m[i-1] + self.m_zone[i] \
                - vol_fac*E[i]*(r_i[i+1]**3 - r[i]**3) \
                + vol_fac*E[i-1]*(r_i[i]**3 - r[i-1]**3)

            self.m_i[i] = self.m_i[i - 1] + E[i-1]*vol[i-1]

            self.X[i] = calc_X(self.m[i], r[i])

        self.M_tot = self.m_i[self.grid.ihi]

        self.X[ilo] = calc_X(self.m[ilo], r[ilo])
        # print(self.X)

        # Calculate interface X values
        self.X_r[ilo] = calc_X(self.m_i[ilo+1], r_i[ilo+1])
        self.X_l[ilo] = 1.0

        # One at a time to check for black holes
        for i in range(ilo + 1, ihi + 1):
            self.X_r[i] = calc_X(self.m_i[i+1], r_i[i+1])
            self.X_l[i] = calc_X(self.m_i[i], r_i[i])

        self.X_l[ihi+1] = calc_X(self.m_i[ihi+1], r_i[ihi+1])

        # Calculate dphi/dr

        h = 1.0 + eps[ilo:ihi+1] + P[ilo:ihi+1]/rho0[ilo:ihi+1]
        v2 = v[ilo:ihi+1]*v[ilo:ihi+1]
        W2 = 1.0/(1.0 - v2)
        self.dphi_dr[ilo:ihi+1] = self.X[ilo:ihi+1]**2 * \
            (self.m[ilo:ihi+1]/(r[ilo:ihi+1]**2) +
            4.0*np.pi*r[ilo:ihi+1]*(P[ilo:ihi+1] +
            rho0[ilo:ihi+1]*h*W2*v2))

        # Calculate phi
        self.phi[ilo] = r[ilo]*self.dphi_dr[ilo]
        self.phi_i[ilo] = 0.0

        for i in range(ilo+1, ihi+1):
            self.phi[i] = self.phi[i-1] + (r_i[i] - r[i-1])*self.dphi_dr[i-1] \
                + (r[i] - r_i[i])*self.dphi_dr[i]

            self.phi_i[i] = self.phi[i-1] + (r_i[i] - r[i-1])*self.dphi_dr[i-1]

        self.phi_i[ihi+1] = \
            self.phi_i[ihi] + (r_i[ihi+1] - r[ihi])*self.dphi_dr[ihi]

        # Match to Schwarzschild solution
        phi_bound = 0.5*np.log(1.0 - 2.0*self.m_i[ihi+1]/r_i[ihi+1])

        self.phi_i[ilo:ihi+2] += phi_bound - self.phi_i[ihi+1]
        self.phi[ilo:ihi+1] += phi_bound - self.phi_i[ihi+1]

        # Calculate the lapse
        self.alpha[ilo:ihi+1] = np.exp(self.phi[ilo:ihi+1])
        self.alpha_r[ilo:ihi+1] = np.exp(self.phi_i[ilo+1:ihi+2])
        self.alpha_l[ilo:ihi+2] = np.exp(self.phi_i[ilo:ihi+2])

    def method_compute_timestep(self):
        """
        Compute the advective timestep (CFL) constraint.  We use the
        driver.cfl parameter to control what fraction of the CFL
        step we actually take.
        """

        cfl = self.rp.get_param("driver.cfl")
        dx = self.cc_data.grid.dx
        ilo = self.grid.ilo
        ihi = self.grid.ihi

        self.eqns.speeds(self.V, self.char, abs=True)

        self.dt = \
            cfl*np.min(dx[ilo:ihi+1]/np.max(np.abs(self.char[:, ilo:ihi+1]),
            axis=0))

        # self.dt = cfl*min(dx)

    def apply_bcs(self, level):

        # Reflective BCs at inner boundary
        ng = self.grid.ng

        # Make sure r = 0 velocity is zero
        self.V_l.velocity[ng] = 0.0

        self.V[:, :ng] = self.V[:, 2*ng-1:ng-1:-1]
        self.V_l[:, :ng] = self.V_r[:, 2*ng-1:ng-1:-1]
        self.V_r[:, :ng] = self.V_l[:, 2*ng-1:ng-1:-1]

        # Flip velocity signs
        self.V.velocity[:ng] *= -1.0
        self.V_l.velocity[:ng] *= -1.0
        self.V_r.velocity[:ng] *= -1.0

        self.U[level][:, :ng] = self.U[level][:, 2*ng-1:ng-1:-1]

        # Outflow for outer boundaries
        ihi = self.grid.ihi
        for i in range(1, ng+1):
            self.V[:, ihi+i] = self.V[:, ihi]
            self.V_l[:, ihi+i] = self.V[:, ihi]
            self.V_r[:, ihi+i-1] = self.V[:, ihi]
            self.U[level][:, ihi+i] = self.U[level][:, ihi]

        self.V_r[:, -1] = self.V[:, ihi]

        # Prevent stuff from going in
        if self.V.velocity[ihi] > 0.0:
            self.V.velocity[ihi+1:] = self.V.velocity[ihi]
            self.V_l.velocity[ihi+1:] = self.V.velocity[ihi]
            self.V_r.velocity[ihi:] = self.V.velocity[ihi]
        else:
            self.V.velocity[ihi+1:] = 0.0
            self.V_l.velocity[ihi+1:] = 0.0
            self.V_r.velocity[ihi:] = 0.0

    def apply_bcs_metric(self):
        ng = self.grid.ng

        # Reflective inner BC
        self.X[:ng] = self.X[2*ng-1:ng-1:-1]
        self.X_l[:ng] = self.X_r[2*ng-1:ng-1:-1]
        self.X_r[:ng] = self.X_l[2*ng-1:ng-1:-1]

        self.alpha[:ng] = self.alpha[2*ng-1:ng-1:-1]
        self.alpha_l[:ng] = self.alpha_r[2*ng-1:ng-1:-1]
        self.alpha_r[:ng] = self.alpha_l[2*ng-1:ng-1:-1]

        self.phi[:ng] = self.phi[2*ng-1:ng-1:-1]
        self.dphi_dr[:ng] = self.dphi_dr[2*ng-1:ng-1:-1]
        self.m[:ng] = self.m[2*ng-1:ng-1:-1]

        # Outflow outer BC
        ihi = self.grid.ihi
        self.X[ihi+1:] = self.X_r[ihi]
        self.X_l[ihi+1:] = self.X_r[ihi]
        self.X_r[ihi+1:] = self.X_r[ihi]

        self.alpha[ihi+1:] = self.alpha_r[ihi]
        self.alpha_l[ihi+1:] = self.alpha_r[ihi]
        self.alpha_r[ihi+1:] = self.alpha_r[ihi]

        self.m[ihi+1:] = self.m[ihi]
        self.dphi_dr[ihi+1:] = self.dphi_dr[ihi]
        self.phi[ihi+1:] = self.phi[ihi]

    def RHS(self, level):
        """Calculate the RHS of the special-relativistic compressible Euler 
        equations.
        """
        dx = self.cc_data.grid.dx
        ng = self.cc_data.grid.ng
        ilo = self.grid.ilo
        ihi = self.grid.ihi

        r = self.cc_data.grid.x
        r_l = self.cc_data.grid.x_i[ilo:ihi+1]
        r_r = self.cc_data.grid.x_i[ilo+1:ihi+2]

        # Calculate conserved variables
        # Reconstruct the primitive variables at the cell interfaces
        self.reconstruct.interface_states(ilo-1, ihi+1,
                                          self.V, self.V_l, self.V_r)

        self.reconstruct.interface_states(ilo-1, ihi+1,
                                          self.U[level], self.U_l, self.U_r)

        # self.eqns.prim2con(self.V, self.X, self.U)
        # self.eqns.prim2con(self.V_l, self.X_l, self.U_l)
        # self.eqns.prim2con(self.V_r, self.X_r, self.U_r)

        # Update BCs
        self.apply_bcs(level)

        if self.evolve_metric == 1:
            self.apply_bcs_metric()

        # Calculate the fluxes at the cell interfaces
        self.eqns.fluxes(self.U_l, self.V_l, self.F_l)
        self.eqns.fluxes(self.U_r, self.V_r, self.F_r)

        # Calculate the characterstic speeds at the interface states
        self.eqns.speeds(self.V_l, self.char_l)
        self.eqns.speeds(self.V_r, self.char_r)

        # Solve the local Riemann problem at the celll interfaces
        self.F[:, 0] = self.F[:, -1] = 0.0
        self.riemann.fluxes(self.U_l[:, :-1], self.U_r[:, 1:],
                            self.V_l[:, :-1], self.V_r[:, 1:],
                            self.F_l[:, :-1], self.F_r[:, 1:],
                            self.char_l[:, :-1], self.char_r[:, 1:],
                            self.F[:, 1:-1])

        X_l = self.X_l
        X_r = self.X_r

        alpha_l = self.alpha_l
        alpha_r = self.alpha_r

        self.dU[:, ilo:ihi+1] = alpha_r[ilo:ihi+1]*r_r**2/X_r[ilo:ihi+1]*self.F[:, ilo+1:ihi+2]
        self.dU[:, ilo:ihi+1] -= alpha_l[ilo:ihi+1]*r_l**2/X_l[ilo:ihi+1]*self.F[:, ilo:ihi+1]
        self.dU[:, ilo:ihi+1] *= -1.0/(r[ilo:ihi+1]*r[ilo:ihi+1]*dx[ilo:ihi+1])

        # Calculate any source terms
        self.eqns.sources(self.U[level], self.V, self.alpha, self.X, self.m,
                          self.source)
        self.dU[:] += self.source[:]

        # Don't update the ghost cells
        self.dU[:, :ng] = 0.0
        self.dU[:, -ng:] = 0.0

    def evolve(self):
        """
        Evolve the equations of special relativistic compressible
        hydrodynamics through a timestep dt.
        """
        ng = self.cc_data.grid.ng
        tm_evolve = self.tc.timer("evolve")
        tm_evolve.begin()

        ilo = self.grid.ilo
        ihi = self.grid.ihi

        self.rho0_old[:] = np.copy(self.V.density[:])
        self.eqns.prim2con(self.V, self.X, self.U)
        # First step
        self.RHS(0)
        self.U[1][:] = self.U[0][:] + self.dt*self.dU
        self.eqns.con2prim(self.U[1][:, ilo:ihi+1], self.X[ilo:ihi+1],
                           self.rho0_old[ilo:ihi+1], self.V[:, ilo:ihi+1])
        if self.evolve_metric == 1:
            self.update_metric_vars(1)
        # self.rho0_old[:] = np.copy(self.V.density[:])
        # Second step
        # self.update_metric_vars(1)
        self.RHS(1)
        self.U[2][:] = 0.5*(self.U[0][:] + self.U[1][:] + self.dt*self.dU)
        self.eqns.con2prim(self.U[2][:, ilo:ihi+1], self.X[ilo:ihi+1],
                           self.rho0_old[ilo:ihi+1], self.V[:, ilo:ihi+1])
        if self.evolve_metric == 1:
            self.update_metric_vars(2)

        # Rotate time levels
        U = self.U[0]
        self.U[0] = self.U[2]
        self.U[2] = self.U[1]
        self.U[1] = U
        self.save_to_patch(0)

        # increment the time
        self.cc_data.t += self.dt
        self.n += 1

        tm_evolve.end()

    def dovis(self):
        """
        Do runtime visualization.
        """

        plt.clf()

        plt.rc("font", size=10)

        rho0 = self.cc_data.get_var("rho0")
        v = self.cc_data.get_var("v")
        eps = self.cc_data.get_var("eps")
        P = self.eos.pressure(rho0)
        m = self.cc_data.get_var("m")

        # Convert to CGS
        rho0 = units.convert(rho0, units.REL.density, units.CGS.density)
        # v = units.convert(rho0, units.REL.density, units.CGS.density)
        # eps = units.convert(eps, units.REL.density, units.CGS.density)
        P = units.convert(P, units.REL.pressure, units.CGS.pressure)

        myg = self.cc_data.grid

        # fields = [rho0, v, P, eps]
        fields = [rho0, m, v, P]
        # field_names = ["\\rho_0", "v", "P", "\\epsilon"]
        field_names = ["\\rho_0", "m", "v", "P"]  # , "\\epsilon"]
        field_units = ["g/cc", "$M_\\odot$", "c", "g cm$^{-1}$ s$^{-2}$"]

        _, axs = plt.subplots(2, 2, sharex=True, constrained_layout=True,
                              num=1)

        x = myg.x[myg.ng:-myg.ng]

        x = units.convert(x, units.REL.length, units.CGS.length)*1e-5

        t = self.cc_data.t
        t = units.convert(t, units.REL.time, units.CGS.time)

        for n in range(len(fields)):
            var = fields[n]

            i = n // 2
            j = n % 2

            axs[i, j].plot(x, var.v())

            axs[i, j].set_xlabel("$r$, (km)")
            # axs[i, j].set_ylabel("${:s}$".format(field_names[n]))

            axs[i, j].set_title("${:s}$, ({:s})".format(field_names[n], field_units[n]))

        # plt.figtext(0.05, 0.0125, "t = {:10.5f}".format(self.cc_data.t))
        plt.figtext(0.05, 0.0125, "t = {:.3e} s".format(t))

        plt.pause(0.0001)
        plt.draw()
