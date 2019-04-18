"""
    simulation.py

    This script implements a method of lines (MOL) solver for the special-relativistic compressible Euler equations.  The scheme below uses a strong stability preserving Runge-Kutta 2nd-order integration.  The formulation of this solver and code follows closely to:

        * Rezzolla & Zanotti "Relativistic Hydrodynamics" (2013)
        * David Radice, Lecture+Notes+Code
          JINA Neutron Star Merger Summer School (2018)
          https://github.com/dradice/JINA_MSU_School_2018/tree/master/Radice
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
import importlib

from scipy.integrate import solve_ivp, cumtrapz

from mesh import patch
from mesh import integration

from simulation_null import NullSimulation, bc_setup_1d
import util.plot_tools as plot_tools

from hydro_base.eos import GammaLawEOS, PolytropicEOS
from hydro_base.reconstruct import Reconstruct1D, MinmodReconstruct1D, minmod
from hydro_base.riemann import HLLE1D, Rusanov1D
import hydro_base.variables as vars

from gr.hydro.equations import RPGSEquations
from gr.metric import RGPSMetric
from gr import units

from gr_radial_solver.polytrope import Polytrope

from util import msg


def grid_setup_1d(rp, ng=2):
    nx = rp.get_param("mesh.nx")

    try:
        xmin = rp.get_param("mesh.xmin")
    except KeyError:
        xmin = 0.0
        msg.warning("mesh.xmin not set, defaulting to 0.0")

    try:
        xmax = rp.get_param("mesh.xmax")
    except KeyError:
        xmax = 1.0
        msg.warning("mesh.xmax not set, defaulting to 1.0")

    xmin = units.convert(xmin, units.CGS.length, units.REL.length)
    xmax = units.convert(xmax, units.CGS.length, units.REL.length)

    my_grid = patch.RadialGrid(nx, xmin=xmin, xmax=xmax, ng=ng)

    return my_grid


class MinmodReconstructRadial(Reconstruct1D):
    """Piecewise-linear minmod reconstruction."""

    def interface_states(self, grid: patch.RadialGrid, U, U_l, U_r):
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
        dx = np.diff(grid.x)
        s = np.diff(U)/dx

        # Compute TVD slopes with minmod
        slope = minmod(s[:, :-1], s[:, 1:])

        # print(grid.x.shape, grid.xl.shape, grid.xr.shape, grid.dxl.shape, grid.dxr.shape)

        # Use TVD slopes to reconstruct interfaces
        U_l[:, 1:-1] = U[:, 1:-1] - 0.5*slope*grid.dxl[1:-1]
        U_r[:, 1:-1] = U[:, 1:-1] + 0.5*slope*grid.dxr[1:-1]

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
        dx = np.diff(grid.x)
        s = np.diff(U)/dx

        # Compute TVD slopes with minmod
        slope = minmod(s[:-1], s[1:])

        # Use TVD slopes to reconstruct interfaces
        U_l[1:-1] = U[1:-1] - 0.5*slope*grid.dxl[1:-1]
        U_r[1:-1] = U[1:-1] + 0.5*slope*grid.dxl[1:-1]


class Simulation(NullSimulation):

    def __init__(self, solver_name, problem_name, rp, timers=None,
                 data_class=patch.CellCenterData1d):
        """Initialize the special-relativistic compressible hydro simulation.  
        Pass through parameters to `NullSimulation` while forcing the data 
        type to be CellCenterData1d.
        """
        super().__init__(solver_name, problem_name, rp, timers, data_class)

    def initialize(self):
        """Initialize the simulation."""
        my_grid = grid_setup_1d(self.rp, ng=2)

        print(my_grid.x[0])

        # create the variables
        my_data = patch.CellCenterData1d(my_grid)
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
        my_data.register_var("Phi", bc)

        # store the EOS gamma as an auxillary quantity so we can have a
        # self-contained object stored in output files to make plots.
        gamma = self.rp.get_param("eos.gamma")
        K = self.rp.get_param("eos.k")
        my_data.set_aux("gamma", gamma)
        my_data.set_aux("k", K)

        my_data.create()

        self.cc_data = my_data

        self.atm_rho = 1e-10
        self.atm_eps = 1e-10

        # Setup solver stuff
        self.eos = PolytropicEOS(K, gamma)
        self.eqns = Polytrope(self.eos, my_grid, self.atm_rho, self.atm_eps)
        self.reconstruct = MinmodReconstructRadial()

        riemann_solvers = {"hlle": HLLE1D, "rusanov": Rusanov1D}
        rsolver = self.rp.get_param("gr_radial_solver.riemann").lower()
        print("Using", rsolver, "Riemann solver")
        self.riemann = riemann_solvers[rsolver](self.eqns)

        # Setup temp storage for calculations
        shape = (my_grid.qx,)

        # Cell-centered values
        self.U = [vars.ConservedVector1D(shape) for i in range(3)]
        self.V = vars.PrimitiveVector1D(shape)
        self.char = vars.CharacteristicVector1D(shape)
        self.source = vars.SourceVector1D(shape)

        self.F = vars.FluxVector1D((my_grid.qx+1,))

        self.dU = vars.ConservedVector1D(shape)
        self.U_new = vars.ConservedVector1D(shape)

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

        # now set the initial conditions for the problem
        problem = importlib.import_module("gr_radial_solver.problems.{}".format(self.problem_name))
        problem.init_data(self.cc_data, self.rp, self.eqns)

        m = self.cc_data.get_var("m")
        Phi = self.cc_data.get_var("Phi")

        self.M = m.max()
        self.R = my_grid.x[m == self.M][0]

        # rho0_c = self.cc_data.get_var("rho0")[0]
        # eps_c = self.cc_data.get_var("eps")[0]
        # v_c = self.cc_data.get_var("v")[0]

        self.V_c = vars.PrimitiveVector1D()
        self.V_c.density = self.cc_data.get_var("rho0")[0]
        self.V_c.velocity = self.cc_data.get_var("v")[0]
        self.V_c.specific_energy = self.cc_data.get_var("eps")[0]

        self.U_c = vars.ConservedVector1D()
        self.U_c.density = self.cc_data.get_var("D")[0]
        self.U_c.momentum = self.cc_data.get_var("S")[0]
        self.U_c.energy = self.cc_data.get_var("tau")[0]

        self.metric = RGPSMetric(m, Phi, my_grid.x)

        self.load_from_patch(0)

    def method_compute_timestep(self):
        """
        Compute the advective timestep (CFL) constraint.  We use the
        driver.cfl parameter to control what fraction of the CFL
        step we actually take.
        """

        cfl = self.rp.get_param("driver.cfl")
        dx = self.cc_data.grid.dx
        ng = self.cc_data.grid.ng

        self.eqns.speeds(self.U[0], self.V, self.metric, self.char)

        # mask = self.char[:, ng:-ng] != 0.0
        # self.dt = cfl*np.min(dx[mask]/np.max(np.abs(self.char[:, ng:-ng][mask]), axis=0))
        self.dt = cfl*np.min(dx[ng:-ng]/np.max(np.abs(self.char[:, ng:-ng]), axis=0))
        # self.dt = cfl*dx

    def RHS(self, level):
        """Calculate the RHS of the special-relativistic compressible Euler 
        equations.
        """
        dx = self.cc_data.grid.dx
        ng = self.cc_data.grid.ng

        # Reconstruct the primitive variables at the cell interfaces
        self.reconstruct.interface_states(self.cc_data.grid,
                                          self.V, self.V_l, self.V_r)

        # self.clamp_primitives(level)

        # Calculate conserved variables at the cell interfaces
        self.eqns.prim2con(self.V_l, self.metric, self.U_l)
        self.eqns.prim2con(self.V_r, self.metric, self.U_r)

        # Calculate the fluxes at the cell interfaces
        self.eqns.fluxes(self.U_l, self.V_l, self.metric, self.F_l)
        self.eqns.fluxes(self.U_r, self.V_r, self.metric, self.F_r)

        # Calculate the characterstic speeds at the interface states
        self.eqns.speeds(self.U_l, self.V_l, self.metric, self.char_l)
        self.eqns.speeds(self.U_r, self.V_r, self.metric, self.char_r)

        # Solve the local Riemann problem at the celll interfaces
        self.F[:, 0] = self.F[:, -1] = 0.0
        self.riemann.fluxes(self.U_l[:, :-1], self.U_r[:, 1:],
                            self.V_l[:, :-1], self.V_r[:, 1:],
                            self.F_l[:, :-1], self.F_r[:, 1:],
                            self.char_l[:, :-1], self.char_r[:, 1:],
                            self.F[:, 1:-1])

        r = self.cc_data.grid.x
        r_l = self.cc_data.grid.xl
        r_r = self.cc_data.grid.xr

        # Need alpha and X on interfaces as well
        X_l = self.U_l.density + self.U_l.energy
        X_r = self.U_r.density + self.U_r.energy

        alpha_l = np.zeros_like(r)
        alpha_r = np.zeros_like(r)

        self.reconstruct.interface_state(self.cc_data.grid,
                                         self.metric.X, X_l, X_r)
        self.reconstruct.interface_state(self.cc_data.grid,
                                         self.metric.alpha, alpha_l, alpha_r)

        # Spatial discretization
        # self.dU[:] = 0.0
        # self.dU[:] -= np.diff(self.F, axis=1)
        # self.dU[:] /= dx

        self.dU[:] = alpha_r*r_r**2/X_r*self.F[:, 1:]
        self.dU[:] -= alpha_l*r_l**2/X_l*self.F[:, :-1]
        self.dU[:] *= -1.0/(r*r*dx)

        # Calculate any source terms
        self.eqns.sources(self.U[level], self.V, self.metric, self.source)
        self.dU[:] += self.source[:]

        # Don't update the ghost cells
        self.dU[:, :ng] = 0.0
        self.dU[:, -ng:] = 0.0

    def apply_BC_hack(self, level):
        """Convert internal stored variables back to CellCenterData1d.  Pyro
        inherenetly is Fortran-esque in its handling of mult-dimensional
        arrays (which is highly inefficient in python/numpy which is by
        default C-style), so for optimal use, grid data is maninpulated by
        interal data structures in this class, and converted back to the
        CellCenterData1d patch for application of boundary conditions since
        this machinery is already present courtesy of `ArrayIndexer`.

        Parameters
        ----------
        level : int
            Which time level to apply the BC update to
        """
        # ng = self.cc_data.grid.ng

        # # Get "pointers" to patch data
        # D = self.cc_data.get_var("D")
        # S = self.cc_data.get_var("S")
        # tau = self.cc_data.get_var("tau")

        # # Copy specified time level data to patch
        # D.v()[:] = self.U[level].density[ng:-ng]
        # S.v()[:] = self.U[level].momentum[ng:-ng]
        # tau.v()[:] = self.U[level].energy[ng:-ng]

        # # Apply the BCs
        # self.cc_data.fill_BC_all()
        self.U[level][:, 0] = self.U_c
        self.V[:, 0] = self.V_c

    def save_to_patch(self, level):
        """Same idea as the BC hack; save internal data to the data patch.

        Parameters
        ----------
        level : int
            Which time level to save to the patch
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
        Phi = self.cc_data.get_var("Phi")

        # Copy internal data to the patch
        D[:] = self.U[level].density
        S[:] = self.U[level].momentum
        tau[:] = self.U[level].energy

        rho0[:] = self.V.density
        v[:] = self.V.velocity
        eps[:] = self.V.specific_energy

        m[:] = self.metric.m
        Phi[:] = self.metric.Phi

    def load_from_patch(self, level):
        """Same idea as the BC hack; load internal data from the data patch.

        Parameters
        ----------
        level : int
            Which time level to load the patch to
        """
        ng = self.cc_data.grid.ng

        # "Pointers" to the variables in the patch
        D = self.cc_data.get_var("D")
        S = self.cc_data.get_var("S")
        E = self.cc_data.get_var("tau")

        rho0 = self.cc_data.get_var("rho0")
        u = self.cc_data.get_var("v")
        eps = self.cc_data.get_var("eps")

        m = self.cc_data.get_var("m")
        Phi = self.cc_data.get_var("Phi")

        self.metric = RGPSMetric(m, Phi, self.cc_data.grid.x)

        # Copy patch tointernal data
        self.U[level].density = D[:]
        self.U[level].momentum = S[:]
        self.U[level].energy = E[:]

        self.V.density = rho0[:]
        self.V.velocity = u[:]
        self.V.specific_energy = eps[:]

    def clamp_primitives(self, level):
        rho0_l, v_l, eps_l = self.V_l
        rho0_r, v_r, eps_r = self.V_r
        rho0, v, eps = self.V

        mask = rho0_l < 1e-8
        rho0_l[mask] = 1e-8
        v_l[mask] = 0.0

        mask = rho0_r < 1e-8
        rho0_r[mask] = 1e-8
        v_r[mask] = 0.0

        mask = rho0 < 1e-8
        rho0[mask] = 1e-8
        v[mask] = 0.0

        _, eps_l[:] = self.eos.from_density(rho0_l)
        _, eps_r[:] = self.eos.from_density(rho0_r)
        _, eps[:] = self.eos.from_density(rho0)

        # For sanity clamp the central zone to initial values
        self.V[:, 0] = self.V_c

        self.eqns.prim2con(self.V_l, self.metric, self.U_l)
        self.eqns.prim2con(self.V_r, self.metric, self.U_r)
        self.eqns.prim2con(self.V, self.metric, self.U[level])

    def update_metric_vars(self):
        r = self.cc_data.grid.x

        rho0, v, eps = self.V
        P = self.eos.pressure(rho0)

        W = 1.0/np.sqrt(1.0 - v*v)

        # Enthalpy
        h = np.ones_like(P) + eps
        mask = rho0 > 1e-8
        h[mask] = 1.0 + eps[mask] + P[mask]/rho0[mask]
        # h = 1.0 + eps + P/rho0

        ridx = np.argmax(r == self.R)
        # R = r[ridx+1]

        X = rho0*h*W*W - P
        # X[ridx:] = 0.0

        m_integrand = 4.0*np.pi*X*r*r

        m = cumtrapz(m_integrand, r, initial=self.metric.m[0])

        # ridx = np.argmax(r == self.R)
        R = self.R  # r[ridx+1]

        m[ridx:] = self.M

        Phi_integrand = X*X*(m/(r*r) + 4.0*np.pi*(X + P)*v*v + P)

        Phi = cumtrapz(Phi_integrand, r, initial=self.metric.Phi[0])

        Phi_R = 0.5*np.log(1.0 - 2*self.M/R)

        offset = Phi_R - Phi[ridx+1]
        Phi += offset

        self.metric = RGPSMetric(m, Phi, r)

    def evolve(self):
        """
        Evolve the equations of special relativistic compressible
        hydrodynamics through a timestep dt.
        """
        ng = self.cc_data.grid.ng
        tm_evolve = self.tc.timer("evolve")
        tm_evolve.begin()

        # First step
        self.RHS(0)
        self.U[1][:] = self.U[0][:] + self.dt*self.dU
        self.apply_BC_hack(1)
        self.eqns.con2prim(self.U[1][:, ng:-ng], self.metric, self.V[:, ng:-ng])
        # self.update_metric_vars()

        # Second step
        self.RHS(1)
        self.U[2][:] = 0.5*(self.U[0][:] + self.U[1][:] + self.dt*self.dU)
        self.apply_BC_hack(2)
        self.eqns.con2prim(self.U[2][:, ng:-ng], self.metric, self.V[:, ng:-ng])
        # self.update_metric_vars()

        # Rotate time levels
        U = self.U[0]
        self.U[0] = self.U[2]
        self.U[2] = self.U[1]
        self.U[1] = U
        # self.clamp_primitives(0)
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

        myg = self.cc_data.grid

        fields = [rho0, v, P, eps]
        field_names = ["\\rho_0", "v", "P", "\\epsilon"]

        _, axs = plt.subplots(2, 2, sharex=True, constrained_layout=True,
                              num=1)

        x = myg.x[myg.ng:-myg.ng]

        for n in range(len(fields)):
            var = fields[n]

            i = n // 2
            j = n % 2

            axs[i, j].plot(x, var.v())

            axs[i, j].set_xlabel("$r$")
            axs[i, j].set_ylabel("${:s}$".format(field_names[n]))

            axs[i, j].set_title("${:s}$".format(field_names[n]))

        plt.figtext(0.05, 0.0125, "t = {:10.5f}".format(self.cc_data.t))

        plt.pause(0.001)
        plt.draw()
