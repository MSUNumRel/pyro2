"""
TODO: Remove one index from all variables, since we are using
CellCenterData1d and not CellCenterData1d
"""
from __future__ import print_function

import importlib

import numpy as np
import matplotlib.pyplot as plt

import compressible.BC as BC
import compressible.eos as eos
import compressible.derives as derives
import compressible.unsplit_fluxes as flx
import mesh.boundary as bnd
import mesh.patch as patch
from simulation_null import NullSimulation, grid_setup_1d, bc_setup_1d
import util.plot_tools as plot_tools
import particles.particles as particles
from gr.tensor import ThreeVector, Tensor

from scipy import optimize 

class Variables(object):
    """
    a container class for easy access to the different compressible
    variable by an integer key
    """
    def __init__(self, myd):
        """myd : CellCenterData1d object"""
        self.nvar = len(myd.names)
        print(myd.names)

        # conserved variables -- we set these when we initialize for
        # they match the CellCenterData2d object
        self.idens = myd.names.index("density")
        self.imom  = myd.names.index("momentum")
        self.iener = myd.names.index("energy")

        # if there are any additional variable, we treat them as
        # passively advected scalars
        self.naux = self.nvar - 3
        if self.naux > 0:
            self.irhox = 4
        else:
            self.irhox = -1

        # primitive variables
        self.nq = 5 + self.naux

        self.irho = 0
        self.iu = 1
        self.ip = 2
        self.im = 3
        self.ipot = 4

        if self.naux > 0:
            self.ix = 4   # advected scalar
        else:
            self.ix = -1



def cons_to_prim(U, gamma, ivars, myg, metric):
    """ 
    Convert an input vector of conserved variables (GR) to primitive variables (GR) M.A.P.
    Consistent with Relativistic Hydrodynamics (Rezzolla & Zanotti)
    
    Input
    -----
    U - vector of conserved variables
    gamma - ratio of specific heats used in gamma law EOS
    ivars - variables used to label conserved variables (Defined in eqn. (5.27) of Baumgarte & Shapiro)
      D   - .idens
      S   - .imom
      tau - .iener


    myg : Grid1D object
        defines discretization, see mesh.patch.Grid1D

    metric : Metric object
        defines spatial metric, see gr.metric.Metric

    Output
    ------
    q - Vector of primitive variables

    """
    #calculate pressure
    def f_p(p, U, gamma, metric, i):
    
        #shortcut variables 
        upd = U[i,ivars.iener] + p + U[i,ivars.idens]
        
        gss = metric.inv_g.xx*U[i,ivars.imom]*q[i,ivars.imom]   

        #density in terms of conservative vars and pressure
        rho = U[i,ivars.idens]/upd*np.sqrt(upd**2 - gss)

        #specific internal energy in terms of conservative vars and pressure
        eps = 1./U[i,ivars.idens]*(np.sqrt(upd**2 - gss) - upd/np.sqrt(upd**2 - gss) - U[i,ivars.idens])

        return p - rho*eps*(gamma - 1) #hard code version of EOS call (for ease of zero finder)

    #x0 = np.zeros_like(U[:,0])

    #print('Ucon2prim')
    #print(U[:,ivars.idens])
    #print(U[:,ivars.iener])
    #print(U[:,ivars.imom])


  
    #array to hold newly calculated primitives
    q = myg.scratch_array(nvar=ivars.nq)

    #solve for pressure numerically
    for i in range(0,len(U[:,ivars.idens])):
        q[i, ivars.ip] = optimize.brentq(f_p,-0.1,1.e20,args=(U,gamma,metric,i)) #different limit b/c of limits?

    #shortcut variables 
    upd = U[:,ivars.iener] + q[:, ivars.ip] + U[:,ivars.idens]  
 
    gss = metric.inv_g.xx*U[:,ivars.imom]*U[:,ivars.imom]

    #solve for density 
    q[:, ivars.irho] =  U[:,ivars.idens]/upd*np.sqrt(upd**2 - gss)
 
    #create variable for spec. int. energy (eps)
    eps = np.copy(q[:,ivars.irho])
    
    #assign eps values
    eps = 1./U[:,ivars.idens]*(np.sqrt(upd**2 - gss) - upd/np.sqrt(upd**2 - gss) - U[:,ivars.idens])

    #function calculating velocity
    def f_v(v,U,q,i):
        #velocity & S relation as used in O'Connor & Ott (2010)

        return (U[i,ivars.imom] - (q[i,ivars.irho] + q[i,ivars.irho]*eps[i] + q[i,ivars.ip])*(1 - v**2)*v)

    #solve for velocity numerically
    for i in range(0,len(U[:,ivars.idens])):
        q[i, ivars.iu] = optimize.brentq(f_v,-0.99,0.99,args=(U,q,i)) #different limit b/c of limits?

    #Below is for additional variables to be treated as 'passively advected scalars' should I keep it?
    if ivars.naux > 0:
        for nq, nu in zip(range(ivars.ix, ivars.ix+ivars.naux),
                          range(ivars.irhox, ivars.irhox+ivars.naux)):
            q[:, nq] = U[:, nu]/q[:, ivars.irho]

    return q


def prim_to_cons(q, gamma, ivars, myg, metric):
    """ convert an input vector of primitive variables to conserved variables 

    Variables
    ---------
    q : np array like
        primitive variables

    gamma : float
        polytropic constant

    ivars : Variables object like
        contains keyword based lookup for indexes of both conservative
        variables U and primitive variables q 

    myg : Grid1D object
        defines discretization, see mesh.patch.Grid1D

    metric : Metric object
        defines spatial metric, see gr.metric.Metric
        """

    #print('qprim2con')
    #print(q[:,ivars.idens])
    #print(q[:,ivars.iu])
    #print(q[:,ivars.ip])


    U = myg.scratch_array(nvar=ivars.nvar)

    v      = q[:, ivars.iu]
    #lapse  = np.exp( q[:, ivars.ipot] )
    P      = q[:, ivars.ip]
    rho0   = q[:, ivars.irho]
    rhoe   = eos.rhoe( gamma, P )
    rho0_h = rho0 + rhoe + P
    W      = np.sqrt(1 - v**2)

    # see O'Connor and Ott 2010, section 2. 
    X      = rho0_h*W**2 - P
    X = X/X #This is just a hack need to change
    # see O'Connor and Ott 2010, section 2.
    v_r    = v / X

    #print('X')
    #print(X)
    #print(rho0)
    #print(W)

    U[:, ivars.idens] = X * rho0 * W
    U[:, ivars.imom]  = rho0_h * W**2 * v
    U[:, ivars.iener] = rho0_h * W**2 - P - U[:, ivars.idens] 

    if ivars.naux > 0:
        for nq, nu in zip(range(ivars.ix, ivars.ix+ivars.naux),
                range(ivars.irhox, ivars.irhox+ivars.naux)):
            U[:, nu] = q[:, nq]*q[:, ivars.irho]

    return U


class Simulation(NullSimulation):
    """The main simulation class for the corner transport upwind
    compressible hydrodynamics solver

    """
    
    def __init__(self, *args, data_class=patch.CellCenterData1d, **kwargs):
        super().__init__(*args, data_class = data_class, **kwargs)

    def initialize(self, extra_vars=None, ng=4):
        """
        Initialize the grid and variables for compressible flow and set
        the initial conditions for the chosen problem.
        """
        my_grid = grid_setup_1d(self.rp, ng=ng)
        my_data = self.data_class(my_grid)

        # define solver specific boundary condition routines
        bnd.define_bc("hse", BC.user, is_solid=False)
        bnd.define_bc("ramp", BC.user, is_solid=False)  # for double mach reflection problem

        bc, bc_xodd = bc_setup_1d(self.rp)

        # are we dealing with solid boundaries? we'll use these for
        # the Riemann solver
        self.solid = bnd.bc_is_solid_1d(bc)

        # density and energy
        my_data.register_var("density", bc)
        my_data.register_var("energy", bc)
        my_data.register_var("momentum", bc_xodd)
        # my_data.register_var("y-momentum", bc_yodd)

        # any extras?
        if extra_vars is not None:
            for v in extra_vars:
                my_data.register_var(v, bc)

        # store the EOS gamma as an auxillary quantity so we can have a
        # self-contained object stored in output files to make plots.
        # store grav because we'll need that in some BCs
        my_data.set_aux("gamma", self.rp.get_param("eos.gamma"))
        my_data.set_aux("grav", self.rp.get_param("compressible.grav"))

        my_data.create()

        self.cc_data = my_data

        if self.rp.get_param("particles.do_particles") == 1:
            self.particles = particles.Particles(self.cc_data, bc, self.rp)

        # some auxillary data that we'll need to fill GC in, but isn't
        # really part of the main solution
        aux_data = self.data_class(my_grid)
        # aux_data.register_var("ymom_src", bc_yodd)
        aux_data.register_var("E_src", bc)
        aux_data.create()
        self.aux_data = aux_data

        self.ivars = Variables(my_data)

        # derived variables
        self.cc_data.add_derived(derives.derive_primitives)

        # initial conditions for the problem
        problem = importlib.import_module("{}.problems.{}".format(
            self.solver_name, self.problem_name))
        print(problem)
        problem.init_data(self.cc_data, self.rp)

        if self.verbose > 0:
            print(my_data)

    def method_compute_timestep(self):
        """
        The timestep function computes the advective timestep (CFL)
        constraint.  The CFL constraint says that information cannot
        propagate further than one zone per timestep.

        We use the driver.cfl parameter to control what fraction of the
        CFL step we actually take.
        """

        cfl = self.rp.get_param("driver.cfl")

        # get the variables we need
        u, v, cs = self.cc_data.get_var(["velocity", "soundspeed"])

        # the timestep is min(dx/(|u| + cs), dy/(|v| + cs))
        xtmp = self.cc_data.grid.dx/(abs(u) + cs)
        ytmp = self.cc_data.grid.dy/(abs(v) + cs)

        self.dt = cfl*float(min(xtmp.min(), ytmp.min()))

    def evolve(self):
        """
        Evolve the equations of compressible hydrodynamics through a
        timestep dt.
        """

        tm_evolve = self.tc.timer("evolve")
        tm_evolve.begin()

        dens = self.cc_data.get_var("density")
        ymom = self.cc_data.get_var("y-momentum")
        ener = self.cc_data.get_var("energy")

        grav = self.rp.get_param("compressible.grav")

        myg = self.cc_data.grid

        Flux_x, Flux_y = flx.unsplit_fluxes(self.cc_data, self.aux_data, self.rp,
                                            self.ivars, self.solid, self.tc, self.dt)

        old_dens = dens.copy()
        old_ymom = ymom.copy()

        # conservative update
        dtdx = self.dt/myg.dx
        dtdy = self.dt/myg.dy

        for n in range(self.ivars.nvar):
            var = self.cc_data.get_var_by_index(n)

            var.v()[:, :] += \
                dtdx*(Flux_x.v(n=n) - Flux_x.ip(1, n=n)) + \
                dtdy*(Flux_y.v(n=n) - Flux_y.jp(1, n=n))

        # gravitational source terms
        ymom[:, :] += 0.5*self.dt*(dens[:, :] + old_dens[:, :])*grav
        ener[:, :] += 0.5*self.dt*(ymom[:, :] + old_ymom[:, :])*grav

        if self.particles is not None:
            self.particles.update_particles(self.dt)

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

        # we do this even though ivars is in self, so this works when
        # we are plotting from a file
        ivars = Variables(self.cc_data)

        # access gamma from the cc_data object so we can use dovis
        # outside of a running simulation.
        gamma = self.cc_data.get_aux("gamma")

        q = cons_to_prim(self.cc_data.data, gamma, ivars, self.cc_data.grid)

        rho = q[:, ivars.irho]
        u = q[:, ivars.iu]
        v = q[:, ivars.iv]
        p = q[:, ivars.ip]
        e = eos.rhoe(gamma, p)/rho

        magvel = np.sqrt(u**2 + v**2)

        myg = self.cc_data.grid

        fields = [rho, magvel, p, e]
        field_names = [r"$\rho$", r"U", "p", "e"]

        _, axes, cbar_title = plot_tools.setup_axes(myg, len(fields))

        for n, ax in enumerate(axes):
            v = fields[n]

            img = ax.imshow(np.transpose(v.v()),
                            interpolation="nearest", origin="lower",
                            extent=[myg.xmin, myg.xmax, myg.ymin, myg.ymax],
                            cmap=self.cm)

            ax.set_xlabel("x")
            ax.set_ylabel("y")

            # needed for PDF rendering
            cb = axes.cbar_axes[n].colorbar(img)
            cb.solids.set_rasterized(True)
            cb.solids.set_edgecolor("face")

            if cbar_title:
                cb.ax.set_title(field_names[n])
            else:
                ax.set_title(field_names[n])

        if self.particles is not None:
            ax = axes[0]
            particle_positions = self.particles.get_positions()
            # dye particles
            colors = self.particles.get_init_positions()[:, 0]

            # plot particles
            ax.scatter(particle_positions[:, 0],
                particle_positions[:, 1], s=5, c=colors, alpha=0.8, cmap="Greys")
            ax.set_xlim([myg.xmin, myg.xmax])
            ax.set_ylim([myg.ymin, myg.ymax])

        plt.figtext(0.05, 0.0125, "t = {:10.5g}".format(self.cc_data.t))

        plt.pause(0.001)
        plt.draw()

    def write_extras(self, f):
        """
        Output simulation-specific data to the h5py file f
        """

        # make note of the custom BC
        gb = f.create_group("BC")

        # the value here is the value of "is_solid"
        gb.create_dataset("hse", data=False)
