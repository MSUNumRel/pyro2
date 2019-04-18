import sys
import numpy as np

import mesh.patch as patch
from util import msg

from gr.hydro.equations import RPGSEquations
from gr.metric import RGPSMetric
from gr import units
from gr.init_data import TOVInitialData

from hydro_base.variables import ConservedVector1D, PrimitiveVector1D

from hydro_base.eos import PolytropicEOS
from gr_radial_solver.polytrope import Polytrope


def init_data(my_data, rp, eqns: Polytrope):
    """ initialize the sod problem """

    msg.bold("initializing the TOV stability problem...")

    # make sure that we are passed a valid patch object
    if not isinstance(my_data, patch.CellCenterData1d):
        print("ERROR: patch invalid in tov.py")
        print(my_data.__class__)
        sys.exit()

    # get the TOV stability parameters
    rho0_c = rp.get_param("tov.rho0_c")

    r = my_data.grid.x

    r_c = r[0]
    r_max = r[-1]
    print(units.convert(r_c, units.REL.length, units.CGS.length))

    gamma = rp.get_param("eos.gamma")
    K = rp.get_param("eos.k")

    # Convert to relativistic units
    # rho0_c = units.convert(rho0_c, units.CGS.density, units.REL.density)
    rho_end = units.convert(1e12, units.CGS.density, units.REL.density)
    # rho_end = 1e-10
    r_c = units.convert(r_c, units.CGS.length, units.REL.length)
    r_max = units.convert(r_max, units.CGS.length, units.REL.length)

    print(rho0_c, rho_end)

    eos = PolytropicEOS(K, gamma)

    grid = my_data.grid

    initial = TOVInitialData(grid)

    initial.initialize(eos)

    initial.solve_to_grid(rho0_c, rho_end)
    print("Obtained TOV solution")

    initial.fill_patch(my_data)

    # get the density, momenta, and energy as separate variables
    rho0 = my_data.get_var("rho0")
    v = my_data.get_var("v")
    eps = my_data.get_var("eps")
    m = my_data.get_var("m")
    Phi = my_data.get_var("Phi")

    rho0[rho0 <= eqns.atm_rho] = eqns.atm_rho
    _, eps[:] = eos.from_density(rho0)

    g = RGPSMetric(m, Phi, grid.x)

    V = np.array((rho0, v, eps)).view(PrimitiveVector1D)
    U = ConservedVector1D(V.shape[1:])
    eqns.prim2con(V, g, U)

    D = my_data.get_var("D")
    S = my_data.get_var("S")
    tau = my_data.get_var("tau")

    D[:] = U.density
    S[:] = U.momentum
    tau[:] = U.energy


def finalize():
    """ print out any information to the user at the end of the run """

    msg = """
          The script analysis/sod_compare.py can be used to compare
          this output to the exact solution.  Some sample exact solution
          data is present as analysis/sod-exact.out
          """

    print(msg)
