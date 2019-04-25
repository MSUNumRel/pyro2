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
from gr_radial_solver.equations import Polytrope


def init_data(my_data, rp, eqns: Polytrope):
    """ initialize the sod problem """

    msg.bold("initializing the TOV stability problem...")

    # make sure that we are passed a valid patch object
    if not isinstance(my_data, patch.CellCenterData1d):
        print("ERROR: patch invalid in tov.py")
        print(my_data.__class__)
        sys.exit()

    ilo = my_data.grid.ilo
    ihi = my_data.grid.ihi
    # get the TOV stability parameters
    rho0_c = rp.get_param("tov.rho0_c")

    # Test something
    rho_test = 1.23e-2
    rho_test = units.convert(rho_test, units.REL.density, units.CGS.density)
    print("test rho: {:.3e}".format(rho_test))

    r = my_data.grid.x

    r_c = r[ilo]
    r_max = r[ihi]
    print(units.convert(r_c, units.REL.length, units.CGS.length))

    gamma = rp.get_param("eos.gamma")
    K = rp.get_param("eos.k")

    # Convert to relativistic units
    rho0_c = units.convert(rho0_c, units.CGS.density, units.REL.density)
    rho_end = units.convert(1e9, units.CGS.density, units.REL.density)
    # rho_end = 1e-10
    # r_c = units.convert(r_c, units.CGS.length, units.REL.length)
    # r_max = units.convert(r_max, units.CGS.length, units.REL.length)
    print(r_c, r_max)

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
    Phi = my_data.get_var("phi")

    D = my_data.get_var("D")
    S = my_data.get_var("S")
    tau = my_data.get_var("tau")

    # rho0[rho0 <= eqns.atm_rho] = eqns.atm_rho
    # v[rho0 <= eqns.atm_rho] = 0.0
    # _, eps[:] = eos.from_density(rho0)

    ridx = np.argmax(m == m.max())
    print(ridx)

    # Hacks
    ng = grid.ng
    ilo = grid.ilo
    ihi = grid.ihi
    for i in range(ilo, ihi+1):
        if rho0[i] <= eqns.atm_rho:
            rho0[i] = eqns.atm_rho
            eps[i] = eqns.eos.energy(rho0[i])
            v[i] = 0.0
            # V.specific_energy[i] = eqns.atm_eps)
            D[i] = rho0[i]
            S[i] = 0.0
            tau[i] = rho0[i]*eps[i]

    # print(V.density)

    V = np.array((rho0, v, eps)).view(PrimitiveVector1D)
    U = ConservedVector1D(V.shape[1:])

    g = RGPSMetric(m, Phi, grid.x)

    eqns.prim2con(V[:, ilo:ihi+1], g.X[ilo:ihi+1], U[:, ilo:ihi+1])

    V2 = PrimitiveVector1D(V.shape[1:])
    V[:, :ng] = V[:, 2*ng-1:ng-1:-1]
    eqns.con2prim(U[:, ilo:ihi+1], g.X[ilo:ihi+1], rho0[ilo:ihi+1], V2[:, ilo:ihi+1])

    print(np.allclose(V[:, ilo:ihi+1], V2[:, ilo:ihi+1]))

    D[:] = U.density
    S[:] = 0.0  # U.momentum
    tau[:] = U.energy


def finalize():
    """ print out any information to the user at the end of the run """

    msg = """
          The script analysis/sod_compare.py can be used to compare
          this output to the exact solution.  Some sample exact solution
          data is present as analysis/sod-exact.out
          """

    print(msg)
