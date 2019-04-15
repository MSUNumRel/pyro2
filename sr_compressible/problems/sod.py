import sys
import numpy as np

import mesh.patch as patch
from util import msg

from hydro_base.equations import Equations1D
from hydro_base.variables import ConservedVector1D, PrimitiveVector1D


def init_data(my_data, rp, eqns: Equations1D):
    """ initialize the sod problem """

    msg.bold("initializing the sod problem...")

    # make sure that we are passed a valid patch object
    if not isinstance(my_data, patch.CellCenterData1d):
        print("ERROR: patch invalid in sod.py")
        print(my_data.__class__)
        sys.exit()

    # get the sod parameters
    dens_left = rp.get_param("sod.dens_left")
    dens_right = rp.get_param("sod.dens_right")

    u_left = rp.get_param("sod.u_left")
    u_right = rp.get_param("sod.u_right")

    eps_left = rp.get_param("sod.eps_left")
    eps_right = rp.get_param("sod.eps_right")

    # get the density, momenta, and energy as separate variables
    rho0 = my_data.get_var("rho0")
    u = my_data.get_var("u")
    eps = my_data.get_var("eps")

    # initialize the components
    xmin = rp.get_param("mesh.xmin")
    xmax = rp.get_param("mesh.xmax")

    xctr = 0.5*(xmin + xmax)

    myg = my_data.grid

    # left
    idxl = myg.x <= xctr
    rho0[idxl] = dens_left
    u[idxl] = u_left
    eps[idxl] = eps_left
    # right
    idxr = myg.x > xctr
    rho0[idxr] = dens_right
    u[idxr] = u_right
    eps[idxr] = eps_right

    V = np.array((rho0, u, eps)).view(PrimitiveVector1D)
    U = ConservedVector1D(V.shape[1:])
    eqns.prim2con(V, U)

    D = my_data.get_var("D")
    S = my_data.get_var("S")
    E = my_data.get_var("E")

    D[:] = U.density
    S[:] = U.momentum
    E[:] = U.energy


def finalize():
    """ print out any information to the user at the end of the run """

    msg = """
          The script analysis/sod_compare.py can be used to compare
          this output to the exact solution.  Some sample exact solution
          data is present as analysis/sod-exact.out
          """

    print(msg)
