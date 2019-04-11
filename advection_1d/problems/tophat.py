from __future__ import print_function

import sys
import mesh.patch as patch
from util import msg


def init_data(myd, rp):
    """ initialize the tophat advection problem """

    msg.bold("initializing the tophat advection 1d problem...")

    # make sure that we are passed a valid patch object
    if not isinstance(myd, patch.CellCenterData1d):
        print("ERROR: patch invalid in tophat.py")
        print(myd.__class__)
        sys.exit()

    dens = myd.get_var("density")

    xmin = myd.grid.xmin
    xmax = myd.grid.xmax

    xctr = 0.5*(xmin + xmax)

    dens[:] = 0.0

    R = 0.1

    inside = (myd.grid.x - xctr)**2 < R**2

    dens[inside] = 1.0


def finalize():
    """ print out any information to the user at the end of the run """
    pass
