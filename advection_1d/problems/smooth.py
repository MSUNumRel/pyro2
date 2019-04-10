from __future__ import print_function

import sys
import mesh.patch as patch
import numpy
from util import msg


def init_data(my_data, rp):
    """ initialize the smooth advection problem """

    msg.bold("initializing the smooth advection 1d problem...")

    # make sure that we are passed a valid patch object
    if not isinstance(my_data, patch.CellCenterData1d):
        print("ERROR: patch invalid in smooth.py")
        print(my_data.__class__)
        sys.exit()

    dens = my_data.get_var("density")

    xmin = my_data.grid.xmin
    xmax = my_data.grid.xmax

    xctr = 0.5*(xmin + xmax)

    dens[:] = 1.0 + numpy.exp(-60.0*((my_data.grid.x-xctr)**2))


def finalize():
    """ print out any information to the user at the end of the run """
    pass
