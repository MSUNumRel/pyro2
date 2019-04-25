import numpy as np

import mesh.array_indexer as ai


class CustomGrid1D(object):

    def __init__(self, nzones, ncenter, dr_center, dr, r_const, r_max, ng=2):

        # Build the star in three zones:
        #   1. Central Zone - starts large and decreases on log scale
        #   2. Constant Zone - constant cell size to better resolve the stars
        #      surface
        #   3. Outer Zone - Normal log scale outside of the star

        # Zone 1: Center
        xi_center = np.zeros((ncenter,), dtype=float)

        dr_factor = (dr_center/dr)**(1.0/float(ncenter))
        gi = ncenter

        for i in range(1, ncenter):
            cell_width = dr*dr_factor**gi
            xi_center[i] = xi_center[i-1] + cell_width
            gi -= 1

        # Zone 2: Constant
        nconst = int((r_const - xi_center[-1])/dr)
        xi_const = np.zeros((nconst,), dtype=float)
        xi_const[0] = xi_center[-1] + dr
        for i in range(1, nconst):
            xi_const[i] = xi_const[i - 1] + dr

        # Zone 3: Outer
        nouter = nzones - nconst - ncenter + ng
        print(nouter)
        xi_outer = np.logspace(np.log10(xi_const[-1]),
                               np.log10(r_max), nouter + 1)[1:]

        self.nx = nzones
        self.ng = ng

        self.qx = int(2*ng + nzones)

        # Setup cell coordinate arrays
        self.x_i = np.zeros((self.qx,), dtype=float)
        self.x_i[ng:] = np.hstack((xi_center, xi_const, xi_outer))

        self.x = np.zeros_like(self.x_i)
        self.x[ng:-1] = 0.5*(self.x_i[ng:-1] + self.x_i[ng+1:])
        self.x[-1] = 2.0*self.x_i[-1] - self.x[-2]

        # Fill inner ghost cells
        self.x[:ng] = -self.x[ng:2*ng][::-1]
        self.x_i[:ng] = -self.x_i[ng+1:2*ng+1][::-1]

        # Spacings
        self.dx = np.zeros_like(self.x)
        self.dx[:-1] = np.diff(self.x_i)
        self.dx[-1] = self.dx[-2]  # 2.0*(self.x[-1] - self.x_i[-1])

        # Volumes
        self.volume = np.zeros_like(self.x)
        self.volume[ng:-1] = \
            4.0/3.0*np.pi*(self.x_i[ng+1:]**3 - self.x_i[ng:-1]**3)
        self.volume[-1] = \
            4.0/3.0*np.pi*((self.x_i[-1] + self.dx[-1])**3 - self.x_i[-1]**3)

        self.volume[:ng] = self.volume[ng:2*ng][::-1]

        # domain extrema
        self.xmin = self.x_i[ng]
        self.xmax = r_max

        # compute the indices of the block interior (excluding guardcells)
        self.ilo = self.ng
        self.ihi = self.ng + self.nx-1

        # center of the grid (for convenience)
        self.ic = self.ilo + self.nx//2 - 1

    def scratch_array(self, nvar=1):
        """
        return a standard numpy array dimensioned to have the size
        and number of ghostcells as the parent grid
        """
        if nvar == 1:
            _tmp = np.zeros((self.qx,), dtype=np.float64)
        else:
            _tmp = np.zeros((self.qx, nvar), dtype=np.float64)
        return ai.ArrayIndexer1d(d=_tmp, grid=self)

    def __str__(self):
        """ print out some basic information about the grid object """
        return "1-d custom radial grid: nx = {}, ng = {}".format(
            self.nx, self.ng)

    def __eq__(self, other):
        """ are two grids equivalent? """
        result = (self.nx == other.nx and self.ng == other.ng and
                  self.xmin == other.xmin and self.xmax == other.xmax)

        return result
