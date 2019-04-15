import abc

from mesh import patch

_default_var_names = {
    "density": "rho0",
    "velocity": "u",
    "specific energy": "eps",
    "mass": "m",
    "potential": "phi",
}


class InitialData1D(abc.ABC):

    def __init__(self, grid: patch.Grid1d):
        self.grid = grid

    @abc.abstractmethod
    def fill_patch(self, cc_patch: patch.CellCenterData1d,
                   var_names=_default_var_names):
        pass
