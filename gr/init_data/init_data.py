"""
    init_data.py

    Base class for initial data generation.
"""
import abc

from mesh import patch

# Default variable names assumed for CellCenterData1d patches
_default_var_names = {
    "density": "rho0",
    "velocity": "v",
    "specific energy": "eps",
    "mass": "m",
    "potential": "phi",
}


class InitialData1D(abc.ABC):
    """Base class for initial data generation."""

    def __init__(self, grid: patch.Grid1d):
        """Initialize the initial data structure.

        Parameters
        ----------
        grid : patch.Grid1d
            The grid this data lives on.
        """
        self.grid = grid

    @abc.abstractmethod
    def fill_patch(self, cc_patch: patch.CellCenterData1d,
                   var_names=_default_var_names):
        """Fill a CellCenterData1d patch with the initial data.

        Parameters
        ----------
        cc_patch : patch.CellCenterData1d
            The data patch to fill
        var_names : dict, optional
            Which variables to fill; see `_default_var_names` above for an 
            example usage.
        """
        pass
