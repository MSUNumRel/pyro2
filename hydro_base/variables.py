"""
    variables.py

    This script containts `numpy.ndarray` subclasses describing the variables 
    and vector quantities used in solving hydrodynamic equations.  These 
    vectors can either be initialized to a specific grid size and later set to 
    the corresponding variable quantities, or "cast" from an exisiting 
    `numpy.ndarray`-type to the below types via the `view(new_type)` method.
"""
import numpy as np


class ConservedVector1D(np.ndarray):
    """Vector describing the 1D conserved hydro variables."""

    def __new__(cls, shape=()):
        """Create a `ndarray` view based on this vector type.

        Parameters
        ----------
        shape : tuple, optional
            The shape of the grid these variables live on (the default is (), which will create a single vector of the conserved variables)

        Returns
        -------
        ConservedVector1D
            Array of conserved variables
        """
        return np.zeros((3, *shape), dtype=float).view(cls)

    def __array_finalize__(self, obj):
        """Required for deriving from `ndarray`."""
        if obj is None:
            return

    ### Named properties to access the variables stored in this class ###
    @property
    def density(self):
        return self[0, ...]

    @density.setter
    def density(self, val):
        self[0, ...] = val

    @property
    def momentum(self):
        return self[1, ...]

    @momentum.setter
    def momentum(self, val):
        self[1, ...] = val

    @property
    def energy(self):
        return self[2, ...]

    @energy.setter
    def energy(self, val):
        self[2, ...] = val


class FluxVector1D(ConservedVector1D):
    """Vector describing the 1D hydro fluxes."""
    pass


class SourceVector1D(ConservedVector1D):
    """Vector describing the 1D hydro sources."""
    pass


class PrimitiveVector1D(np.ndarray):
    """Vector describing the 1D primitive hydro variables."""

    def __new__(cls, shape=()):
        """Create a `ndarray` view based on this vector type.

        Parameters
        ----------
        shape : tuple, optional
            The shape of the grid these variables live on (the default is (), which will create a single vector of the primitive variables)

        Returns
        -------
        PrimitiveVector1D
            Array of primitive variables
        """
        return np.zeros((3, *shape), dtype=float).view(cls)

    def __array_finalize__(self, obj):
        """Required for deriving from `ndarray`."""
        if obj is None:
            return

    ### Named properties to access the variables stored in this class ###

    @property
    def density(self):
        return self[0, ...]

    @density.setter
    def density(self, val):
        self[0, ...] = val

    @property
    def velocity(self):
        return self[1, ...]

    @velocity.setter
    def velocity(self, val):
        self[1, ...] = val

    @property
    def specific_energy(self):
        return self[2, ...]

    @specific_energy.setter
    def specific_energy(self, val):
        self[2, ...] = val


class CharacteristicVector1D(np.ndarray):
    """Vector describing the 1D hydro characteristic speeds."""

    def __new__(cls, shape=()):
        """Create a `ndarray` view based on this vector type.

        Parameters
        ----------
        shape : tuple, optional
            The shape of the grid these variables live on (the default is (), which will create a single vector of the characteristic speeds)

        Returns
        -------
        CharacteristicVector1D
            Array of characteristic speeds
        """
        return np.zeros((3, *shape), dtype=float).view(cls)

    def __array_finalize__(self, obj):
        """Required for deriving from `ndarray`."""
        if obj is None:
            return

    ### Named properties to access the variables stored in this class ###

    @property
    def plus(self):
        return self[2, ...]

    @plus.setter
    def plus(self, val):
        self[2, ...] = val

    @property
    def center(self):
        return self[1, ...]

    @center.setter
    def center(self, val):
        self[1, ...] = val

    @property
    def minus(self):
        return self[0, ...]

    @minus.setter
    def minus(self, val):
        self[0, ...] = val
