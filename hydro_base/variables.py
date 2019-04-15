import numpy as np


class ConservedVector1D(np.ndarray):

    def __new__(cls, shape=()):

        return np.zeros((3, *shape), dtype=float).view(cls)

    def __array_finalize__(self, obj):
        """Required for deriving from `ndarray`."""
        if obj is None:
            return

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
    pass


class SourceVector1D(ConservedVector1D):
    pass


class PrimitiveVector1D(np.ndarray):

    def __new__(cls, shape=()):

        return np.zeros((3, *shape), dtype=float).view(cls)

    def __array_finalize__(self, obj):
        """Required for deriving from `ndarray`."""
        if obj is None:
            return

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

    def __new__(cls, shape=()):

        return np.zeros((3, *shape), dtype=float).view(cls)

    def __array_finalize__(self, obj):
        """Required for deriving from `ndarray`."""
        if obj is None:
            return

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
