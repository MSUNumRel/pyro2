import numpy as np
import abc

from hydro_base.variables import ConservedVector1D, PrimitiveVector1D, \
    FluxVector1D, SourceVector1D, CharacteristicVector1D


class Equations1D(abc.ABC):

    @abc.abstractmethod
    def fluxes(self, U: ConservedVector1D, V: PrimitiveVector1D,
               F: FluxVector1D):
        pass

    @abc.abstractmethod
    def sources(self, U: ConservedVector1D, V: PrimitiveVector1D,
                S: SourceVector1D):
        pass

    @abc.abstractmethod
    def speeds(self, U: ConservedVector1D, V: PrimitiveVector1D,
               chars: CharacteristicVector1D):
        pass

    @abc.abstractmethod
    def prim2con(self, V: PrimitiveVector1D, U: ConservedVector1D):
        pass

    @abc.abstractmethod
    def con2prim(self, U: ConservedVector1D, V: PrimitiveVector1D):
        pass
