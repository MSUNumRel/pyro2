"""
    equations.py

    Interface for a system of hydrodynamic equations.
"""
import numpy as np
import abc

from hydro_base.variables import ConservedVector1D, PrimitiveVector1D, \
    FluxVector1D, SourceVector1D, CharacteristicVector1D


class Equations1D(abc.ABC):
    """Base class for a system of hydrodynamic equations"""

    @abc.abstractmethod
    def fluxes(self, U: ConservedVector1D, V: PrimitiveVector1D,
               F: FluxVector1D):
        """Calculate the fluxes

        Parameters
        ----------
        U : ConservedVector1D
            Conserved variables
        V : PrimitiveVector1D
            Primitive variables
        F : FluxVector1D
            [out] Flux result vector
        """
        pass

    @abc.abstractmethod
    def sources(self, U: ConservedVector1D, V: PrimitiveVector1D,
                S: SourceVector1D):
        """Calculate the source terms

        Parameters
        ----------
        U : ConservedVector1D
            Conserved variables
        V : PrimitiveVector1D
            Primitive variables
        S : SourceVector1D
            [out] Source result vector
        """
        pass

    @abc.abstractmethod
    def speeds(self, U: ConservedVector1D, V: PrimitiveVector1D,
               chars: CharacteristicVector1D):
        """Calculate the characteristic speeds

        Parameters
        ----------
        U : ConservedVector1D
            Conserved variables
        V : PrimitiveVector1D
            Primitive variables
        chars : CharacteristicVector1D
            [out] Characteristic speeds result vector
        """
        pass

    @abc.abstractmethod
    def prim2con(self, V: PrimitiveVector1D, U: ConservedVector1D):
        """Convert the primitive variables to conserved quantities

        Parameters
        ----------
        V : PrimitiveVector1D
            Primitive variables
        U : ConservedVector1D
            [out] Conserved variables
        """
        pass

    @abc.abstractmethod
    def con2prim(self, U: ConservedVector1D, V: PrimitiveVector1D):
        """Convert the conserved variables to primitives

        Parameters
        ----------
        U : ConservedVector1D
            Conserved variables
        V : PrimitiveVector1D
            [out] Primitive variables
        """
        pass
