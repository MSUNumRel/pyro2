import numpy as np
import abc

from hydro_base.variables import ConservedVector1D, PrimitiveVector1D, \
    FluxVector1D, CharacteristicVector1D

from hydro_base.equations import Equations1D


class RiemannSolver1D(abc.ABC):

    @abc.abstractmethod
    def fluxes(self, U_l: ConservedVector1D, U_r: ConservedVector1D,
               V_l: PrimitiveVector1D, V_r: PrimitiveVector1D,
               F_l: FluxVector1D, F_r: FluxVector1D,
               char_l: CharacteristicVector1D, char_r: CharacteristicVector1D,
               F: FluxVector1D):
        pass


class HLLE1D(RiemannSolver1D):

    def __init__(self, equations: Equations1D):

        self.eqns = equations

    def fluxes(self, U_l: ConservedVector1D, U_r: ConservedVector1D,
               V_l: PrimitiveVector1D, V_r: PrimitiveVector1D,
               F_l: FluxVector1D, F_r: FluxVector1D,
               char_l: CharacteristicVector1D, char_r: CharacteristicVector1D,
               F: FluxVector1D):

        s_l = np.minimum(0.0, np.minimum(char_l.minus, char_r.minus))
        s_r = np.maximum(0.0, np.minimum(char_l.plus, char_r.plus))

        F[:] = (s_r*F_l - s_l*F_r + s_r*s_l*(U_r - U_l))/(s_r - s_l)


class Rusanov1D(RiemannSolver1D):

    def __init__(self, equations: Equations1D):

        self.eqns = equations

    def fluxes(self, U_l: ConservedVector1D, U_r: ConservedVector1D,
               V_l: PrimitiveVector1D, V_r: PrimitiveVector1D,
               F_l: FluxVector1D, F_r: FluxVector1D,
               char_l: CharacteristicVector1D, char_r: CharacteristicVector1D,
               F: FluxVector1D):

        s_l = np.minimum(0.0, np.minimum(char_l.minus, char_r.minus))
        s_r = np.maximum(0.0, np.minimum(char_l.plus, char_r.plus))

        s = np.maximum(np.abs(s_l), np.abs(s_r))

        F[:] = 0.5*(F_l + F_r) - 0.5*s*(U_r - U_l)
