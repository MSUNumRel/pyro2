import numpy as np
import abc


class EOS(abc.ABC):

    @abc.abstractmethod
    def pressure(self, rho0, eps):
        pass

    @abc.abstractmethod
    def energy(self, rho0, eps):
        pass

    @abc.abstractmethod
    def sound_speed(self, rho0, eps):
        pass


class GammaLawEOS(EOS):

    def __init__(self, gamma):

        self.gamma = gamma
        self.gamma_m1 = gamma - 1.0
        self.inv_gamma = 1.0/gamma
        self.inv_gamma_m1 = 1.0/self.inv_gamma

    def pressure(self, rho0, eps):
        return self.gamma_m1*rho0*eps

    def energy(self, rho0, eps):
        return rho0*(1.0 + eps)

    def sound_speed(self, rho0, eps):
        return np.sqrt((self.gamma_m1*self.gamma*eps)/(self.gamma*eps + 1.0))
