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


class IsentropicEOS(abc.ABC):

    @abc.abstractmethod
    def from_pressure(self, P):
        pass

    @abc.abstractmethod
    def from_density(self, rho0):
        pass

    @abc.abstractmethod
    def from_energy(self, eps):
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


class PolytropicEOS(IsentropicEOS):

    def __init__(self, K, gamma):
        self.K = K
        self.gamma = gamma

        self.gamma_m1 = gamma - 1.0
        self.inv_gamma = 1.0/gamma
        self.inv_gamma_m1 = 1.0/(gamma - 1.0)
        self.inv_K = 1.0/K

    def from_density(self, rho0):
        P = self.K*rho0**self.gamma
        eps = self.inv_gamma_m1*P/rho0

        return P, eps

    def from_pressure(self, P):
        rho0 = (P*self.inv_K)**self.inv_gamma
        eps = self.inv_gamma_m1*P/rho0

        return rho0, eps

    def from_energy(self, eps):
        rho0 = (eps*self.gamma_m1*self.inv_K)**self.inv_gamma_m1
        P = self.K*rho0**self.gamma

        return rho0, P
