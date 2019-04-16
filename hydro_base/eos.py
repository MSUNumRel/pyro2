import numpy as np
import abc


class EOS(abc.ABC):
    """Base class for equations of state."""

    @abc.abstractmethod
    def pressure(self, rho0, eps):
        """Calculate the pressure

        Parameters
        ----------
        rho0 : float(s)
            Density
        eps : float(s)
            Specific internal energy

        Returns
        -------
        P : float
            The pressure
        """
        pass

    @abc.abstractmethod
    def energy(self, rho0, eps):
        """Calculate the energy density

        Parameters
        ----------
        rho0 : float(s)
            Density
        eps : float(s)
            Specific internal energy

        Returns
        -------
        E : float
            The energy density
        """
        pass

    @abc.abstractmethod
    def sound_speed(self, rho0, eps):
        """Calculate the sound speed

        Parameters
        ----------
        rho0 : float(s)
            Density
        eps : float(s)
            Specific internal energy

        Returns
        -------
        a : float
            The sound speed
        """
        pass


class IsentropicEOS(abc.ABC):
    """Base class for an isentropic equation of state."""

    @abc.abstractmethod
    def from_pressure(self, P):
        """Evaluate the EOS at a given pressure

        Parameters
        ----------
        P : float(s)
            Pressure

        Returns
        -------
        tuple of float(s)
            (density, specific energy)
        """
        pass

    @abc.abstractmethod
    def from_density(self, rho0):
        """Evaluate the EOS at a given density

        Parameters
        ----------
        rho0 : float(s)
            Density

        Returns
        -------
        tuple of float(s)
            (pressure, specific energy)        
        """
        pass

    @abc.abstractmethod
    def from_energy(self, eps):
        """Evaluate the EOS at a given specific energy

        Parameters
        ----------
        eps : float(s)
            Specific energy

        Returns
        -------
        tuple of float(s)
            (density, pressure)        
        """
        pass


class GammaLawEOS(EOS):
    """A Gamma-Law equation of state."""

    def __init__(self, gamma):
        """Initialize the EOS

        Parameters
        ----------
        gamma : float
            Adiabatic index for the gas
        """
        self.gamma = gamma

        # Repeatedly used in the EOS
        self.gamma_m1 = gamma - 1.0
        self.inv_gamma = 1.0/gamma
        self.inv_gamma_m1 = 1.0/self.inv_gamma

    def pressure(self, rho0, eps):
        """Calculate the pressure

        Parameters
        ----------
        rho0 : float(s)
            Density
        eps : float(s)
            Specific internal energy

        Returns
        -------
        P : float
            The pressure
        """
        return self.gamma_m1*rho0*eps

    def energy(self, rho0, eps):
        """Calculate the energy density

        Parameters
        ----------
        rho0 : float(s)
            Density
        eps : float(s)
            Specific internal energy

        Returns
        -------
        E : float
            The energy density
        """
        return rho0*(1.0 + eps)

    def sound_speed(self, rho0, eps):
        """Calculate the sound speed

        Parameters
        ----------
        rho0 : float(s)
            Density
        eps : float(s)
            Specific internal energy

        Returns
        -------
        a : float
            The sound speed
        """
        return np.sqrt((self.gamma_m1*self.gamma*eps)/(self.gamma*eps + 1.0))


class PolytropicEOS(IsentropicEOS):
    """A polytropic equation of state."""

    def __init__(self, K, gamma):
        """Initialize the polytropic EOS

        Parameters
        ----------
        K : float
            Polytropic constant
        gamma : float
            Adiabatic index
        """

        self.K = K
        self.gamma = gamma

        self.gamma_m1 = gamma - 1.0
        self.inv_gamma = 1.0/gamma
        self.inv_gamma_m1 = 1.0/(gamma - 1.0)
        self.inv_K = 1.0/K

    def from_density(self, rho0):
        """Evaluate the EOS at a given density

        Parameters
        ----------
        rho0 : float(s)
            Density

        Returns
        -------
        tuple of float(s)
            (pressure, specific energy)        
        """
        P = self.K*rho0**self.gamma
        eps = self.inv_gamma_m1*P/rho0

        return P, eps

    def from_pressure(self, P):
        """Evaluate the EOS at a given pressure

        Parameters
        ----------
        P : float(s)
            Pressure

        Returns
        -------
        tuple of float(s)
            (density, specific energy)
        """
        rho0 = (P*self.inv_K)**self.inv_gamma
        eps = self.inv_gamma_m1*P/rho0

        return rho0, eps

    def from_energy(self, eps):
        """Evaluate the EOS at a given specific energy

        Parameters
        ----------
        eps : float(s)
            Specific energy

        Returns
        -------
        tuple of float(s)
            (density, pressure)        
        """
        rho0 = (eps*self.gamma_m1*self.inv_K)**self.inv_gamma_m1
        P = self.K*rho0**self.gamma

        return rho0, P
