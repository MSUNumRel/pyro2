import numpy as np
import abc
from scipy.optimize import brentq

from hydro_base.variables import ConservedVector1D, PrimitiveVector1D, \
    FluxVector1D, SourceVector1D, CharacteristicVector1D

from hydro_base.equations import Equations1D

from hydro_base.eos import EOS


def lorentz_and_vel(u):
    W = np.sqrt(1.0 + u*u)
    v = u/W

    return W, v


class CompressibleSR1D(Equations1D):

    def __init__(self, eos: EOS):

        self.eos = eos

    def fluxes(self, U: ConservedVector1D, V: PrimitiveVector1D,
               F: FluxVector1D):
        D, S, E = U
        rho0, u, eps = V

        W, v = lorentz_and_vel(u)

        P = self.eos.pressure(rho0, eps)

        F.density = D*v
        F.momentum = S*v + P
        F.energy = (E + P)*v

    def sources(self, U: ConservedVector1D, V: PrimitiveVector1D,
                S: SourceVector1D):

        S[:] = 0.0

    def speeds(self, U: ConservedVector1D, V: PrimitiveVector1D,
               chars: CharacteristicVector1D):
        rho0, u, eps = V
        W, v = lorentz_and_vel(u)
        a = self.eos.sound_speed(rho0, eps)

        chars.minus = (v - a)/(1.0 - v*a)
        chars.center = v
        chars.plus = (v + a)/(1.0 + v*a)

    def prim2con(self, V: PrimitiveVector1D, U: ConservedVector1D):
        rho0, u, eps = V

        W, v = lorentz_and_vel(u)

        P = self.eos.pressure(rho0, eps)
        rho = self.eos.energy(rho0, eps)

        factor = (P + rho)*W*W
        U.density = rho0*W
        U.momentum = factor*v
        U.energy = factor - P

    def con2prim(self, U: ConservedVector1D, V: PrimitiveVector1D):

        def guess(P, D, S, E):
            try:
                v = S/(E + P)
                W = 1.0/np.sqrt(1.0 - v*v)
                u = W*v
                rho0 = D/W
                eps = (E - D*W + P*(1.0 - W*W))/(D*W)
            except ValueError:
                print((P, D, S, E))

            return rho0, u, eps

        def root_func(P, D, S, E):
            rho0, u, eps = guess(P, D, S, E)

            return self.eos.pressure(rho0, eps) - P

        D, S, E = U

        N = D.shape[0]

        for i in range(N):
            rho0_max = D[i]
            eps_max = E[i]/D[i]

            P_min = 1e-6
            P_max = self.eos.pressure(rho0_max, eps_max)

            P = brentq(root_func, P_min, P_max, args=(D[i], S[i], E[i]))
            rho0, u, eps = guess(P, D[i], S[i], E[i])

            V.density[i] = rho0
            V.velocity[i] = u
            V.specific_energy[i] = eps
