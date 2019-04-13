import numpy as np
from numba import njit


@njit(cache=True)
def states(idir, ng, dx, dt,
           irho, iu, iv, ip, ix, nspec,
           gamma, qv, dqv):
    r"""
    predict the cell-centered state to the edges in one-dimension
    using the reconstructed, limited slopes.

    We follow the convection here that ``V_l[i]`` is the left state at the
    i-1/2 interface and ``V_l[i+1]`` is the left state at the i+1/2
    interface.

    We need the left and right eigenvectors and the eigenvalues for
    the system projected along the x-direction.

    Taking our state vector as :math:`Q = (\rho, u, v, p)^T`, the eigenvalues
    are :math:`u - c`, :math:`u`, :math:`u + c`.

    We look at the equations of hydrodynamics in a split fashion --
    i.e., we only consider one dimension at a time.

    Considering advection in the x-direction, the Jacobian matrix for
    the primitive variable formulation of the Euler equations
    projected in the x-direction is::

             / u   r   0   0 \
             | 0   u   0  1/r |
         A = | 0   0   u   0  |
             \ 0  rc^2 0   u  /

    The right eigenvectors are::

             /  1  \        / 1 \        / 0 \        /  1  \
             |-c/r |        | 0 |        | 0 |        | c/r |
        r1 = |  0  |   r2 = | 0 |   r3 = | 1 |   r4 = |  0  |
             \ c^2 /        \ 0 /        \ 0 /        \ c^2 /

    In particular, we see from r3 that the transverse velocity (v in
    this case) is simply advected at a speed u in the x-direction.

    The left eigenvectors are::

         l1 =     ( 0,  -r/(2c),  0, 1/(2c^2) )
         l2 =     ( 1,     0,     0,  -1/c^2  )
         l3 =     ( 0,     0,     1,     0    )
         l4 =     ( 0,   r/(2c),  0, 1/(2c^2) )

    The fluxes are going to be defined on the left edge of the
    computational zones::

            |             |             |             |
            |             |             |             |
           -+------+------+------+------+------+------+--
            |     i-1     |      i      |     i+1     |
                         ^ ^           ^
                     q_l,i q_r,i  q_l,i+1

    q_r,i and q_l,i+1 are computed using the information in zone i,j.

    Parameters
    ----------
    idir : int
        Are we predicting to the edges in the x-direction (1) or y-direction (2)?
    ng : int
        The number of ghost cells
    dx : float
        The cell spacing
    dt : float
        The timestep
    irho, iu, iv, ip, ix : int
        Indices of the density, x-velocity, y-velocity, pressure and species in the
        state vector
    nspec : int
        The number of species
    gamma : float
        Adiabatic index
    qv : ndarray
        The primitive state vector
    dqv : ndarray
        Spatial derivitive of the state vector

    Returns
    -------
    out : ndarray, ndarray
        State vector predicted to the left and right edges
    """

    qx, qy, nvar = qv.shape

    q_l = np.zeros_like(qv)
    q_r = np.zeros_like(qv)

    nx = qx - 2 * ng
    ny = qy - 2 * ng
    ilo = ng
    ihi = ng + nx
    jlo = ng
    jhi = ng + ny

    ns = nvar - nspec

    dtdx = dt / dx
    dtdx4 = 0.25 * dtdx

    lvec = np.zeros((nvar, nvar))
    rvec = np.zeros((nvar, nvar))
    e_val = np.zeros(nvar)
    betal = np.zeros(nvar)
    betar = np.zeros(nvar)

    # this is the loop over zones.  For zone i, we see q_l[i+1] and q_r[i]
    for i in range(ilo - 2, ihi + 2):
        for j in range(jlo - 2, jhi + 2):

            dq = dqv[i, j, :]
            q = qv[i, j, :]

            cs = np.sqrt(gamma * q[ip] / q[irho])

            lvec[:, :] = 0.0
            rvec[:, :] = 0.0
            e_val[:] = 0.0

            # compute the eigenvalues and eigenvectors
            if (idir == 1):
                e_val[:] = np.array([q[iu] - cs, q[iu], q[iu], q[iu] + cs])

                lvec[0, :ns] = [0.0, -0.5 *
                                 q[irho] / cs, 0.0, 0.5 / (cs * cs)]
                lvec[1, :ns] = [1.0, 0.0,
                                 0.0, -1.0 / (cs * cs)]
                lvec[2, :ns] = [0.0, 0.0,             1.0, 0.0]
                lvec[3, :ns] = [0.0, 0.5 *
                                 q[irho] / cs,  0.0, 0.5 / (cs * cs)]

                rvec[0, :ns] = [1.0, -cs / q[irho], 0.0, cs * cs]
                rvec[1, :ns] = [1.0, 0.0,       0.0, 0.0]
                rvec[2, :ns] = [0.0, 0.0,       1.0, 0.0]
                rvec[3, :ns] = [1.0, cs / q[irho],  0.0, cs * cs]

                # now the species -- they only have a 1 in their corresponding slot
                e_val[ns:] = q[iu]
                for n in range(ix, ix + nspec):
                    lvec[n, n] = 1.0
                    rvec[n, n] = 1.0

            else:
                e_val[:] = np.array([q[iv] - cs, q[iv], q[iv], q[iv] + cs])

                lvec[0, :ns] = [0.0, 0.0, -0.5 *
                                 q[irho] / cs, 0.5 / (cs * cs)]
                lvec[1, :ns] = [1.0, 0.0,
                                 0.0,             -1.0 / (cs * cs)]
                lvec[2, :ns] = [0.0, 1.0, 0.0,             0.0]
                lvec[3, :ns] = [0.0, 0.0, 0.5 *
                                 q[irho] / cs,  0.5 / (cs * cs)]

                rvec[0, :ns] = [1.0, 0.0, -cs / q[irho], cs * cs]
                rvec[1, :ns] = [1.0, 0.0, 0.0,       0.0]
                rvec[2, :ns] = [0.0, 1.0, 0.0,       0.0]
                rvec[3, :ns] = [1.0, 0.0, cs / q[irho],  cs * cs]

                # now the species -- they only have a 1 in their corresponding slot
                e_val[ns:] = q[iv]
                for n in range(ix, ix + nspec):
                    lvec[n, n] = 1.0
                    rvec[n, n] = 1.0

            # define the reference states
            if (idir == 1):
                # this is one the right face of the current zone,
                # so the fastest moving eigenvalue is e_val[3] = u + c
                factor = 0.5 * (1.0 - dtdx * max(e_val[3], 0.0))
                q_l[i + 1, j, :] = q + factor * dq

                # left face of the current zone, so the fastest moving
                # eigenvalue is e_val[3] = u - c
                factor = 0.5 * (1.0 + dtdx * min(e_val[0], 0.0))
                q_r[i,  j, :] = q - factor * dq

            else:

                factor = 0.5 * (1.0 - dtdx * max(e_val[3], 0.0))
                q_l[i, j + 1, :] = q + factor * dq

                factor = 0.5 * (1.0 + dtdx * min(e_val[0], 0.0))
                q_r[i, j, :] = q - factor * dq

            # compute the Vhat functions
            for m in range(nvar):
                sum = np.dot(lvec[m, :], dq)

                betal[m] = dtdx4 * (e_val[3] - e_val[m]) * \
                    (np.copysign(1.0, e_val[m]) + 1.0) * sum
                betar[m] = dtdx4 * (e_val[0] - e_val[m]) * \
                    (1.0 - np.copysign(1.0, e_val[m])) * sum

            # construct the states
            for m in range(nvar):
                sum_l = np.dot(betal, rvec[:, m])
                sum_r = np.dot(betar, rvec[:, m])

                if (idir == 1):
                    q_l[i + 1, j, m] = q_l[i + 1, j, m] + sum_l
                    q_r[i,  j, m] = q_r[i,  j, m] + sum_r
                else:
                    q_l[i, j + 1, m] = q_l[i, j + 1, m] + sum_l
                    q_r[i, j,  m] = q_r[i, j,  m] + sum_r

    return q_l, q_r


def gamma_plus_minus(un, ut, c_s):
    r"""
    a helper function for the general relativistice HLLE
    below
    
    it computes the wave plus and minus characteristics
    of at each interface using equation 2.249 of Rezzolla and Zanotti
    
    it takes
    gamma_plus_minus(un, ut, c_s)
    
    un is the normal velocity
    ut is the transverse velocity
    c_s is the sound speed
    
    it returns
    gamma_minus, gamma_plus
    
    the two characteristics
    """
    #the squared total velocity
    v_sq = un**2 + ut**2
    
    #two useful things to sto
    a = un*(1 - c_s**2)
    b = c_s*np.power((1 - v_sq)*(1 - un**2 - ut**2*c_s**2), 0.5)
    c = (1 - v_sq*c_s**2)

    gamma_minus = (a - b)/c
    gamma_plus = (a + b)/c

    return gamma_minus, gamma_plus


def riemann_hlle(idir, ng,
                 idens, ixmom, iymom, iener, irhoX, nspec,
                 lower_solid, upper_solid,
                 gamma, U_l, U_r):
    r"""
    This is the relativistic HLLE Riemann solver.  The implementation follows
    directly out of Toro's book.  Note: this is hard coded for a
    polytropic eos.
    
    Parameters
    ----------
    idir : int
        Are we predicting to the edges in the x-direction (1) or y-direction (2)?
    ng : int
        The number of ghost cells
    nspec : int
        The number of species
    idens, ixmom, iymom, iener, irhoX : int
        The indices of the density, x-momentum, y-momentum, internal energy density
        and species partial densities in the conserved state vector.
    lower_solid, upper_solid : int
        Are we at lower or upper solid boundaries?
    gamma : float
        Adiabatic index
    U_l, U_r : ndarray
        Conserved state on the left and right cell edges.

    Returns
    -------
    out : ndarray
        Conserved flux
    """

    qx, qy, nvar = U_l.shape

    #returned flux
    F = np.zeros((qx, qy, nvar))
    #for starred region
    F_r = np.zeros((qx, qy, nvar))
    F_l = np.zeros((qx, qy, nvar))

    U_state = np.zeros(nvar)

    nx = qx - 2 * ng
    ny = qy - 2 * ng
    ilo = ng
    ihi = ng + nx
    jlo = ng
    jhi = ng + ny

    for i in range(ilo - 1, ihi + 1):
        for j in range(jlo - 1, jhi + 1):

            # primitive variable states
            rho_l = U_l[i, j, idens]
            rho_r = U_r[i, j, idens]

            # compute the sound speeds
            # calculated from eq. 2.249
            # this assumes a polytrope eos
            c_l = (1/gamma/rho_l**(gamma - 1) + 1/(gamma - 1))**(-0.5)
            c_r = (1/gamma/rho_r**(gamma - 1) + 1/(gamma - 1))**(-0.5)

            # un = normal velocity; ut = transverse velocity
            if (idir == 1):
                un_l = U_l[i, j, ixmom] / rho_l
                ut_l = U_l[i, j, iymom] / rho_l
            else:
                un_l = U_l[i, j, iymom] / rho_l
                ut_l = U_l[i, j, ixmom] / rho_l

            if (idir == 1):
                un_r = U_r[i, j, ixmom] / rho_r
                ut_r = U_r[i, j, iymom] / rho_r
            else:
                un_r = U_r[i, j, iymom] / rho_r
                ut_r = U_r[i, j, ixmom] / rho_r

            #now calculate the eigenvalues
            gamma_l_m, gamma_l_p  = gamma_plus_minus(un_l, ut_l, c_l)
            gamma_r_m, gamma_r_p  = gamma_plus_minus(un_r, ut_r, c_r)
            
            
            S_l = np.min((0, gamma_l_m, gamma_r_m))
            S_r = np.min((0, gamma_l_p, gamma_r_p))

            # figure out which region we are in and compute the state and
            # the interface fluxes using the HLLC Riemann solver
            if (S_r <= 0.0):
                # R region
                U_state[:] = U_r[i, j, :]

                F[i, j, :] = consFlux(idir, gamma, idens, ixmom, iymom, iener, irhoX, nspec,
                                      U_state)
           
            elif (S_r > 0.0 and S_l < 0.0):
                # find the left starred flux
                F_l[i, j, :] = consFlux(idir, gamma, idens, ixmom, iymom, iener, irhoX, nspec,
                                      U_l[i, j, :])

                # find the right starred flux
                F_r[i, j, :] = consFlux(idir, gamma, idens, ixmom, iymom, iener, irhoX, nspec,
                                      U_r[i, j, :])

                # correct the flux
                F[i, j, :] = (S_r * F_l[i, j, :] - S_l * F_r[i, j, :] + \
                             S_l * S_r * (U_r[i, j, :] - U_l[i, j, :]))/(S_r - S_l)

                # * region
                U_state[:] = (S_r * U_r[i, j, :] - S_l * U_l[i, j, :] + \
                       F_l[i, j, :] - F_r[i, j, :])/(S_r - S_l)

            else:
                # L region
                U_state[:] = U_l[i, j, :]

                F[i, j, :] = consFlux(idir, gamma, idens, ixmom, iymom, iener, irhoX, nspec,
                                      U_state)

            # we should deal with solid boundaries somehow here
    
    return F


@njit(cache=True)
def consFlux(idir, gamma, idens, ixmom, iymom, iener, irhoX, nspec, U_state):
    r"""
    Calculate the conservative flux.

    Parameters
    ----------
    idir : int
        Are we predicting to the edges in the x-direction (1) or y-direction (2)?
    gamma : float
        Adiabatic index
    idens, ixmom, iymom, iener, irhoX : int
        The indices of the density, x-momentum, y-momentum, internal energy density
        and species partial densities in the conserved state vector.
    nspec : int
        The number of species
    U_state : ndarray
        Conserved state vector.

    Returns
    -------
    out : ndarray
        Conserved flux
    """

    F = np.zeros_like(U_state)

    u = U_state[ixmom] / U_state[idens]
    v = U_state[iymom] / U_state[idens]

    p = (U_state[iener] - 0.5 * U_state[idens] * (u * u + v * v)) * (gamma - 1.0)

    if (idir == 1):
        F[idens] = U_state[idens] * u
        F[ixmom] = U_state[ixmom] * u + p
        F[iymom] = U_state[iymom] * u
        F[iener] = (U_state[iener] + p) * u

        if (nspec > 0):
            F[irhoX:irhoX + nspec] = U_state[irhoX:irhoX + nspec] * u

    else:
        F[idens] = U_state[idens] * v
        F[ixmom] = U_state[ixmom] * v
        F[iymom] = U_state[iymom] * v + p
        F[iener] = (U_state[iener] + p) * v

        if (nspec > 0):
            F[irhoX:irhoX + nspec] = U_state[irhoX:irhoX + nspec] * v

    return F


@njit(cache=True)
def artificial_viscosity(ng, dx, dy,
                         cvisc, u, v):
    r"""
    Compute the artifical viscosity.  Here, we compute edge-centered
    approximations to the divergence of the velocity.  This follows
    directly Colella \ Woodward (1984) Eq. 4.5

    data locations::

        j+3/2--+---------+---------+---------+
               |         |         |         |
          j+1  +         |         |         |
               |         |         |         |
        j+1/2--+---------+---------+---------+
               |         |         |         |
             j +         X         |         |
               |         |         |         |
        j-1/2--+---------+----Y----+---------+
               |         |         |         |
           j-1 +         |         |         |
               |         |         |         |
        j-3/2--+---------+---------+---------+
               |    |    |    |    |    |    |
                   i-1        i        i+1
             i-3/2     i-1/2     i+1/2     i+3/2

    ``X`` is the location of ``avisco_x[i,j]``
    ``Y`` is the location of ``avisco_y[i,j]``

    Parameters
    ----------
    ng : int
        The number of ghost cells
    dx, dy : float
        Cell spacings
    cvisc : float
        viscosity parameter
    u, v : ndarray
        x- and y-velocities

    Returns
    -------
    out : ndarray, ndarray
        Artificial viscosity in the x- and y-directions
    """

    qx, qy = u.shape

    avisco_x = np.zeros((qx, qy))
    avisco_y = np.zeros((qx, qy))

    nx = qx - 2 * ng
    ny = qy - 2 * ng
    ilo = ng
    ihi = ng + nx
    jlo = ng
    jhi = ng + ny

    for i in range(ilo - 1, ihi + 1):
        for j in range(jlo - 1, jhi + 1):

            # start by computing the divergence on the x-interface.  The
            # x-difference is simply the difference of the cell-centered
            # x-velocities on either side of the x-interface.  For the
            # y-difference, first average the four cells to the node on
            # each end of the edge, and: difference these to find the
            # edge centered y difference.
            divU_x = (u[i, j] - u[i - 1, j]) / dx + \
                0.25 * (v[i, j + 1] + v[i - 1, j + 1] -
                        v[i, j - 1] - v[i - 1, j - 1]) / dy

            avisco_x[i, j] = cvisc * max(-divU_x * dx, 0.0)

            # now the y-interface value
            divU_y = 0.25 * (u[i + 1, j] + u[i + 1, j - 1] - u[i - 1, j] - u[i - 1, j - 1]) / dx + \
                (v[i, j] - v[i, j - 1]) / dy

            avisco_y[i, j] = cvisc * max(-divU_y * dy, 0.0)

    return avisco_x, avisco_y
