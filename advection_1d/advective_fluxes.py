import mesh.reconstruction as reconstruction


def unsplit_fluxes(my_data, rp, dt, scalar_name):
    r"""
    Construct the fluxes through the interfaces for the linear advection
    equation:

    .. math::

       a_t  + u a_x  = 0

    We use a second-order (piecewise linear) unsplit Godunov method
    (following Colella 1990).

    In the pure advection case, there is no Riemann problem we need to
    solve -- we just simply do upwinding.  So there is only one 'state'
    at each interface, and the zone the information comes from depends
    on the sign of the velocity.

    Our convection is that the fluxes are going to be defined on the
    left edge of the computational zones::

        |             |             |             |
        |             |             |             |
       -+------+------+------+------+------+------+--
        |     i-1     |      i      |     i+1     |

                 a_l,i  a_r,i   a_l,i+1


    a_r,i and a_l,i+1 are computed using the information in
    zone i.

    Parameters
    ----------
    my_data : CellCenterData2d object
        The data object containing the grid and advective scalar that
        we are advecting.
    rp : RuntimeParameters object
        The runtime parameters for the simulation
    dt : float
        The timestep we are advancing through.
    scalar_name : str
        The name of the variable contained in my_data that we are
        advecting

    Returns
    -------
    out : ndarray
        The fluxes on the x-interfaces

    """

    myg = my_data.grid

    a = my_data.get_var(scalar_name)

    # get the advection velocities
    u = rp.get_param("advection.u")

    cx = u*dt/myg.dx

    # --------------------------------------------------------------------------
    # monotonized central differences
    # --------------------------------------------------------------------------

    limiter = rp.get_param("advection.limiter")

    ldelta_ax = reconstruction.limit_1d(a, myg, limiter)

    a_x = myg.scratch_array()

    # upwind
    if u < 0:
        # a_x[i,j] = a[i,j] - 0.5*(1.0 + cx)*ldelta_a[i,j]
        a_x.v(buf=1)[:] = a.v(buf=1) - 0.5*(1.0 + cx)*ldelta_ax.v(buf=1)
    else:
        # a_x[i,j] = a[i-1,j] + 0.5*(1.0 - cx)*ldelta_a[i-1,j]
        a_x.v(buf=1)[:] = a.ip(-1, buf=1) + 0.5*(1.0 - cx)*ldelta_ax.ip(-1, buf=1)

    F_x = myg.scratch_array()

    # the zone where we grab the transverse flux derivative from
    # depends on the sign of the advective velocity

    F_x.v(buf=1)[:] = u*(a_x.v(buf=1))

    return F_x
