#
#  MicroHH
#  Copyright (c) 2011-2024 Chiel van Heerwaarden
#  Copyright (c) 2011-2024 Thijs Heus
#  Copyright (c) 2014-2024 Bart van Stratum
#
#  This file is part of MicroHH
#
#  MicroHH is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  MicroHH is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with MicroHH.  If not, see <http://www.gnu.org/licenses/>.
#

# Standard library

# Third-party.
import numpy as np
from scipy.fft import dct
from numba import jit

# Local library
from microhhpy.logger import logger


@jit(nopython=True, nogil=True, fastmath=True)
def adjust_boundary_mass_flux(u, v, w, rho, rhoh, dz, dx, dy, itot, jtot, ktot):
    """
    Adjust velocities at boundaries to make each horizontal slice globally divergence free.
    The input inbalance is distributed equally over the in- and outflow of both `u` and `v`, leaving `w` untouched.
    The pressure solver step next solves for each grid point individually.
    
    Parameters:
    ----------
    u : np.ndarray, shape (ktot, jtot, itot+1)
        Zonal wind field.
    v : np.ndarray, shape (ktot, jtot+1, itot)
        Meridional wind field.
    w : np.ndarray, shape (ktot+1, jtot, itot)
        Vertical velocity field.
    rho : np.ndarray, shape (ktot,)
        Base state density at full levels.
    rhoh : np.ndarray, shape (ktot+1,)
        Base state density at half levels.
    dz : np.ndarray, shape (ktot,)
        Vertical grid spacing.
    dx : float
        Horizontal grid spacing in x-direction.
    dy : float
        Horizontal grid spacing in y-direction.
    """
    
    xsize = itot * dx
    ysize = jtot * dy
    
    for k in range(ktot):
        # Calculate mass fluxes through boundaries.
        u_in = 0.
        u_out = 0.
        for j in range(jtot):
            u_in  += u[k, j, 0]  * rho[k] * dy * dz[k]
            u_out += u[k, j, -1] * rho[k] * dy * dz[k]
        
        v_in = 0.
        v_out = 0.
        for i in range(itot):
            v_in  += v[k, 0, i]  * rho[k] * dx * dz[k]
            v_out += v[k, -1, i] * rho[k] * dx * dz[k]
        
        w_in = 0.
        w_out = 0.
        for j in range(jtot):
            for i in range(itot):
                w_in  += w[k,   j, i] * rhoh[k]   * dx * dy
                w_out += w[k+1, j, i] * rhoh[k+1] * dx * dy
        
        # Net mass imbalance.
        net = u_in - u_out + v_in - v_out + w_in - w_out
        
        # Adjustments in- and outflow.
        du = 0.5 * net / (2 * rho[k] * dz[k] * ysize)
        dv = 0.5 * net / (2 * rho[k] * dz[k] * xsize)
        
        # Apply adjustments to boundaries.
        for j in range(jtot):
            u[k, j, 0]  -= du
            u[k, j, -1] += du
        
        for i in range(itot):
            v[k, 0, i]  -= dv
            v[k, -1, i] += dv


def check_global_mass_inbalance(u, v, w, rho, rhoh, dz, dx, dy):
    """
    Check global mass flux inbalance over in- and outflow boundaries.

    Parameters:
    ----------
    u : np.ndarray, shape (ktot, jtot, itot+1)
        Zonal wind field.
    v : np.ndarray, shape (ktot, jtot+1, itot)
        Meridional wind field.
    w : np.ndarray, shape (ktot+1, jtot, itot)
        Vertical velocity field.
    rho : np.ndarray, shape (ktot,)
        Base state density at full levels.
    rhoh : np.ndarray, shape (ktot+1,)
        Base state density at half levels.
    dz : np.ndarray, shape (ktot,)
        Vertical grid spacing.
    dx : float
        Horizontal grid spacing in x-direction.
    dy : float
        Horizontal grid spacing in y-direction.

    Returns:
    -------
    float
        Net mass flux imbalance in kg/s.
    """
    u_in  = (u[:,:,0]  * rho[:,None] * dy * dz[:,None]).sum()
    u_out = (u[:,:,-1] * rho[:,None] * dy * dz[:,None]).sum()

    v_in  = (v[:,0,:]  * rho[:,None] * dx * dz[:,None]).sum()
    v_out = (v[:,-1,:] * rho[:,None] * dx * dz[:,None]).sum()

    w_in  = (w[0 ,:,:] * rhoh[0 ] * dx * dy).sum()
    w_out = (w[-1,:,:] * rhoh[-1] * dx * dy).sum()

    return u_in - u_out + v_in - v_out + w_in - w_out


@jit(nopython=True, boundscheck=False, fastmath=True)
def calc_divergence(u, v, w, rho, rhoh, dzi, dxi, dyi, itot, jtot, ktot):
    """
    Calculate maximum divergence in the velocity field and its location.

    Parameters:
    ----------
    u : np.ndarray, shape (ktot, jtot, itot+1)
        Zonal wind field.
    v : np.ndarray, shape (ktot, jtot+1, itot)
        Meridional wind field.
    w : np.ndarray, shape (ktot+1, jtot, itot)
        Vertical velocity field.
    rho : np.ndarray, shape (ktot,)
        Base state density at full levels.
    rhoh : np.ndarray, shape (ktot+1,)
        Base state density at half levels.
    dzi : np.ndarray, shape (ktot,)
        Inverse vertical grid spacing.
    dxi : float
        Inverse horizontal grid spacing in x-direction.
    dyi : float
        Inverse horizontal grid spacing in y-direction.

    Returns:
    -------
    tuple of (float, int, int, int)
        Maximum divergence value (kg/m3/s) and its location (i, j, k).
    """

    max_div = 0.
    max_i = 0
    max_j = 0
    max_k = 0
    for k in range(ktot):
        for j in range(jtot):
            for i in range(itot):
                div = rho[k] * ((u[k,j,i+1] - u[k,j,i]) * dxi \
                             +  (v[k,j+1,i] - v[k,j,i]) * dyi) \
                             + (rhoh[k+1] * w[k+1,j,i] - rhoh[k] * w[k,j,i]) * dzi[k]
                div = abs(div)

                if div > max_div:
                    max_div = div
                    max_i = i
                    max_j = j
                    max_k = k

    return max_div, max_i, max_j, max_k


@jit(nopython=True, boundscheck=False, fastmath=True)
def input_kernel(
        p, u, v, w, ut, vt, wt, rho, rhoh, dzi, dxi, dyi, dti, itot, jtot, ktot):
    """
    Prepare input for DCTs by calculating divergence field.

    Parameters:
    ----------
    p : np.ndarray, shape (ktot, jtot, itot)
        Pressure field (output, modified in place).
    u : np.ndarray, shape (ktot, jtot, itot+1)
        Zonal wind field.
    v : np.ndarray, shape (ktot, jtot+1, itot)
        Meridional wind field.
    w : np.ndarray, shape (ktot+1, jtot, itot)
        Vertical velocity field.
    ut : np.ndarray, shape (ktot, jtot, itot+1)
        Zonal wind tendency.
    vt : np.ndarray, shape (ktot, jtot+1, itot)
        Meridional wind tendency.
    wt : np.ndarray, shape (ktot+1, jtot, itot)
        Vertical velocity tendency.
    rho : np.ndarray, shape (ktot,)
        Base state density at full levels.
    rhoh : np.ndarray, shape (ktot+1,)
        Base state density at half levels.
    dzi : np.ndarray, shape (ktot,)
        Inverse vertical grid spacing.
    dxi : float
        Inverse horizontal grid spacing in x-direction.
    dyi : float
        Inverse horizontal grid spacing in y-direction.
    dti : float
        Inverse time step.
    itot : int
        Number of grid points in x-direction.
    jtot : int
        Number of grid points in y-direction.
    ktot : int
        Number of grid points in z-direction.
    """

    for k in range(ktot):
        for j in range(jtot):
            for i in range(itot):
                p[k,j,i] = rho[k] * ( (ut[k,j,i+1] + u[k,j,i+1] * dti) - (ut[k,j,i] + u[k,j,i] * dti) ) * dxi \
                         + rho[k] * ( (vt[k,j+1,i] + v[k,j+1,i] * dti) - (vt[k,j,i] + v[k,j,i] * dti) ) * dyi \
                         + ( rhoh[k+1] * (wt[k+1,j,i] + w[k+1,j,i] * dti) \
                         -   rhoh[k  ] * (wt[k,  j,i] + w[k,  j,i] * dti) ) * dzi[k]


@jit(nopython=True, boundscheck=False, fastmath=True)
def solve_pre_kernel(b, p, dz, rho, bmati, bmatj, a, c, itot, jtot, ktot):
    """
    Prepare input for TDMA solver by setting up the tridiagonal matrix coefficients.

    Parameters:
    ----------
    b : np.ndarray, shape (ktot, jtot, itot)
        Diagonal coefficients of tridiagonal matrix (output, modified in place).
    p : np.ndarray, shape (ktot, jtot, itot)
        Pressure field (modified in place).
    dz : np.ndarray, shape (ktot,)
        Vertical grid spacing.
    rho : np.ndarray, shape (ktot,)
        Base state density at full levels.
    bmati : np.ndarray, shape (itot,)
        Modified wave numbers for x-direction.
    bmatj : np.ndarray, shape (jtot,)
        Modified wave numbers for y-direction.
    a : np.ndarray, shape (ktot,)
        Lower diagonal coefficients of tridiagonal matrix.
    c : np.ndarray, shape (ktot,)
        Upper diagonal coefficients of tridiagonal matrix.
    itot : int
        Number of grid points in x-direction.
    jtot : int
        Number of grid points in y-direction.
    ktot : int
        Number of grid points in z-direction.
    """

    for k in range(ktot):
        for j in range(jtot):
            for i in range(itot):
                b[k,j,i] = dz[k]*dz[k] * rho[k]*(bmati[i]+bmatj[j]) - (a[k]+c[k])
                p[k,j,i] = dz[k]*dz[k] * p[k,j,i]

    for j in range(jtot):
        for i in range(itot):
            b[0,j,i] += a[0]

            k = ktot - 1
            if i == 0 and j == 0:
                b[k,j,i] -= c[k]
            else:
                b[k,j,i] += c[k]


@jit(nopython=True, boundscheck=False, fastmath=True)
def tdma_kernel(p, a, b, c, work2d, work3d, itot, jtot, ktot):
    """
    Solve Poisson equation using TriDiagonal Matrix Algorithm (TDMA).

    Parameters:
    ----------
    p : np.ndarray, shape (ktot, jtot, itot)
        Pressure field (modified in place to contain the solution).
    a : np.ndarray, shape (ktot,)
        Lower diagonal coefficients of tridiagonal matrix.
    b : np.ndarray, shape (ktot, jtot, itot)
        Diagonal coefficients of tridiagonal matrix.
    c : np.ndarray, shape (ktot,)
        Upper diagonal coefficients of tridiagonal matrix.
    work2d : np.ndarray, shape (jtot, itot)
        2D work array for temporary storage.
    work3d : np.ndarray, shape (ktot, jtot, itot)
        3D work array for temporary storage.
    itot : int
        Number of grid points in x-direction.
    jtot : int
        Number of grid points in y-direction.
    ktot : int
        Number of grid points in z-direction.
    """

    for j in range(jtot):
        for i in range(itot):
            work2d[j,i] = b[0,j,i]
            p[0,j,i] /= work2d[j,i]

            for k in range(1, ktot):
                work3d[k,j,i] = c[k-1] / work2d[j,i]
                work2d[j,i] = b[k,j,i] - a[k] * work3d[k,j,i]
                p[k,j,i] -= a[k] * p[k-1,j,i]
                p[k,j,i] /= work2d[j,i]

            for k in range(ktot-2, -1, -1):
                p[k,j,i] -= work3d[k+1,j,i] * p [k+1,j,i]


@jit(nopython=True, boundscheck=False, fastmath=True)
def solve_2d_kernel(p, dz, rho, bmati, bmatj, itot, jtot, ktot):
    """
    Solve 2D Poisson equation for each level independently.
    Since we're only updating u and v (not w), there's no vertical coupling.

    Parameters:
    ----------
    p : np.ndarray, shape (ktot, jtot, itot)
        Pressure field (modified in place to contain the solution).
    dz : np.ndarray, shape (ktot,)
        Vertical grid spacing.
    rho : np.ndarray, shape (ktot,)
        Base state density at full levels.
    bmati : np.ndarray, shape (itot,)
        Modified wave numbers for x-direction.
    bmatj : np.ndarray, shape (jtot,)
        Modified wave numbers for y-direction.
    itot : int
        Number of grid points in x-direction.
    jtot : int
        Number of grid points in y-direction.
    ktot : int
        Number of grid points in z-direction.
    """

    for k in range(ktot):
        dz2 = dz[k] * dz[k]
        for j in range(jtot):
            for i in range(itot):
                b = dz2 * rho[k] * (bmati[i] + bmatj[j])

                if i == 0 and j == 0:
                    p[k,j,i] = 0.
                else:
                    p[k,j,i] = (dz2 * p[k,j,i]) / b


@jit(nopython=True, boundscheck=False, fastmath=True)
def calc_tendency_kernel(ut, vt, wt, p, dzhi, dxi, dyi, itot, jtot, ktot, solve_w):
    """
    Calculate tendencies from pressure field.
    Exclude left- and right most u, v, top/bottom w, etc.
    In MicroHH, this is done by settings the pressure gradients to zero.

    Parameters:
    ----------
    ut : np.ndarray, shape (ktot, jtot, itot+1)
        Zonal wind tendency (modified in place).
    vt : np.ndarray, shape (ktot, jtot+1, itot)
        Meridional wind tendency (modified in place).
    wt : np.ndarray, shape (ktot+1, jtot, itot)
        Vertical velocity tendency (modified in place).
    p : np.ndarray, shape (ktot, jtot, itot)
        Pressure field.
    dzhi : np.ndarray, shape (ktot+1,)
        Inverse vertical grid spacing at half levels.
    dxi : float
        Inverse horizontal grid spacing in x-direction.
    dyi : float
        Inverse horizontal grid spacing in y-direction.
    itot : int
        Number of grid points in x-direction.
    jtot : int
        Number of grid points in y-direction.
    ktot : int
        Number of grid points in z-direction.
    solve_w : bool
        If True, also solve for vertical velocity tendency.
    """

    for k in range(ktot):
        for j in range(jtot):
            for i in range(1,itot):
                ut[k,j,i] -= (p[k,j,i] - p[k,j,i-1]) * dxi

    for k in range(ktot):
        for j in range(1,jtot):
            for i in range(itot):
                vt[k,j,i] -= (p[k,j,i] - p[k,j-1,i]) * dyi

    if solve_w:
        for k in range(1,ktot):
            for j in range(jtot):
                for i in range(itot):
                    wt[k,j,i] -= (p[k,j,i] - p[k-1,j,i]) * dzhi[k]


def dct_forward(fld):
    """
    Perform 2D forward discrete cosine transform (DCT-II) over horizontal dimensions.

    Parameters:
    ----------
    fld : np.ndarray, shape (ktot, jtot, itot)
        Input field to transform.

    Returns:
    -------
    np.ndarray, shape (ktot, jtot, itot)
        Transformed field.
    """

    fld = dct(fld, type=2, axis=2, norm='ortho')
    fld = dct(fld, type=2, axis=1, norm='ortho')
    return fld


def dct_backward(fld):
    """
    Perform 2D inverse discrete cosine transform (DCT-III) over horizontal dimensions.

    Parameters:
    ----------
    fld : np.ndarray, shape (ktot, jtot, itot)
        Input field to transform.

    Returns:
    -------
    np.ndarray, shape (ktot, jtot, itot)
        Transformed field.
    """

    fld = dct(fld, type=3, axis=1, norm='ortho')
    fld = dct(fld, type=3, axis=2, norm='ortho')
    return fld


def solve_pressure_dct(
        p, u, v, w,
        ut, vt, wt,
        rho, rhoh,
        dx, dy,
        dz, dzi, dzhi,
        dt,
        solve_w,
        float_type):
    """
    Solve pressure using discrete cosine transform, and update velocities
    to obtain divergence free velocity fields.

    Updates `p`, `u`, `v`, and `w` in place.

    NOTE: The input velocity fields should be shaped:
        u[ktot, jtot, itot+1]
        v[ktot, jtot+1, itot]
        w[ktot+1, jtot, itot]
    Where u[:,:,0] and u[:,:,-1], v[:,0,:] and v[:,-1,:], w[0,:,:], w[-1,:,:] etc. are the BCs to solve to.

    Parameters:
    ----------
    p : np.ndarray, shape (ktot, jtot, itot)
        Pressure field (modified in place).
    u : np.ndarray, shape (ktot, jtot, itot+1)
        Zonal wind field.
    v : np.ndarray, shape (ktot, jtot+1, itot)
        Meridional wind field.
    w : np.ndarray, shape (ktot+1, jtot, itot)
        Vertical velocity field.
    ut : np.ndarray, shape (ktot, jtot, itot+1)
        Zonal wind tendency.
    vt : np.ndarray, shape (ktot, jtot+1, itot)
        Meridional wind tendency.
    wt : np.ndarray, shape (ktot+1, jtot, itot)
        Vertical velocity tendency.
    rho : np.ndarray, shape (ktot,)
        Base state density at full levels.
    rhoh : np.ndarray, shape (ktot+1,)
        Base state density at half levels.
    dx : float
        Horizontal grid spacing in x-direction.
    dy : float
        Horizontal grid spacing in y-direction.
    dz : np.ndarray, shape (ktot,)
        Vertical grid spacing.
    dzi : np.ndarray, shape (ktot,)
        Inverse vertical grid spacing.
    dzhi : np.ndarray, shape (ktot+1,)
        Inverse vertical grid spacing at half levels.
    dt : float
        Time step.
    solve_w : bool
        If True, solve for vertical velocity using 3D Poisson equation;
        if False, solve only for u and v using 2D Poisson equation per level.
    float_type : dtype
        Floating point precision to use for calculations.
    """

    ktot, jtot, itot = p.shape

    dxi = 1./dx
    dyi = 1./dy

    dxidxi = dxi*dxi
    dyidyi = dyi*dyi

    dti = 1./dt

    # Velocity tendencies.
    ut = np.zeros_like(u)
    vt = np.zeros_like(v)
    wt = np.zeros_like(w)

    # Help/tmp arrays.
    work3d = np.zeros_like(p)
    work2d = np.zeros((jtot, itot), dtype=float_type)

    # Modified wave numbers for the cosine transform.
    bmati = 2. * (np.cos(np.pi * np.arange(itot)/itot) - 1.) * dxidxi
    bmatj = 2. * (np.cos(np.pi * np.arange(jtot)/jtot) - 1.) * dyidyi

    # Create vectors that go into the tridiagonal matrix solver.
    a = dz[:] * rhoh[:-1] * dzhi[:-1]
    b = np.zeros_like(p)
    c = dz[:] * rhoh[1: ] * dzhi[1: ]

    # Prepare input for DCTs.
    input_kernel(p, u, v, w, ut, vt, wt, rho, rhoh, dzi, dxi, dyi, dti, itot, jtot, ktot)

    # Forward 2D DCT over full 3D field.
    p[:,:,:] = dct_forward(p)

    if solve_w:
        # Prepare input for TDMA solver.
        solve_pre_kernel(b, p, dz, rho, bmati, bmatj, a, c, itot, jtot, ktot)

        # Solve vertical Poisson equation.
        tdma_kernel(p, a, b, c, work2d, work3d, itot, jtot, ktot)
    else:
        # Solve 2D Poisson equation.
        solve_2d_kernel(p, dz, rho, bmati, bmatj, itot, jtot, ktot)

    # Backward 2D DCT over full 3D field.
    p[:,:,:] = dct_backward(p)


def make_divergence_free_dct(u, v, w, rho, rhoh, dx, dy, dz, dzi, dzhi, solve_w, float_type):
    """
    Make velocity fields divergence free using (modified) Poisson solver.

    Parameters:
    ----------
    u : np.ndarray, shape (ktot, jtot, itot+1)
        Zonal wind field (modified in place).
    v : np.ndarray, shape (ktot, jtot+1, itot)
        Meridional wind field (modified in place).
    w : np.ndarray, shape (ktot+1, jtot, itot)
        Vertical velocity field (modified in place).
    rho : np.ndarray, shape (ktot,)
        Base state density at full levels.
    rhoh : np.ndarray, shape (ktot+1,)
        Base state density at half levels.
    dx : float
        Horizontal grid spacing in x-direction.
    dy : float
        Horizontal grid spacing in y-direction.
    dz : np.ndarray, shape (ktot,)
        Vertical grid spacing.
    dzi : np.ndarray, shape (ktot,)
        Inverse vertical grid spacing.
    dzhi : np.ndarray, shape (ktot+1,)
        Inverse vertical grid spacing at half levels.
    solve_w : bool
        If True, solve for vertical velocity; if False, only solve for u and v.
    float_type : dtype
        Floating point precision to use for calculations.
    """
    itot = w.shape[2]
    jtot = w.shape[1]
    ktot = u.shape[0]

    dxi = 1/dx
    dyi = 1/dy

    dt = 1.     # dummy...

    p = np.zeros((ktot, jtot, itot), dtype=float_type)
    ut = np.zeros_like(u)
    vt = np.zeros_like(v)
    wt = np.zeros_like(w)

    # Correct global mass inbalance.
    net_in = check_global_mass_inbalance(u, v, w, rho, rhoh, dz, dx, dy)
    adjust_boundary_mass_flux(u, v, w, rho, rhoh, dz, dx, dy, itot, jtot, ktot)
    net_out = check_global_mass_inbalance(u, v, w, rho, rhoh, dz, dx, dy)

    # Solve pressure.
    solve_pressure_dct(p, u, v, w, ut, vt, wt, rho, rhoh, dx, dy, dz, dzi, dzhi, dt, solve_w, float_type)

    # Calculate tendencies from pressure gradient.
    calc_tendency_kernel(ut, vt, wt, p, dzhi, dxi, dyi, itot, jtot, ktot, solve_w)

    # Integrate to obtain divergence free solution.
    u += ut * dt
    v += vt * dt
    w += wt * dt

    div,i,j,k = calc_divergence(u, v, w, rho, rhoh, dzi, dxi, dyi, itot, jtot, ktot)

    logger.debug(f'Mass inbalance input = {net_in:.1e} kg/s, output = {net_out:.1e} kg/s, max div. = {div:.1e} kg/m3/s @ i={i}, j={j}, k={k}')