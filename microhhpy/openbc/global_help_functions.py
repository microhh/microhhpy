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
from numba import jit
from scipy.ndimage import gaussian_filter
import xarray as xr

# Local library
from microhhpy.logger import logger


def gaussian_filter_wrapper(fld, sigma):
    """
    Wrapper around Scipy's `gaussian_filter`. Older versions don't support the axis keyword,
    so manually loop over height. This does not influence performance, so Scipy probably does the same..
    """
    for k in range(fld.shape[0]):
        fld[k,:,:] = gaussian_filter(fld[k,:,:], sigma, mode='nearest')


@jit(nopython=True, nogil=True, fastmath=True)
def blend_w_to_zero_at_sfc(w, zh, zmax):
    """
    Blend `w` towards zero at the surface, from a height `zmax` down.

    Arguments:
    ---------
    w : np.ndarray, shape (3,)
        Large-scale vertical velocity.
    zh : np.ndarray, shape (3,)
        Half-level model level heights.
    zmax : float
        Ending height of blending.
    """
    kmax = np.argmin(np.abs(zh - zmax))
    zmax = zh[kmax]

    _, jtot, itot = w.shape

    for j in range(jtot):
        for i in range(itot):
            dwdz = w[kmax,j,i] / zmax

            for k in range(kmax):
                f = zh[k] / zmax
                w[k,j,i] = f * w[k,j,i] + (1-f) * dwdz * zh[k]


def correct_div_uv(
        u,
        v,
        wls,
        rho,
        rhoh,
        dzi,
        x,
        y,
        xsize,
        ysize,
        n_pad):
    """
    Apply horizontal divergence correction to `u` and `v` fields so that their divergence
    matches the target from the large-scale subsidence `wls`.

    The target divergence is calculated from the vertical derivative of `rho * wls`,
    while the actual divergence is derived from `u` and `v`. The mismatch is distributed equally
    over both horizontal velocity components.

    NOTE: This function modifies `u` and `v` in-place.

    Arguments:
    ---------
    u : np.ndarray, shape (3,)
        Zonal wind on LES grid.
    v : np.ndarray, shape (3,)
        Meridional wind on LES grid.
    wls : np.ndarray, shape (3,)
        Large-scale vertical velocity field on LES grid.
    rho : np.ndarray, shape (1,)
        Basestate air density at full levels.
    rhoh : np.ndarray, shape (1,)
        Basestate air density at half levels.
    dzi : np.ndarray, shape (1,)
        Inverse vertical grid spacing between half levels.
    x : np.ndarray, shape (1,)
        x-coordinates of LES grid.
    y : np.ndarray, shape (1,)
        y-coordinates of LES grid.
    xsize : float
        Domain width in x-direction.
    ysize : float
        Domain width in y-direction.
    n_pad : int
        Number of horizontal ghost cells in LES domain, including padding.

    Returns:
    -------
    None
    """
    logger.debug('Correcting horizontal divergence u and v')

    # Take mean over interpolated field without ghost cells, to get target mean subsidence velocity.
    w_target = wls[:, n_pad:-n_pad, n_pad:-n_pad].mean(axis=(1,2))

    # Calculate target horizontal divergence `rho * (du/dx + dv/dy)`.
    rho_w = rhoh * w_target
    div_uv_target = -(rho_w[1:] - rho_w[:-1]) * dzi[:]

    # Calculate actual divergence from interpolated `u,v` fields.
    div_u = rho * (u[:, n_pad:-n_pad, -n_pad].mean(axis=1) - u[:, n_pad:-n_pad,  n_pad].mean(axis=1)) / xsize
    div_v = rho * (v[:, -n_pad, n_pad:-n_pad].mean(axis=1) - v[:, n_pad,  n_pad:-n_pad].mean(axis=1)) / ysize
    div_uv_actual = div_u + div_v

    # Required change in horizontal divergence.
    diff_div = div_uv_target - div_uv_actual

    # Distribute over `u,v`: how to best do this? For now 50/50.
    du_dx = diff_div / 2. / rho
    dv_dy = diff_div / 2. / rho

    logger.debug(f'Velocity corrections: mean du/dx={du_dx.mean()*1000} m/s/km, dv/dy={dv_dy.mean()*1000} m/s/km')

    # Distance from domain center in `x,y`.
    xp = x - xsize / 2.
    yp = y - ysize / 2.

    # Correct velocities in-place.
    u[:,:,:] += du_dx[:,None,None] * xp[None,None,:]
    v[:,:,:] += dv_dy[:,None,None] * yp[None,:,None]


@jit(nopython=True, nogil=True, fastmath=True)
def calc_w_from_uv(
        w,
        u,
        v,
        rho,
        rhoh,
        dz,
        dxi,
        dyi,
        istart, iend,
        jstart, jend,
        ktot):
    """
    Calculate vertical velocity `w` from horizontal wind components `u` and `v` and the
    continuity equation, with w==0 at a lower boundary condition.

    Arguments:
    ---------
    w : np.ndarray, shape (3,)
        Vertical velocity field.
    u : np.ndarray, shape (3,)
        Zonal wind field.
    v : np.ndarray, shape (3,)
        Meridional wind field.
    rho : np.ndarray, shape (1,)
        Basestate air density at full levels.
    rhoh : np.ndarray, shape (1,)
        Basestate air density at half levels.
    dz : np.ndarray, shape (1,)
        Vertical grid spacing between full levels.
    dxi : float
        Inverse horizontal grid spacing in x-direction.
    dyi : float
        Inverse horizontal grid spacing in y-direction.
    istart : int
        Start index in x-direction.
    iend : int
        End index in x-direction.
    jstart : int
        Start index in y-direction.
    jend : int
        End index in y-direction.
    ktot : int
        Number of full vertical levels.

    Returns:
    -------
    None
    """

    for j in range(jstart, jend):
        for i in range(istart, iend):
            w[0,j,i] = 0.

    for k in range(ktot):
        for j in range(jstart, jend):
            for i in range(istart, iend):
                w[k+1,j,i] = -(rho[k] * ((u[k,j,i+1] - u[k,j,i]) * dxi + \
                                         (v[k,j+1,i] - v[k,j,i]) * dyi) * dz[k] - \
                                         rhoh[k] * w[k,j,i]) / rhoh[k+1]


@jit(nopython=True, nogil=True, fastmath=True)
def check_divergence(
        u,
        v,
        w,
        rho,
        rhoh,
        dxi,
        dyi,
        dzi,
        istart, iend,
        jstart, jend,
        ktot):
    """
    Calculate the maximum mass divergence in the LES domain using the continuity equation.

    Arguments:
    ---------
    u : np.ndarray, shape (3,)
        Zonal wind field.
    v : np.ndarray, shape (3,)
        Meridional wind field.
    w : np.ndarray, shape (3,)
        Vertical velocity field.
    rho : np.ndarray, shape (1,)
        Basestate density at full levels.
    rhoh : np.ndarray, shape (1,)
        Basestate density at half levels.
    dxi : float
        Inverse horizontal grid spacing in x-direction.
    dyi : float
        Inverse horizontal grid spacing in y-direction.
    dzi : np.ndarray, shape (1,)
        Inverse vertical grid spacing.
    istart : int
        Start index in x-direction.
    iend : int
        End index in x-direction.
    jstart : int
        Start index in y-direction.
    jend : int
        End index in y-direction.
    ktot : int
        Number of full vertical levels.

    Returns:
    -------
    div_max : float
        Maximum absolute divergence.
    i_max : int
        x-index of maximum divergence.
    j_max : int
        y-index of maximum divergence.
    k_max : int
        z-index of maximum divergence.
    """

    div_max = 0.
    i_max = 0
    j_max = 0
    k_max = 0

    for k in range(ktot):
        for j in range(jstart, jend):
            for i in range(istart, iend):
                div = rho[k] * (u[k,j,i+1] - u[k,j,i]) * dxi + \
                      rho[k] * (v[k,j+1,i] - v[k,j,i]) * dyi + \
                      ((rhoh[k+1] * w[k+1,j,i]) - (rhoh[k] * w[k,j,i])) * dzi[k]

                if abs(div) > div_max:
                    div_max = abs(div)
                    i_max = i
                    j_max = j
                    k_max = k

    return div_max, i_max, j_max, k_max


def create_lbc_ds(
        fields,
        x,
        y,
        z,
        xh,
        yh,
        zh,
        time,
        start_date,
        n_ghost,
        n_sponge,
        x_offset=0,
        y_offset=0,
        dtype=np.float64):
    """
    Create an Xarray Dataset with lateral boundary conditions for MicroHH.

    Arguments:
    ---------
    fields : list of str
        Field names to include.
    x : np.ndarray
        x-coordinates on full levels.
    y : np.ndarray
        y-coordinates on full levels.
    z : np.ndarray
        z-coordinates on full levels.
    xh : np.ndarray
        x-coordinates on half levels.
    yh : np.ndarray
        y-coordinates on half levels.
    zh : np.ndarray
        z-coordinates on half levels.
    time : np.ndarray
        Time values (seconds since start).
    start_date : datetime.datetime
        Start date of the simulation.
    n_ghost : int
        Number of ghost cells.
    n_sponge : int
        Number of sponge cells.
    x_offset : float, optional
        Offset in x-direction (default: 0).
    y_offset : float, optional
        Offset in y-direction (default: 0).
    dtype : np.dtype, optional
        Data type for field arrays (default: np.float64).

    Returns:
    -------
    ds : xr.Dataset
        Dataset with initialized boundary fields and coordinates for MicroHH LBC input.
    """

    nt = time.size
    itot = x.size
    jtot = y.size
    ktot = z.size

    nlbc = n_ghost + n_sponge

    # Dimension sizes.
    dims = {
        'time': nt,
        'x': itot + 2*n_ghost,
        'xh': itot + 2*n_ghost,
        'xgw': nlbc,
        'xge': nlbc,
        'xhgw': nlbc + 1,
        'xhge': nlbc,
        'y': jtot + 2*n_ghost,
        'yh': jtot + 2*n_ghost,
        'ygn': nlbc,
        'ygs': nlbc,
        'yhgs': nlbc + 1,
        'yhgn': nlbc,
        'z': ktot}

    # Coordinates.
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    # Pad x,y dimensions with ghost cells.
    xp = np.zeros(x.size+2*n_ghost)
    xhp = np.zeros(x.size+2*n_ghost)

    yp = np.zeros(y.size+2*n_ghost)
    yhp = np.zeros(y.size+2*n_ghost)

    xp[n_ghost:-n_ghost] = x
    xhp[n_ghost:-n_ghost] = xh

    yp[n_ghost:-n_ghost] = y
    yhp[n_ghost:-n_ghost] = yh

    for i in range(n_ghost):
        xp[i] = x[0] - (n_ghost-i)*dx
        xhp[i] = xh[0] - (n_ghost-i)*dx

        yp[i] = y[0] - (n_ghost-i)*dy
        yhp[i] = yh[0] - (n_ghost-i)*dy

        xp[itot+n_ghost+i] = x[-1] + (i+1)*dx
        xhp[itot+n_ghost+i] = xh[-1] + (i+1)*dx

        yp[jtot+n_ghost+i] = y[-1] + (i+1)*dy
        yhp[jtot+n_ghost+i] = yh[-1] + (i+1)*dy

    # Add domain offsets (location in parent).
    xp += x_offset
    xhp += x_offset

    yp += y_offset
    yhp += y_offset

    # Define coordinates.
    coords = {
        'time': time,
        'x': xp,
        'xh': xhp,
        'y': yp,
        'yh': yhp,
        'z': z,
        'zh': zh,
        'xgw': xp[:nlbc],
        'xge': xp[itot+n_ghost-n_sponge:],
        'xhgw': xhp[:nlbc+1],
        'xhge': xhp[itot+n_ghost-n_sponge:],
        'ygs': yp[:nlbc],
        'ygn': yp[jtot+n_ghost-n_sponge:],
        'yhgs': yhp[:nlbc+1],
        'yhgn': yhp[jtot+n_ghost-n_sponge:]}

    # Create Xarray dataset.
    ds = xr.Dataset(coords=coords)

    def get_dim_size(dim_in):
        out = []
        for dim in dim_in:
            out.append(coords[dim].size)
        return out

    def add_var(name, dims):
        dim_size = get_dim_size(dims)
        ds[name] = (dims, np.zeros(dim_size, dtype=dtype))

    for fld in fields:
        if fld not in ('u','v','w'):
            add_var(f'{fld}_west', ('time', 'z', 'y', 'xgw'))
            add_var(f'{fld}_east', ('time', 'z', 'y', 'xge'))
            add_var(f'{fld}_south', ('time', 'z', 'ygs', 'x'))
            add_var(f'{fld}_north', ('time', 'z', 'ygn', 'x'))

    if 'u' in fields:
        add_var('u_west', ('time', 'z', 'y', 'xhgw'))
        add_var('u_east', ('time', 'z', 'y', 'xhge'))
        add_var('u_south', ('time', 'z', 'ygs', 'xh'))
        add_var('u_north', ('time', 'z', 'ygn', 'xh'))

    if 'v' in fields:
        add_var('v_west', ('time', 'z', 'yh', 'xgw'))
        add_var('v_east', ('time', 'z', 'yh', 'xge'))
        add_var('v_south', ('time', 'z', 'yhgs', 'x'))
        add_var('v_north', ('time', 'z', 'yhgn', 'x'))

    if 'w' in fields:
        add_var('w_west', ('time', 'zh', 'y', 'xgw'))
        add_var('w_east', ('time', 'zh', 'y', 'xge'))
        add_var('w_south', ('time', 'zh', 'ygs', 'x'))
        add_var('w_north', ('time', 'zh', 'ygn', 'x'))

    ds.time.attrs['units'] = f'seconds since {start_date.isoformat()}'

    return ds