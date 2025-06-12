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

# Local library
from microhhpy.logger import logger


def strip_mask(array):
    """
    Remote mask from Numpy array if present. Numba does not like masked arrays...
    """
    if isinstance(array, np.ma.MaskedArray):
        return array.data
    else:
        return array


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
    Calculate maximum divergence in LES domain.
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
