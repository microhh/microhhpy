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
from numba import jit, prange
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

    logger.debug(f'Velocity corrections: mean duv/dxy={du_dx.mean()*1000} m/s/km')

    # Distance from domain center in `x,y`.
    xp = x - xsize / 2.
    yp = y - ysize / 2.

    # Correct velocities in-place.
    u[:,:,:] += du_dx[:,None,None] * xp[None,None,:]
    v[:,:,:] += dv_dy[:,None,None] * yp[None,:,None]