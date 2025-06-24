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

# Local library

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


@jit(nopython=True, nogil=True, fastmath=True)
def block_perturb_field(
    fld,
    block_size,
    amplitude):
    """
    Add random perturbations to field, in block sizes in all spatial directions of size `block_size`.

    Arguments:
    ---------
    fld : np.ndarray, shape (3,)
        Field to perturb.
    block_size : int
        Size of the blocks in each spatial direction.
    amplitude : float
        Amplitude of the perturbations to add.

    Returns:
    -------
    None
    """
    
    ktot, jtot, itot = fld.shape
    nblock_k = int(np.ceil(ktot / block_size))
    nblock_j = int(np.ceil(jtot / block_size))
    nblock_i = int(np.ceil(itot / block_size))

    for bk in range(nblock_k):
        for bj in range(nblock_j):
            for bi in prange(nblock_i):
                random_val = 2 * amplitude * (np.random.random()-0.5)

                for dk in range(block_size):
                    for dj in range(block_size):
                        for di in range(block_size):
                            k = bk*block_size + dk
                            j = bj*block_size + dj
                            i = bi*block_size + di

                            # Bounds check in case `dim % blocksize != 0`.
                            if k < ktot and j<jtot and i<itot:
                                fld[k,j,i] += random_val