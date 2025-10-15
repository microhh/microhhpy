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


@jit(nopython=True, boundscheck=False, fastmath=True)
def input_kernel(
        p, u, v, w, ut, vt, wt, rho, rhoh, dzi, dxi, dyi, dti, itot, jtot, ktot):
    """
    Prepare input for DCTs.
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
    Prepare input for TDMA solver.
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
    Solve Poisson equation.
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
def calc_tendency_kernel(ut, vt, wt, p, dzhi, dxi, dyi, itot, jtot, ktot):
    """
    Calculate tendencies from pressure field.
    Exlude left- and right most u, v, top/bottom w, etc.
    In MicroHH, this is done by settings the pressure gradients to zero.
    """
    for k in range(ktot):
        for j in range(jtot):
            for i in range(1,itot):
                ut[k,j,i] -= (p[k,j,i] - p[k,j,i-1]) * dxi

    for k in range(ktot):
        for j in range(1,jtot):
            for i in range(itot):
                vt[k,j,i] -= (p[k,j,i] - p[k,j-1,i]) * dyi

    for k in range(1,ktot):
        for j in range(jtot):
            for i in range(itot):
                wt[k,j,i] -= (p[k,j,i] - p[k-1,j,i]) * dzhi[k]


@jit(nopython=True, boundscheck=False, fastmath=True)
def calc_divergence(u, v, w, rho, rhoh, dzi, dxi, dyi):

    ktot = u.shape[0]
    jtot = u.shape[1]
    itot = v.shape[2]

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


def dct_forward(fld):
    fld = dct(fld, type=2, axis=2, norm='ortho')
    fld = dct(fld, type=2, axis=1, norm='ortho')
    return fld


def dct_backward(fld):
    fld = dct(fld, type=3, axis=1, norm='ortho')
    fld = dct(fld, type=3, axis=2, norm='ortho')
    return fld


def solve_pressure_dct(
        p, u, v, w,
        rho, rhoh,
        dx, dy,
        dz, dzi, dzhi,
        float_type):
    """
    Solve pressure using discrete cosine transform, and update velocities
    to obtain divergence free velocity fields.

    Updates `p`, `u`, `v`, and `w` in place.

    NOTE: The input velocity fields should be shaped:
        u[ktot, jtot, itot+1]
        v[ktot, jtot+1, itot]
        w[ktot+1, jtot, itot]
    Where u[:,:,0] and u[:,:,-1], v[0,:,0], w[0,:,:], w[-1,:,:] etc. are the BCs to solve to.
    """

    ktot, jtot, itot = p.shape

    dxi = 1./dx
    dyi = 1./dy

    dxidxi = dxi*dxi
    dyidyi = dyi*dyi

    dt = 1.
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
    p = dct_forward(p)

    # Prepare input for TDMA solver.
    solve_pre_kernel(b, p, dz, rho, bmati, bmatj, a, c, itot, jtot, ktot)

    # Solve vertical Poisson equation.
    tdma_kernel(p, a, b, c, work2d, work3d, itot, jtot, ktot)

    # Backward 2D DCT over full 3D field.
    p = dct_backward(p)

    # Calculate tendencies from pressure gradients.
    calc_tendency_kernel(ut, vt, wt, p, dzhi, dxi, dyi, itot, jtot, ktot)

    # Integrate to obtain divergence free solution.
    # Boundary conditions (e.g. u[:,:,0], w[-1,:,:], etc) are excluded in `calc_tendency_kernel`.
    u += ut * dt
    v += vt * dt
    w += wt * dt
