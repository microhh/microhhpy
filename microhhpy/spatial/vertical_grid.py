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

# Local library


def calc_vertical_grid_2nd(z, zsize, remove_ghost=True, float_type=np.float64):
    """
    Calculate vertical grid, identical to definition in MicroHH.

    Parameters:
    ----------
    z : np.ndarray, shape (1,)
        Array with input full level heights, like in `case_input.nc`.
    zsize : float
        Height of domain top.
    remove_ghost : bool, optional
        Clip off the ghost cells, leaving `ktot` full and `ktot+` half levels. Default is True.
    float_type : np.float32 or np.float64, optional
        Output datatype of arrays.

    Returns:
    -------
    vertical grid : dict
        Dictionary containing grid properties.
    """

    z_in = z.copy()

    ktot = z.size
    kcells = ktot+2

    kstart = 1
    kend = ktot+1

    z = np.zeros(kcells, float_type)
    zh = np.zeros(kcells, float_type)

    dz = np.zeros(kcells, float_type)
    dzh = np.zeros(kcells, float_type)

    # Full level heights
    z[kstart:kend] = z_in
    z[kstart-1] = -z[kstart]
    z[kend] = 2*zsize - z[kend-1]

    # Half level heights
    for k in range(kstart+1, kend):
        zh[k] = 0.5*(z[k-1] + z[k])
    zh[kstart] = 0.
    zh[kend] = zsize

    for k in range(1, kcells):
        dzh[k] = z[k] - z[k-1]
    dzh[kstart-1] = dzh[kstart+1]
    dzhi = 1./dzh

    for k in range(1, kcells-1):
        dz[k] = zh[k+1] - zh[k]
    dz[kstart-1] = dz[kstart]
    dz[kend] = dz[kend-1]
    dzi = 1./dz

    if remove_ghost:
        # Clip off the ghost cells, leaving `ktot` full levels and `ktot+1` half levels.
        z    = z   [kstart:kend]
        dz   = dz  [kstart:kend]
        dzi  = dzi [kstart:kend]

        zh   = zh  [kstart:kend+1]
        dzh  = dzh [kstart:kend+1]
        dzhi = dzhi[kstart:kend+1]

    return dict(
        ktot=ktot,
        zsize=zsize,
        z=z,
        zh=zh,
        dz=dz,
        dzh=dzh,
        dzi=dzi,
        dzhi=dzhi
    )


def refine_grid_for_nesting(z, zh, ratio):
    """
    1. Refine vertical grid by dividing each full level height confined by
       `zh[k]-zh[k+1]` into `ratio` equal steps.
    2. Find full level heights which linearly interpolated result in the half level
       heights from the previous step.

    Parameters:
    ---------
    z : np.ndarray, shape(1,)
        Input full level heights.
    zh : np.ndarray, shape(1,)
        Input half level heights.
    ratio : int
        Refinement ratio.

    Returns:
    -------
    zn : np.ndarray, shape (1,)
        Output full level heights.
    zhn : np.ndarray, shape (1,)
        Output half level heights.
    """
    if ratio == 1:
        return z, zh
    else:
        ktot = z.size
        ktot_n = ktot * ratio

        # Calculate new half level heights.
        zhn = np.zeros(ktot_n + 1)
        for k in range(ktot):
            dzh = (zh[k+1] - zh[k]) / ratio
            for s in range(ratio):
                zhn[k*ratio+s] = zh[k] + s * dzh
        zhn[-1] = zh[-1]

        # Reconstruct full level heights.
        zn = np.zeros(ktot_n)
        zn[0] = 0.5 * (zhn[0] + zhn[1])
        for k in range(1, ktot_n):
            zn[k] = 2 * zhn[k] - zn[k-1]

        return zn, zhn