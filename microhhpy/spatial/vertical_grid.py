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


def calculate_vertical_grid_2nd(z, zsize, remove_ghost=True, dtype=np.float64):
    """
    Calculate vertical grid, identical to definition in MicroHH.

    Arguments:
    ----------
    z : np.ndarray, shape (1,)
        Array with input full level heights, like in `case_input.nc`.
    zsize : float
        Height of domain top.
    remove_ghost : bool, optional
        Clip off the ghost cells, leaving `ktot` full and `ktot+` half levels. Default is True.
    dtype : np.dtype, optional
        Output datatype (np.float32 or np.float64) of arrays.

    Returns:
    --------
    vertical grid : dict
        Dictionary containing grid properties.
    """

    z_in = z.copy()

    ktot = z.size
    kcells = ktot+2

    kstart = 1
    kend = ktot+1

    z = np.zeros(kcells, dtype)
    zh = np.zeros(kcells, dtype)

    dz = np.zeros(kcells, dtype)
    dzh = np.zeros(kcells, dtype)

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
        dzh[k] = z[k] - z[k-1];
    dzh[kstart-1] = dzh[kstart+1]
    dzhi = 1./dzh

    for k in range(1, kcells-1):
        dz[k] = zh[k+1] - zh[k];
    dz[kstart-1] = dz[kstart];
    dz[kend] = dz[kend-1];
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
        z=z,
        zh=zh,
        dz=dz,
        dzh=dzh,
        dzi=dzi,
        dzhi=dzhi
    )


class Vertical_grid_2nd:
    def __init__(self, z_in, zsize_in, remove_ghost=True, dtype=np.float64):
        """
        Calculate vertical grid, identical to definition in MicroHH,
        including one ghost cell at the bottom and top.

        Arguments:
        ----------
        z_in : np.ndarray, shape (1,)
            Array with input full level heights, like in `case_input.nc`.
        zsize : float
            Height of domain top.
        remove_ghost : bool, optional
            Clip off the ghost cells, leaving `ktot` full and `ktot+` half levels. Default is True.
        dtype : np.dtype, optional
            Output datatype (np.float32 or np.float64) of arrays.
        """

        self.zsize = zsize_in

        self.kmax = z_in.size
        self.kcells = self.kmax+2

        self.kstart = 1
        self.kend = self.kmax+1

        self.z = np.zeros(self.kcells, dtype)
        self.zh = np.zeros(self.kcells, dtype)

        self.dz = np.zeros(self.kcells, dtype)
        self.dzh = np.zeros(self.kcells, dtype)

        # Full level heights
        self.z[self.kstart:self.kend] = z_in
        self.z[self.kstart-1] = -self.z[self.kstart]
        self.z[self.kend] = 2*self.zsize - self.z[self.kend-1]

        # Half level heights
        for k in range(self.kstart+1, self.kend):
            self.zh[k] = 0.5*(self.z[k-1] + self.z[k])
        self.zh[self.kstart] = 0.
        self.zh[self.kend] = self.zsize

        for k in range(1, self.kcells):
            self.dzh[k] = self.z[k] - self.z[k-1];
        self.dzh[self.kstart-1] = self.dzh[self.kstart+1]
        self.dzhi = 1./self.dzh

        for k in range(1, self.kcells-1):
            self.dz[k] = self.zh[k+1] - self.zh[k];
        self.dz[self.kstart-1] = self.dz[self.kstart];
        self.dz[self.kend] = self.dz[self.kend-1];
        self.dzi = 1./self.dz

        if remove_ghost:
            """
            Clip off the ghost cells, leaving `ktot` full levels and `ktot+1` half levels.
            """
            self.z    = self.z   [self.kstart:self.kend]
            self.dz   = self.dz  [self.kstart:self.kend]
            self.dzi  = self.dzi [self.kstart:self.kend]

            self.zh   = self.zh  [self.kstart:self.kend+1]
            self.dzh  = self.dzh [self.kstart:self.kend+1]
            self.dzhi = self.dzhi[self.kstart:self.kend+1]
