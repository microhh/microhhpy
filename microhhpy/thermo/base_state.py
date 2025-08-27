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

# Local librariy
import microhhpy.constants as cst
from microhhpy.spatial import calc_vertical_grid_2nd
from .base_thermo import exner, virtual_temperature, sat_adjust


def calc_moist_basestate(
        thl,
        qt,
        pbot,
        z,
        zsize,
        dtype=np.float64):
    """
    Calculate moist thermodynamic base state from the
    provided liquid water potential temperature, total
    specific humidity, and surface pressure.

    Parameters:
    -----------
    thl : np.ndarray, shape (1,)
        Liquid water potential temperature on full levels (K).
    qt : np.ndarray, shape (1,)
        Total specific humidity on full levels (kg kg-1).
    pbot : float
        Surface pressure (Pa).
    z : np.ndarray, shape (1,)
        Full level height (m).
    zsize : float
        Domain top height (m).
    dtype : np.dtype
        Floating point precision, np.float32 or np.float64.

    returns:
    --------
    base_state: dict
        Dictionary with base state fields.
    """

    thl_in = thl.copy()
    qt_in = qt.copy()

    gd = calc_vertical_grid_2nd(z, zsize, float_type=dtype, remove_ghost=False)

    z = gd['z']

    kcells = gd['ktot'] + 2
    kstart = 1
    kend = gd['ktot'] + 1

    p = np.zeros(kcells, dtype)
    ph = np.zeros(kcells, dtype)

    rho = np.zeros(kcells, dtype)
    rhoh = np.zeros(kcells, dtype)

    thv = np.zeros(kcells, dtype)
    thvh = np.zeros(kcells, dtype)

    ex = np.zeros(kcells, dtype)
    exh = np.zeros(kcells, dtype)

    # Add ghost cells to input profiles
    thl = np.zeros(kcells, dtype)
    qt  = np.zeros(kcells, dtype)

    thl[kstart:kend] = thl_in
    qt [kstart:kend] = qt_in

    # Calculate surface and domain top values.
    thl0s = thl[kstart] - gd['z'][kstart] * (thl[kstart+1] - thl[kstart]) * gd['dzhi'][kstart+1]
    qt0s  = qt[kstart]  - gd['z'][kstart] * (qt [kstart+1] - qt [kstart]) * gd['dzhi'][kstart+1]

    thl0t = thl[kend-1] + (gd['zh'][kend] - gd['z'][kend-1]) * (thl[kend-1]-thl[kend-2]) * gd['dzhi'][kend-1]
    qt0t  = qt[kend-1]  + (gd['zh'][kend] - gd['z'][kend-1]) * (qt [kend-1]- qt[kend-2]) * gd['dzhi'][kend-1]

    # Set the ghost cells for the reference temperature and moisture
    thl[kstart-1]  = 2.*thl0s - thl[kstart]
    thl[kend]      = 2.*thl0t - thl[kend-1]

    qt[kstart-1]   = 2.*qt0s  - qt[kstart]
    qt[kend]       = 2.*qt0t  - qt[kend-1]

    # Calculate profiles.
    ph[kstart] = pbot
    exh[kstart] = exner(pbot)

    _, ql, qi, _ = sat_adjust(thl0s, qt0s, pbot)

    thvh[kstart] = virtual_temperature(
            exh[kstart], thl0s, qt0s, ql, qi)
    rhoh[kstart] = pbot / (cst.Rd * exh[kstart] * thvh[kstart])

    # Calculate the first full level pressure
    p[kstart] = \
        ph[kstart] * np.exp(-cst.grav * gd['z'][kstart] / (cst.Rd * exh[kstart] * thvh[kstart]))

    for k in range(kstart+1, kend+1):
        # 1. Calculate remaining values (thv and rho) at full-level[k-1]
        ex[k-1] = exner(p[k-1])
        _, ql, qi, _ = sat_adjust(thl[k-1], qt[k-1], p[k-1], ex[k-1])
        thv[k-1] = virtual_temperature(ex[k-1], thl[k-1], qt[k-1], ql, qi)
        rho[k-1] = p[k-1] / (cst.Rd * ex[k-1] * thv[k-1])

        # 2. Calculate pressure at half-level[k]
        ph[k] = ph[k-1] * np.exp(-cst.grav * gd['dz'][k-1] / (cst.Rd * ex[k-1] * thv[k-1]))
        exh[k] = exner(ph[k])

        # 3. Use interpolated conserved quantities to calculate half-level[k] values
        thli = 0.5*(thl[k-1] + thl[k])
        qti  = 0.5*(qt[k-1] + qt[k])
        _, qli, qii, _ = sat_adjust(thli, qti, ph[k], exh[k])

        thvh[k] = virtual_temperature(exh[k], thli, qti, qli, qii)
        rhoh[k] = ph[k] / (cst.Rd * exh[k] * thvh[k])

        # 4. Calculate pressure at full-level[k]
        p[k] = p[k-1] * np.exp(-cst.grav * gd['dzh'][k] / (cst.Rd * exh[k] * thvh[k]))

    p[kstart-1] = 2. * ph[kstart] - p[kstart]

    """
    Strip off the ghost cells, to leave `ktot` full levels and `ktot+1` half levels.
    """
    p = p[kstart:kend]
    ph = ph[kstart:kend+1]

    rho = rho[kstart:kend]
    rhoh = rhoh[kstart:kend+1]

    thv = thv[kstart:kend]
    thvh = thvh[kstart:kend+1]

    ex = ex[kstart:kend]
    exh = exh[kstart:kend+1]

    thl = thl[kstart:kend]
    qt  = qt[kstart:kend]


    return dict(
        thl=thl,
        qt=qt,
        thv=thv,
        thvh=thvh,
        p=p,
        ph=ph,
        exner=ex,
        exnerh=exh,
        rho=rho,
        rhoh=rhoh
    )


def save_moist_basestate(
        base_state,
        file_name):
    """
    Save moist thermodynamic base state to binary file.

    Parameters:
    -----------
    base_state : dict
        Dictionary with base state fields.
    file_name : str
        Path to the output binary file.
    """
    fields = [
        base_state['thl'],
        base_state['qt'],
        base_state['thv'],
        base_state['thvh'],
        base_state['p'],
        base_state['ph'],
        base_state['exner'],
        base_state['exnerh'],
        base_state['rho'],
        base_state['rhoh']
    ]

    bs = np.concatenate(fields)
    bs.tofile(file_name)


def read_moist_basestate(
        file_name,
        dtype=np.float64):
    """
    Read moist thermodynamic base state from binary file.

    Parameters:
    -----------
    file_name : str
        Path to the input binary file.
    dtype : np.dtype
        Floating point precision, np.float32 or np.float64.

    Returns:
    --------
    base_state : dict
        Dictionary with base state fields.
    """

    bs = np.fromfile(file_name, dtype=dtype)

    # This is not at all dangerous.
    n = int((bs.size - 4) / 10)
    sizes = [n, n, n, n+1, n, n+1, n, n+1, n, n+1]
    fields = np.split(bs, np.cumsum(sizes)[:-1])

    return dict(
        thl = fields[0],
        qt = fields[1],
        thv = fields[2],
        thvh = fields[3],
        p = fields[4],
        ph = fields[5],
        exner = fields[6],
        exnerh = fields[7],
        rho = fields[8],
        rhoh = fields[9]
    )


def save_basestate_density(
        rho,
        rhoh,
        file_name):
    """
    Save base state density to binary file.

    Parameters:
    -----------
    rho : np.ndarray, shape (1,)
        Density at full levels.
    rhoh : np.ndarray, shape (1,)
        Density at half levels.
    file_name : str
        Path to the output binary file.
    """

    bs = np.concatenate([rho, rhoh])
    bs.tofile(file_name)


def read_basestate_density(
        file_name,
        dtype=np.float64):
    """
    Read base state density from binary file.

    Parameters:
    -----------
    file_name : str
        Path to the input binary file.
    dtype : np.dtype
        Floating point precision, np.float32 or np.float64.

    Returns:
    --------
    rho : np.ndarray, shape (1,)
        Density at full levels.
    rhoh : np.ndarray, shape (1,)
        Density at half levels.
    """

    bs = np.fromfile(file_name, dtype=dtype)
    n = int((bs.size - 1) / 2)

    return dict(rho=bs[:n], rhoh=bs[n:])



#class Basestate_dry:
#    def __init__(self, th, pbot, z, zsize, remove_ghost=False, dtype=np.float64):
#        """
#        Calculate dry thermodynamic base state from the
#        provided potential temperature and surface pressure.
#
#        Parameters:
#        -----------
#        th : np.ndarray, shape (1,)
#            Potential temperature on full levels (K).
#        pbot : float
#            Surface pressure (Pa).
#        z : np.ndarray, shape (1,)
#            Full level height (m).
#        zsize : float
#            Domain top height (m).
#        remove_ghost : bool, default=False
#            Remove single ghost cells from bottom/top of output arrays.
#        dtype : np.dtype
#            Floating point precision, np.float32 or np.float64.
#        """
#
#        gd = Vertical_grid_2nd(z, zsize, dtype=dtype)
#
#        self.gd = gd
#        self.remove_ghost = remove_ghost
#        self.dtype = dtype
#
#        self.p = np.zeros(gd.kcells)
#        self.ph = np.zeros(gd.kcells)
#
#        self.rho = np.zeros(gd.kcells)
#        self.rhoh = np.zeros(gd.kcells)
#
#        self.ex = np.zeros(gd.kcells)
#        self.exh = np.zeros(gd.kcells)
#
#        # Add ghost cells to input profiles
#        self.th = np.zeros(gd.kcells, dtype)
#        self.thh = np.zeros(gd.kcells, dtype)
#
#        self.th[gd.kstart:gd.kend] = th
#
#        # Extrapolate the input sounding to get the bottom value
#        self.thh[gd.kstart] = self.th[gd.kstart] - gd.z[gd.kstart]*(self.th[gd.kstart+1]-self.th[gd.kstart])*gd.dzhi[gd.kstart+1]
#
#        # Extrapolate the input sounding to get the top value
#        self.thh[gd.kend] = self.th[gd.kend-1] + (gd.zh[gd.kend]-gd.z[gd.kend-1])*(self.th[gd.kend-1]-self.th[gd.kend-2])*gd.dzhi[gd.kend-1]
#
#        # Set the ghost cells for the reference potential temperature
#        self.th[gd.kstart-1] = 2.*self.thh[gd.kstart] - self.th[gd.kstart]
#        self.th[gd.kend]     = 2.*self.thh[gd.kend]   - self.th[gd.kend-1]
#
#        # Interpolate the input sounding to half levels.
#        for k in range(gd.kstart+1, gd.kend):
#            self.thh[k] = 0.5*(self.th[k-1] + self.th[k])
#
#        # Calculate pressure.
#        self.ph[gd.kstart] = pbot
#        self.p [gd.kstart] = pbot * np.exp(-cst.grav * gd.z[gd.kstart] / (cst.Rd * self.thh[gd.kstart] * exner(self.ph[gd.kstart])))
#
#        for k in range(gd.kstart+1, gd.kend+1):
#            self.ph[k] = self.ph[k-1] * np.exp(-cst.grav * gd.dz[k-1] / (cst.Rd * self.th[k-1] * exner(self.p[k-1])))
#            self.p [k] = self.p [k-1] * np.exp(-cst.grav * gd.dzh[k ] / (cst.Rd * self.thh[k ] * exner(self.ph[k ])))
#        self.p[gd.kstart-1] = 2*self.ph[gd.kstart] - self.p[gd.kstart]
#
#        # Calculate density and exner
#        for k in range(0, gd.kcells):
#            self.ex[k]  = exner(self.p[k] )
#            self.rho[k]  = self.p[k]  / (cst.Rd * self.th[k]  * self.ex[k] )
#
#        for k in range(1, gd.kcells):
#            self.exh[k] = exner(self.ph[k])
#            self.rhoh[k] = self.ph[k] / (cst.Rd * self.thh[k] * self.exh[k])
#
#        if remove_ghost:
#            """
#            Strip off the ghost cells, to leave `ktot` full levels and `ktot+1` half levels.
#            """
#            self.p = self.p[gd.kstart:gd.kend]
#            self.ph = self.ph[gd.kstart:gd.kend+1]
#
#            self.rho = self.rho[gd.kstart:gd.kend]
#            self.rhoh = self.rhoh[gd.kstart:gd.kend+1]
#
#            self.ex = self.ex[gd.kstart:gd.kend]
#            self.exh = self.exh[gd.kstart:gd.kend+1]
#
#            self.th = self.th[gd.kstart:gd.kend]
#            self.thh = self.thh[gd.kstart:gd.kend+1]
#
#
#    #def to_binary(self, grid_file):
#    #    """
#    #    Save base state in format required by MicroHH.
#    #    """
#
#    #    if self.remove_ghost:
#    #        rho = self.rho
#    #        rhoh = self.rhoh
#    #    else:
#    #        gd = self.gd
#    #        rho = self.rho[gd.kstart:gd.kend]
#    #        rhoh = self.rhoh[gd.kstart:gd.kend+1]
#
#    #    bs = np.concatenate((rho, rhoh)).astype(self.dtype)
#    #    bs.tofile(grid_file)
