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
import xarray as xr

# Local library
from microhhpy.logger import logger


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