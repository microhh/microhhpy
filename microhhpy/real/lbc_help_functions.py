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


def create_lbc_ds(
        fields,
        time,
        x,
        y,
        z,
        xh,
        yh,
        zh,
        n_ghost,
        n_sponge,
        x_offset=0,
        y_offset=0,
        dtype=np.float64):
    """
    Create an Xarray Dataset with lateral boundary conditions for MicroHH.

    Parameters:
    ----------
    fields : list of str
        Field names to include.
    time : np.ndarray, shape (1,)
        Time steps for the dataset (seconds since start of simulation).
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

    itot = x.size
    jtot = y.size

    nlbc = n_ghost + n_sponge

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

    ds.time.attrs['units'] = 'Seconds since start of simulation'

    return ds


def lbc_ds_to_binary(ds, path, dtype):
    """
    Save an Xarray Dataset with lateral boundary conditions to binary files for MicroHH.

    Parameters:
    ----------
    ds : xr.Dataset
        Dataset with boundary conditions.
    path : str
        Path to save the binary files.
    save_tsteps : bool
        If True, save each time step in a separate binary.
    dtype : np.float32 or np.float64
        Data type for the binary files.
    """

    for var in ds.data_vars:
        #if save_tsteps:
        for t, time in enumerate(ds.time.values):
            ds[var][t].values.astype(dtype).tofile(f'{path}/lbc_{var}.{time:07d}')
        #else:
        #    ds[var].values.astype(dtype).tofile(f'{path}/lbc_{var}.0000000')



def setup_lbc_slices(n_ghost, n_sponge):
    """
    Setup dictionary with slices of lateral boundary conditions in full 3D fields.
    The LBCs contain both the ghost and lateral sponge cells.
    
    Parameters:
    ----------
    n_ghost : int
        Number of ghost cells.
    n_sponge : int
        Number of sponge cells.
    
    Returns:
    -------
    slices : dict
        Dictionary with slices for each boundary condition.
    """

    n_lbc = n_ghost + n_sponge + 1

    slices = dict(
            s_west = np.s_[:, 1:-1, 1:n_lbc],
            s_east = np.s_[:, 1:-1, -n_lbc:-1],
            s_south = np.s_[:, 1:n_lbc, 1:-1],
            s_north = np.s_[:, -n_lbc:-1, 1:-1],

            u_west = np.s_[:, 1:-1, 1:n_lbc+1],
            u_east = np.s_[:, 1:-1, -n_lbc:-1],
            u_south = np.s_[:, 1:n_lbc, 1:-1],
            u_north = np.s_[:, -n_lbc:-1, 1:-1],

            v_west = np.s_[:, 1:-1, 1:n_lbc],
            v_east = np.s_[:, 1:-1, -n_lbc:-1],
            v_south = np.s_[:, 1:n_lbc+1, 1:-1],
            v_north = np.s_[:, -n_lbc:-1, 1:-1],

            w_west = np.s_[:-1, 1:-1, 1:n_lbc],
            w_east = np.s_[:-1, 1:-1, -n_lbc:-1],
            w_south = np.s_[:-1, 1:n_lbc, 1:-1],
            w_north = np.s_[:-1, -n_lbc:-1, 1:-1])

    return slices
