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
import glob

# Third-party.
import xarray as xr
import numpy as np

# Local library
from microhhpy.logger import logger


def regrid_les(
        fields_3d,
        fields_2d,
        xsize_in,
        ysize_in,
        z_in,
        zh_in,
        itot_in,
        jtot_in,
        xsize_out,
        ysize_out,
        z_out,
        zh_out,
        itot_out,
        jtot_out,
        xstart_out,
        ystart_out,
        path_in,
        path_out,
        float_type,
        name_suffix=''):
    """
    Interpolate 3D LES fields from one LES grid to another and save in binary format.

    Time stamps can be:
    - Single integer
    - List or np.array of integers
    - '*'

    NOTE: uses Xarray for interpolations, so not the fastest approach...

    Arguments:
    ----------
    fields_3d : dict with name : time(s)
        3D fields to interpolate, with time stamps.
    fields_2d : dict with name : time(s)
        2D fields to interpolate, with time stamps.
    xsize_in : float
        Domain size x-direction input grid.
    ysize_in : float
        Domain size y-direction input grid.
    z_in : np.ndarray, shape (1,)
        Input full level heights (m).
    zh_in : np.ndarray, shape (1,)
        Input half level heights (m).
    itot_in : int
        Number of grid points x-direction input grid.
    jtot_in : int
        Number of grid points y-direction input grid.
    ktot_in : int
        Number of grid points z-direction (shared between grids).
    xsize_out : float
        Domain size x-direction output grid.
    ysize_out : float
        Domain size y-direction output grid.
    z_out : np.ndarray, shape (1,)
        Output full level heights (m).
    zh_out : np.ndarray, shape (1,)
        Output half level heights (m).
    itot_out : int
        Number of grid points x-direction output grid.
    jtot_out : int
        Number of grid points y-direction output grid.
    xstart_out : float
        Start x-coordinate output domain in input domain (m).
    ystart_out : float
        Start y-coordinate output domain in input domain (m).
    time : int
        Timestep index to process (e.g. 0 for `0000000`).
    path_in : str
        Path to input binary files.
    path_out : str
        Path to write interpolated binary files.
    float_type : np.float32 or np.float64
        Floating point precision.
    name_suffix : string
        Save output fields with `name_suffix` appended (e.g. `thl_somename.0000000`)

    Returns:
    --------
    None
    """
    logger.info(f'Regridding LES fields from {path_in} to {path_out}.')

    name_suffix = f'_{name_suffix}' if name_suffix else ''

    # Input grid.
    dx_in = xsize_in / itot_in
    dy_in = ysize_in / jtot_in

    x_in  = np.arange(dx_in/2, xsize_in, dx_in)
    xh_in = np.arange(0, xsize_in, dx_in)

    y_in  = np.arange(dy_in/2, ysize_in, dy_in)
    yh_in = np.arange(0, ysize_in, dy_in)

    # Output grid.
    dx_out = xsize_out / itot_out
    dy_out = ysize_out / jtot_out

    x_out  = np.arange(dx_out/2, xsize_out, dx_out)
    xh_out = np.arange(0, xsize_out, dx_out)

    y_out  = np.arange(dy_out/2, ysize_out, dy_out)
    yh_out = np.arange(0, ysize_out, dy_out)

    # Add offset to output grid for easier interpolations.
    x_out += xstart_out
    xh_out += xstart_out

    y_out += ystart_out
    yh_out += ystart_out


    def get_dims(field):
        """
        Get correct in- and output dims for staggered location of `field`.
        """
        dim_x_in = xh_in if field == 'u' else x_in
        dim_x_out = xh_out if field == 'u' else x_out

        dim_y_in = yh_in if field == 'v' else y_in
        dim_y_out = yh_out if field == 'v' else y_out

        dim_z_in = zh_in[:-1] if field == 'w' else z_in
        dim_z_out = zh_out[:-1] if field == 'w' else z_out

        return dim_x_in, dim_x_out, dim_y_in, dim_y_out, dim_z_in, dim_z_out


    def parse_times(field, times):
        """
        Parse input time options.
        """
        if isinstance(times, int):
            times = [times]
        elif isinstance(times, str) and times == '*':
            # Find time stamps.
            files = glob.glob(f'{path_in}/{field}.0*')
            files.sort()
            if len(files) == 0:
                logger.critical(f'Found 0 files for field {field}!')
            times = [int(x.split('.')[-1]) for x in files]

        return times


    """
    Parse 3D fields.
    """
    for field, times in fields_3d.items():

        # Get correct dimensions.
        dim_x_in, dim_x_out, dim_y_in, dim_y_out, dim_z_in, dim_z_out = get_dims(field)

        # Parse input time options.
        times = parse_times(field, times)

        logger.debug(f'Regridding {field} for time(s) {times}.')

        for time in times:

            # Read binary and cast to correct shape.
            data_in = np.fromfile(f'{path_in}/{field}.{time:07d}', dtype=float_type)
            data_in = data_in.reshape((dim_z_in.size, jtot_in, itot_in))

            # Use Xarray for easy interpolations.
            da_in = xr.DataArray(
                data_in,
                coords={"z": dim_z_in, "y": dim_y_in, "x": dim_x_in},
                dims=["z", "y", "x"])

            da_out = da_in.interp(x=dim_x_out, y=dim_y_out, z=dim_z_out, method='nearest', kwargs={'fill_value': 'extrapolate'})

            # Save as binary.
            da_out.values.tofile(f'{path_out}/{field}{name_suffix}.{time:07d}')


    """
    Parse 2D fields.
    """
    for field, times in fields_2d.items():

        # Get correct dimensions.
        dim_x_in, dim_x_out, dim_y_in, dim_y_out, _, _ = get_dims(field)

        # Parse input time options.
        times = parse_times(field, times)

        logger.debug(f'Regridding {field} for time(s) {times}.')

        for time in times:

            # Read binary and cast to correct shape.
            data_in = np.fromfile(f'{path_in}/{field}.{time:07d}', dtype=float_type)
            data_in = data_in.reshape((jtot_in, itot_in))

            # Use Xarray for easy interpolations.
            da_in = xr.DataArray(
                data_in,
                coords={"y": dim_y_in, "x": dim_x_in},
                dims=["y", "x"])

            da_out = da_in.interp(x=dim_x_out, y=dim_y_out, method='nearest', kwargs={'fill_value': 'extrapolate'})

            # Save as binary.
            da_out.values.tofile(f'{path_out}/{field}{name_suffix}.{time:07d}')
