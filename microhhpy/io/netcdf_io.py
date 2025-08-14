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
import xarray as xr
import netCDF4 as nc4

# Local library

def xr_open_groups(nc_file, groups=None):
    """
    Read all (or selection of) NetCDF groups from NetCDF
    file using xarray. If `groups=None`, all NetCDF groups
    are loaded.

    Parameters:
    -----------
    nc_file : str
        Full path to NetCDF file.
    groups : list
        List of NetCDF groups to read. Optional, default (None) reads all available groups.

    Returns:
    -------
    xarray dataset.
    """

    nc = nc4.Dataset(nc_file)
    if groups is None:
        groups = list(nc.groups)

    # Check of NetCDF file has meaningful time units.
    if nc.variables['time'].units == 'seconds since start':
        decode_times = False
    else:
        decode_times = True

    nc.close()

    # Read all groups into a single Dataset.
    dss = [xr.open_dataset(nc_file, decode_times=decode_times)]
    for group in groups:
        dss.append(xr.open_dataset(nc_file, group=group, decode_times=decode_times))
    return xr.merge(dss)