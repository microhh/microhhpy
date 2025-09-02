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
import os

# Third-party.
import netCDF4 as nc4
import xarray as xr
import numpy as np

# Local library


def save_case_input(
        case_name,
        init_profiles,
        tdep_surface=None,
        tdep_ls=None,
        tdep_source=None,
        tdep_chem=None,
        tdep_aerosol=None,
        tdep_radiation=None,
        radiation=None,
        soil=None,
        source=None,
        trajectories=None,
        output_dir=''):
    """
    Create a MicroHH NetCDF input file from dictionaries containing variable data.

    Parameters:
    ----------
    case_name : str
        Name of the simulation case.
    init_profiles : dict
        Initial atmospheric profiles. Must contain 'z' key.
    tdep_surface : dict, optional
        Time-dependent surface variables. Must contain 'time_surface' key.
    tdep_ls : dict, optional
        Time-dependent large-scale variables. Must contain 'time_ls' key.
    tdep_source : dict, optional
        Time-dependent source terms. Must contain 'time_source' key.
    tdep_chem : dict, optional
        Time-dependent chemistry variables. Must contain 'time_chem' key.
    tdep_aerosol : dict, optional
        Time-dependent aerosol concentrations. Must contain 'time_rad' key.
    tdep_radiation : dict, optional
        Time-dependent radiation profiles. Must contain 'time_rad' key.
    radiation : dict, optional
        Non-time-dependent radiation profiles. Must contain 'p_lay' and 'p_lev' keys.
    soil : dict, optional
        Soil profile data. Must contain 'z' key.
    source : dict, optional
        Source location and strength data. Must contain 'sourcelist' key.
    trajectories : dict, optional
        Trajectory data. Each trajectory must contain 'time', 'x', 'y', 'z' keys.
    output_dir : str, optional
        Output directory. Default is '' (current directory).

    Notes:
    -----
    Output file: '{output_dir}/{case_name}_input.nc'
    """

    # Precision of input files can be double for single precision runs.
    float_type = np.float64

    def add_variable(nc_group, name, dims, data, float_type):
        """
        Add variable to NetCDF file (or group), and write data
        """
        if name in nc_group.variables:
            print(f'Warning: variable {name} already exists!')
            return

        var = nc_group.createVariable(name, float_type, dims or ())
        var[:] = data if dims is None else data[:]

    def add_dim(nc_group, name, size):
        """
        Add NetCDF dimension, if it does not already exist.
        """
        if name not in nc_group.dimensions:
            nc_group.createDimension(name, size)

    def is_array(data):
        """
        Check if value is array or scalar
        """
        return isinstance(data, (np.ndarray, xr.DataArray))


    # Define new NetCDF file
    nc_name = os.path.join(output_dir, f'{case_name}_input.nc')
    nc_file = nc4.Dataset(nc_name, mode='w', datamodel='NETCDF4')

    # Create height dimension, and set height coordinate
    add_dim(nc_file, 'z', init_profiles['z'].size)
    add_variable(nc_file, 'z', ('z'), init_profiles['z'], float_type)

    # Create a group called "init" for the initial profiles.
    nc_group_init = nc_file.createGroup('init')

    # Check if any of the timedep groups are active.
    tdep_groups = (tdep_surface, tdep_ls, tdep_source, tdep_chem, tdep_aerosol, tdep_radiation)
    has_tdep = any(tdep_groups)

    # Create a group called `timedep` for the time dependent input.
    if has_tdep:
        nc_group_timedep = nc_file.createGroup('timedep')

        # Set timedep dimensions for radiation/aerosols.
        if tdep_radiation is not None:
            for name, data in tdep_radiation.items():
                if 'lay' in name:
                    add_dim(nc_group_timedep, 'lay', data.shape[1])
                elif 'lev' in name:
                    add_dim(nc_group_timedep, 'lev', data.shape[1])

        if tdep_aerosol is not None:
            for name, data in tdep_aerosol.items():
                if 'bg' in name:
                    add_dim(nc_group_timedep, 'lay', data.shape[1])


    # Set the initial profiles
    for name, data in init_profiles.items():
        # Switch between vector and scalar values
        dims = 'z' if is_array(data) else None
        add_variable(nc_group_init, name, dims, data, float_type)


    # Write the time dependent surface values
    if tdep_surface is not None:
        add_dim(nc_group_timedep, 'time_surface', tdep_surface['time_surface'].size)

        for name, data in tdep_surface.items():
            add_variable(nc_group_timedep, name, ('time_surface'), data, float_type)


    # Write the time dependent atmospheric values
    if tdep_ls is not None:
        add_dim(nc_group_timedep, 'time_ls', tdep_ls['time_ls'].size)

        for name, data in tdep_ls.items():
            dims = ('time_ls') if name == 'time_ls' else ('time_ls', 'z')
            add_variable(nc_group_timedep, name, dims, data, float_type)


    # Write time dependent source strength/location.
    if tdep_source is not None:
        add_dim(nc_group_timedep, 'time_source', tdep_source['time_source'].size)

        for name, data in tdep_source.items():
            add_variable(nc_group_timedep, name, ('time_source'), data, float_type)


    # Write time dependent chemistry variables.
    if tdep_chem is not None:
        nc_group_timedep_chem = nc_file.createGroup('timedep_chem')
        add_dim(nc_group_timedep_chem, 'time_chem', tdep_chem['time_chem'].size)

        for name, data in tdep_chem.items():
            add_variable(nc_group_timedep_chem, name, ('time_chem'), data, float_type)


    # Write time dependent aerosol concentrations.
    if tdep_aerosol is not None:
        add_dim(nc_group_timedep, 'time_rad', tdep_aerosol['time_rad'].size)

        for name, data in tdep_aerosol.items():
            if name == 'time_rad':
                add_variable(nc_group_timedep, name, ('time_rad'), data, float_type)
            elif 'bg' in name:
                # Background profiles on background radiation grid.
                add_variable(nc_group_timedep, name, ('time_rad', 'lay'), data, float_type)
            else:
                # Profiles on LES grid.
                add_variable(nc_group_timedep, name, ('time_rad', 'z'), data, float_type)


    # Non time-dependent radiation profiles (T, h2o, gasses, ...).
    if radiation is not None:
        nc_group_rad = nc_file.createGroup('radiation')

        add_dim(nc_group_rad, 'lay', radiation['p_lay'].size)
        add_dim(nc_group_rad, 'lev', radiation['p_lev'].size)

        for name, data in radiation.items():
            # Switch between vector and scalar values
            if not is_array(data):
                dims = None
            else:
                dims = ('lay') if data.size == radiation['p_lay'].size else ('lev')

            add_variable(nc_group_rad, name, dims, data, float_type)


    # Time dependent radiation profiles.
    if tdep_radiation is not None:
        add_dim(nc_group_timedep, 'time_rad', tdep_radiation['time_rad'].size)

        for name, data in tdep_radiation.items():
            if name == 'time_rad':
                add_variable(nc_group_timedep, name, ('time_rad'), data, float_type)
            elif 'lay' in name or 'bg' in name:
                add_variable(nc_group_timedep, name, ('time_rad', 'lay'), data, float_type)
            elif 'lev' in name:
                add_variable(nc_group_timedep, name, ('time_rad', 'lev'), data, float_type)
            else:
                add_variable(nc_group_timedep, name, ('time_rad', 'z'), data, float_type)


    # Soil profiles.
    if soil is not None:
        nc_group_soil = nc_file.createGroup('soil')
        add_dim(nc_group_soil, 'z', soil['z'].size)

        for name, data in soil.items():
            add_variable(nc_group_soil, name, 'z', data, float_type)


    # Source location/strength.
    if source is not None:

        n_sources = source['sourcelist'].shape[0]
        string_len = source['sourcelist'].shape[1]

        nc_group_source = nc_file.createGroup('source')
        add_dim(nc_group_source, 'emission', n_sources)
        add_dim(nc_group_source, 'string_len', string_len)

        for name, data in source.items():
            if name == 'sourcelist':
                add_variable(nc_group_source, name, ('emission', 'string_len'), data, 'S1')
            elif data.dtype == np.int32:
                add_variable(nc_group_source, name, 'emission', data, np.int32)
            else:
                add_variable(nc_group_source, name, 'emission', data, float_type)


    # Plane/car/... trajectory statistics.
    if trajectories is not None:
        for name,trajectory in trajectories.items():

            nc_group_traj = nc_file.createGroup(f'trajectory_{name}')
            add_dim(nc_group_traj, 'itraj', trajectory['time'].size)

            for var_name in ['time', 'x', 'y', 'z']:
                var = nc_group_traj.createVariable(var_name, float_type, ('itraj'))
                var[:] = trajectory[var_name]

    nc_file.close()
