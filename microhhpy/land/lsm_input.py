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
import xarray as xr
import numpy as np

# Local library
from microhhpy.logger import logger


class Land_surface_input:
    def __init__(
        self,
        itot,
        jtot,
        ktot,
        exclude_fields=[],
        exclude_soil=False,
        exclude_veg=False,
        debug=False,
        float_type=np.float64):
        """
        Data structure for the required input for the MicroHH LSM.

        Parameters:
        -----------
            itot : int
                Number of grid points in x-direction.
            jtot : int
                Number of grid points in y-direction.
            ktot : int
                Number of soil layers.
            exclude_fields : list(str)
                Exclude fields from being saved as binary.
            exlude_soil : bool, default=False
                Exclude soil fields.
            exlude_veg : bool, default=False
                Exclude vegetation fields.
            debug : bool, default: False
                Switch to fill the emtpy fields with a large negative number,
                to ensure that every grid point is initialized before saving.
            float_type : np.float32 or np.float64
                Floating point precision used by MicroHH.

        Returns:
        --------
            None
        """

        self.itot = itot
        self.jtot = jtot
        self.ktot = ktot
        self.debug = debug

        # List of fields which are written to the binary input files for MicroHH
        self.fields_2d = [
                'c_veg', 'z0m', 'z0h', 'gD', 'lai',
                'alb_dir', 'alb_dif',
                'rs_veg_min', 'rs_soil_min',
                'lambda_stable', 'lambda_unstable',
                'cs_veg', 'water_mask', 't_bot_water',
                'index_veg']
        self.fields_3d = [
                't_soil', 'theta_soil', 'index_soil', 'root_frac']

        self.exclude_fields = exclude_fields
        if exclude_soil:
            self.exclude_fields += ['t_soil', 'theta_soil', 'index_soil']
        if exclude_veg:
            self.exclude_fields += self.fields_2d + ['root_frac']

        # Horizonal grid (cell centers)
        self.x = np.zeros(itot, dtype=float_type)
        self.y = np.zeros(jtot, dtype=float_type)

        # Lat/lon coordinates of each grid point (not used by LES)
        self.lon = np.zeros((jtot, itot), dtype=float_type)
        self.lat = np.zeros((jtot, itot), dtype=float_type)

        # Create empty 2D/3D fields
        def get_dtype(name):
            return float_type
            #return np.uint16 if 'index' in name else float_type

        for fld in self.fields_2d:
            if fld not in self.exclude_fields:
                setattr(self, fld, np.zeros((jtot, itot), dtype=get_dtype(fld)))

        for fld in self.fields_3d:
            if fld not in self.exclude_fields:
                setattr(self, fld, np.zeros((ktot, jtot, itot), dtype=get_dtype(fld)))

        if debug:
            # Init all values at large negative number
            for fld in self.fields_2d + self.fields_3d:
                if fld not in self.exclude_fields:
                    data = getattr(self, fld)
                    data[:] = -1 if 'index' in fld else 1e12


    def check(self):
        """
        Check if all values have been set (only for `debug=True`).
        """

        if not self.debug:
            logger.warning('Cannot check land-surface input value with `debug=False`.')
        else:
            not_initialised = []
            for fld in self.fields_2d + self.fields_3d:
                if fld in self.exclude_fields:
                    continue

                data = getattr(self, fld)
                if 'index' in fld and np.any(data == -1):
                    not_initialised.append(fld)
                elif np.any(data > 1e11):
                    not_initialised.append(fld)
            
            if len(not_initialised) > 0:
                fld_str = ', '.join(not_initialised)
                logger.warning(f'Uninitialised land-surface fields: {fld_str}')


    def to_binaries(self, path='.', allow_overwrite=False):
        """
        Write all required MicroHH input fields in binary format

        Arguments:
            path : str
                File path of the output.
            allow_overwrite : bool, default: False
                Allow overwriting of existing files.
        """
        if self.debug:
            self.check()

        logger.info('Writing land-surface binaries')

        def save_bin(data, bin_file):
            if not allow_overwrite and os.path.exists(bin_file):
                logger.critical(f'Binary file \"{bin_file}\" already exists!')
            else:
                data.tofile(bin_file)

        for fld in self.fields_2d + self.fields_3d:
            if fld not in self.exclude_fields:
                logger.debug(f'Saving {fld}.0000000')
                data = getattr(self, fld)
                save_bin(data, f'{os.path.join(path, fld)}.0000000')


    def to_netcdf(self, nc_file, allow_overwrite=False):
        """
        Save MicroHH input to NetCDF file. Not used by MicroHH, but useful for visualisation/debug.

        Arguments:
            nc_file : str
                Name of output NetCDF file.
            allow_overwrite : bool, default: False
                Allow overwriting of existing files.
        """

        if not allow_overwrite and os.path.exists(nc_file):
            logger.critical(f'NetCDF file \"{nc_file}\" already exists and `allow_overwrite=False`.')

        coords = {'x': self.x, 'y': self.y}

        data_vars = {
            'lon': (['y', 'x'], self.lon),
            'lat': (['y', 'x'], self.lat)}

        for field in self.fields_2d:
            if field not in self.exclude_fields:
                data_vars[field] = (['y', 'x'], getattr(self, field))

        for field in self.fields_3d:
            if field not in self.exclude_fields:
                data_vars[field] = (['z', 'y', 'x'], getattr(self, field))

        self.ds = xr.Dataset(
            data_vars=data_vars,
            coords=coords)

        self.ds.to_netcdf(nc_file)