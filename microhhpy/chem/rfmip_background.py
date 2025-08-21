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
from importlib import resources

# Third-party.
import xarray as xr
import numpy as np

# Local library
from microhhpy.logger import logger


def get_rfmip_species(lat, lon, exp):
    """
    Get background chemical concentration from RFMIP intercomparison.
    
    Parameters:
    -----------
    lat : float
        Target latitude
    lon : float
        Target longitude
    exp : int
        RFMIP experiment, 0=present day, 1=pre-industrial, 2=..
    
    Returns:
    --------
    rfmip_dict : dict
        Dictionary with RFMIP species.
    """

    def get_data_file(file_name):
        with resources.as_file(resources.files('microhhpy.data') / file_name) as path:
            return path

    nc_file = get_data_file('rfmip_species.nc')
    ds = xr.open_dataset(nc_file)

    # Select nearest location based on simple lat/lon distance.
    site_idx = ((ds.lon - lon)**2 + (ds.lat - lat)**2).argmin()
    dsn = ds.isel(site=site_idx, expt=exp)

    exp_label = str(dsn.expt_label.values)
    lon = float(dsn.lon)
    lat = float(dsn.lat)
    logger.info(f'Reading RFMIP species for lon={lon}º, lat={lat}º, experiment={exp_label}')

    def get_value(ds, var):
        """
        Get variable `var` from `ds`, and scale with its units
        """
        base_var = float(ds[var])
        unit = float(ds[var].units)
        return base_var * unit

    dict_out = {
        'co2'     : get_value(dsn, 'carbon_dioxide_GM'),
        'ch4'     : get_value(dsn, 'methane_GM'),
        'n2o'     : get_value(dsn, 'nitrous_oxide_GM'),
        'n2'      : get_value(dsn, 'nitrogen_GM'),
        'o2'      : get_value(dsn, 'oxygen_GM'),
        'co'      : get_value(dsn, 'carbon_monoxide_GM'),
        'ccl4'    : get_value(dsn, 'carbon_tetrachloride_GM'),
        'cfc11'   : get_value(dsn, 'cfc11_GM'),
        'cfc12'   : get_value(dsn, 'cfc12_GM'),
        'hcfc22'  : get_value(dsn, 'hcfc22_GM'),
        'hfc143a' : get_value(dsn, 'hfc143a_GM'),
        'hfc125'  : get_value(dsn, 'hfc125_GM'),
        'hfc23'   : get_value(dsn, 'hfc23_GM'),
        'hfc32'   : get_value(dsn, 'hfc32_GM'),
        'hfc134a' : get_value(dsn, 'hfc134a_GM'),
        'cf4'     : get_value(dsn, 'cf4_GM'),
        'no2'     : 0.}

    return dict_out