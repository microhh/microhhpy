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
from enum import Enum
import os

# Third-party.
from attr.filters import exclude
import numpy as np
from scipy.interpolate import NearestNDInterpolator

# Local library
from microhhpy.land import ifs_vegetation
from microhhpy.logger import logger

from .corine_landuse import read_corine, corine_to_ifs_ids
from .lcc_landuse import read_lcc, lcc_to_ifs_ids

from .lsm_input import Land_surface_input
from .ifs_vegetation import get_ifs_vegetation_lut, calc_root_fraction_3d


class Lu_src(Enum):
    CORINE_100M = "corine_100m"
    LCC_100M = "lcc_100m"


def create_land_surface_input(
    lon,
    lat,
    z_soil,
    land_use_tiff,
    land_use_source = 'lcc_100m',
    save_binaries=True,
    output_dir='.',
    save_netcdf=False,
    netcdf_file='',
    float_type=np.float64):

    # From string -> enum, less brittle..
    try:
        land_use = Lu_src(land_use_source)
    except ValueError:
        logger.critical(f'Invalid land use source: "{land_use_source}".')

    # Lookup table with IFS vegetation properties (z0m, c_veg, ..).
    ifs_vegetation = get_ifs_vegetation_lut()


    """
    Read GeoTIFF and translate original to IFS vegetation types.
    """
    if land_use == Lu_src.CORINE_100M:
        logger.debug('Reading Corine GeoTIFF.')

        # Read GeoTIFF.
        da_lu, _ = read_corine(
            land_use_tiff,
            lon.min()-0.25, lon.max()+0.25,
            lat.min()-0.25, lat.max()+0.25)

        # Translate Corine land-use index to IFS index
        ifs_index = corine_to_ifs_ids(da_lu)

    elif land_use == Lu_src.LCC_100M:
        logger.debug('Reading 100 m LCC GeoTIFF.')

        da_lu = read_lcc(
            land_use_tiff,
            lon.min()-0.25, lon.max()+0.25,
            lat.min()-0.25, lat.max()+0.25)

        # Translate Corine land-use index to IFS index
        ifs_index = lcc_to_ifs_ids(da_lu)


    """
    Interpolate (NN) vegetation types onto LES grid.
    """
    if da_lu.lon.ndim == 1:
        # LCC
        lonlat = np.zeros((da_lu.lon.size*da_lu.lat.size, 2))
        lon2d, lat2d = np.meshgrid(da_lu.lon, da_lu.lat)
        lonlat[:,0] = lon2d.flatten()
        lonlat[:,1] = lat2d.flatten()
    else:
        # Corine
        lonlat = np.zeros((da_lu.lon.size, 2))
        lonlat[:,0] = da_lu.lon.values.flatten()
        lonlat[:,1] = da_lu.lat.values.flatten()

    interp = NearestNDInterpolator(lonlat, ifs_index.flatten())
    ifs_index_nn = interp(lon, lat).astype(np.int32)


    """
    Create input fields for MicroHH.
    """
    jtot, itot = lon.shape
    exclude = ['t_bot_water', 't_soil', 'theta_soil', 'index_soil']

    lsm_input = Land_surface_input(
        itot, jtot, ktot=z_soil.size,
        exclude_fields=exclude,
        float_type=float_type, debug=True)

    lsm_input.lon = lon
    lsm_input.lat = lat

    # Coefficients used to calculate root fraction.
    a_r = np.zeros_like(lon, dtype=np.float32)
    b_r = np.zeros_like(lon, dtype=np.float32)

    for code in np.unique(ifs_index_nn):
        mask = (ifs_index_nn == code)

        if code == -9999:
            print('WARNING: missing data in land-use dataset. Setting to grass land...')
            code = 1

        if code in [13, 14, 15, 19]:
            lsm_input.c_veg[mask] = 0
            lsm_input.z0m[mask] = 0.001
            lsm_input.z0h[mask] = 0.0001
            lsm_input.gD[mask] = 0
            lsm_input.lai[mask] = 0
            lsm_input.rs_veg_min[mask] = 0
            lsm_input.rs_soil_min[mask] = 0
            lsm_input.lambda_stable[mask] = 0
            lsm_input.lambda_unstable[mask] = 0
            lsm_input.cs_veg[mask] = 0
            lsm_input.alb_dir[mask] = 0.08
            lsm_input.alb_dif[mask] = 0.08
            lsm_input.water_mask[mask] = 1
            lsm_input.index_veg[mask] = 13
            a_r[mask] = -1
            b_r[mask] = -1
        else:
            lsm_input.c_veg[mask] = ifs_vegetation['c_veg'][code]
            lsm_input.z0m[mask] = ifs_vegetation['z0m'][code]
            lsm_input.z0h[mask] = ifs_vegetation['z0h'][code]
            lsm_input.gD[mask] = ifs_vegetation['gD'][code]
            lsm_input.lai[mask] = ifs_vegetation['lai'][code]
            lsm_input.rs_veg_min[mask] = ifs_vegetation['rs_min'][code]
            lsm_input.rs_soil_min[mask] = 50
            lsm_input.lambda_stable[mask] = ifs_vegetation['lambda_s'][code]
            lsm_input.lambda_unstable[mask] = ifs_vegetation['lambda_us'][code]
            lsm_input.cs_veg[mask] = 0
            lsm_input.alb_dir[mask] = ifs_vegetation['albedo'][code]
            lsm_input.alb_dif[mask] = ifs_vegetation['albedo'][code]
            lsm_input.water_mask[mask] = 0
            lsm_input.index_veg[mask] = code
            a_r[mask] = ifs_vegetation['a_r'][code]
            b_r[mask] = ifs_vegetation['b_r'][code]

    """
    Calculate root fraction from a,b coefficients.
    """
    ktot = z_soil.size
    zh_soil = np.zeros(ktot + 1)
    for k in range(ktot-1, -1, -1):
        zh_soil[k] = zh_soil[k+1] + 2*(z_soil[k] - zh_soil[k+1])

    calc_root_fraction_3d(lsm_input.root_frac, a_r, b_r, zh_soil)


    """
    Save in NetCDF for visualisation.
    """
    if save_netcdf:
        file_path = os.path.join(output_dir, netcdf_file)
        logger.debug(f'Saving {file_path}')
        lsm_input.to_netcdf(file_path, allow_overwrite=True)


    """
    Save in binary format for MicroHH input.
    """
    if save_binaries:
        logger.debug(f'Saving land-surface binaries in {output_dir}')
        lsm_input.to_binaries(path=output_dir, allow_overwrite=True)


    return lsm_input