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
from scipy.ndimage import gaussian_filter

# Local library
from microhhpy.logger import logger
from microhhpy.interpolate.interpolate_kernels import Rect_to_curv_interpolation_factors
from microhhpy.interpolate.interpolate_kernels import interpolate_rect_to_curv

def _strip_mask(array):
    if isinstance(array, np.ma.MaskedArray):
        return array.data
    else:
        return array

def _gaussian_filter(fld, sigma):
    for k in range(fld.shape[0]):
        fld[k,:,:] = gaussian_filter(fld[k,:,:], sigma, mode='nearest')


def create_initial_fields_from_era5(
        fields_era,
        lon_era,
        lat_era,
        z_era,
        z_les,
        zh_les,
        rho_les,
        rhoh_les,
        domain,
        correct_div_h,
        sigma_h,
        name_suffix='',
        output_dir='.',
        dtype=np.float64):

    # TO-DO: add input checks.

    proj_pad = domain.proj_pad
    n_pad = domain.n_ghost + 1

    # Strip masked arrays.
    lon_era = _strip_mask(lon_era)
    lat_era = _strip_mask(lat_era)

    # Calculate horizontal interpolation factors at all staggered grid locations.
    # Horizonal only, so `w` factors equal to scalar factors.
    if_u = Rect_to_curv_interpolation_factors(
        lon_era, lat_era, proj_pad.lon_u, proj_pad.lat_u, dtype)

    if_v = Rect_to_curv_interpolation_factors(
        lon_era, lat_era, proj_pad.lon_v, proj_pad.lat_v, dtype)

    if_s = Rect_to_curv_interpolation_factors(
        lon_era, lat_era, proj_pad.lon, proj_pad.lat, dtype)

    #if correct_div_h:
    #    # Correct horizontal divergence to match mean ERA5 vertical velocity with mean LES velocity.

    #    if not all(k in fields_era for k in ['u', 'v', 'w']):
    #        logger.critical('Requested divergence correction, but u, v, or w missing!')

    # LES field with ghost cells!
    fld_les = np.empty((z_les.size, proj_pad.jtot, proj_pad.itot), dtype=dtype)

    # Numpy slice of domain without ghost cells.
    s_int = np.s_[:, n_pad:-n_pad, n_pad:-n_pad]

    # Filter size from `m` to n grid cells.
    sigma_n = int(np.ceil(sigma_h / (6 * proj_pad.dx)))
    if sigma_n > 0:
        logger.debug(f'Using Gaussian filter with sigma = {sigma_n} grid cells')

    for name, fld_era in fields_era.items():
        # Only process scalars.
        if name not in ('u', 'v', 'w'):

            logger.debug(f'Interpolating initial field {name} from ERA to LES')

            # Tri-linear interpolation.
            interpolate_rect_to_curv(
                fld_les,
                fld_era,
                if_s.il,
                if_s.jl,
                if_s.fx,
                if_s.fy,
                z_les,
                z_era,
                dtype)

            # Apply Gaussian filter.
            if sigma_n > 0:
                _gaussian_filter(fld_les, sigma_n)

            # Save 3D field without ghost cells.
            if name_suffix == '':
                f_out = f'{output_dir}/{name}.0000000'
            else:
                f_out = f'{output_dir}/{name}_{name_suffix}.0000000'

            fld_les[s_int].tofile(f_out)