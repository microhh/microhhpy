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

# Local library
from microhhpy.logger import logger
from microhhpy.interpolate.interpolate_kernels import Rect_to_curv_interpolation_factors
from microhhpy.interpolate.interpolate_kernels import interpolate_rect_to_curv

from .help_functions import strip_mask, gaussian_filter_wrapper, blend_w_to_zero_at_sfc, correct_div_uv


def create_initial_fields_from_era5(
        fields_era,
        lon_era,
        lat_era,
        z_era,
        z,
        zh,
        dzi,
        rho,
        rhoh,
        domain,
        correct_div_h,
        sigma_h,
        name_suffix='',
        output_dir='.',
        dtype=np.float64):

    """
    Inline/lambda help functions.
    """
    def save_field(fld, name):
        """
        Save 3D field without ghost cells to file.
        """
        if name_suffix == '':
            f_out = f'{output_dir}/{name}.0000000'
        else:
            f_out = f'{output_dir}/{name}_{name_suffix}.0000000'

        fld[s_int].tofile(f_out)


    def get_interpolation_factors(name):
        """
        Get interpolation factors for a given field name.
        """
        if name == 'u':
            return if_u
        elif name == 'v':
            return if_v
        else:
            return if_s


    def interpolate(fld_les, name):
        """
        Tri-linear interpolation of ERA5 to LES grid,
        """
        logger.debug(f'Interpolating initial field {name} from ERA to LES')

        if_loc = get_interpolation_factors(name)
        z_loc = zh if name == 'w' else z

        # Tri-linear interpolation from ERA5 to LES grid.
        interpolate_rect_to_curv(
            fld_les,
            fields_era[name],
            if_loc.il,
            if_loc.jl,
            if_loc.fx,
            if_loc.fy,
            z_loc,
            z_era,
            dtype)

        # Apply Gaussian filter.
        if sigma_n > 0:
            gaussian_filter_wrapper(fld_les, sigma_n)

    proj_pad = domain.proj_pad
    n_pad = domain.n_ghost + 1

    # Strip masked arrays.
    lon_era = strip_mask(lon_era)
    lat_era = strip_mask(lat_era)

    # Calculate horizontal interpolation factors at all staggered grid locations.
    # Horizonal only, so `w` factors equal to scalar factors.
    if_u = Rect_to_curv_interpolation_factors(
        lon_era, lat_era, proj_pad.lon_u, proj_pad.lat_u, dtype)

    if_v = Rect_to_curv_interpolation_factors(
        lon_era, lat_era, proj_pad.lon_v, proj_pad.lat_v, dtype)

    if_s = Rect_to_curv_interpolation_factors(
        lon_era, lat_era, proj_pad.lon, proj_pad.lat, dtype)

    # LES field with ghost cells. Needed to make momentum fields divergence free,
    # as this requires one ghost cell in the north- and east-most grid points.
    dims_full = (z.size,  proj_pad.jtot, proj_pad.itot)
    dims_half = (zh.size, proj_pad.jtot, proj_pad.itot)

    fld_full = np.empty(dims_full, dtype=dtype)
    fld_half = np.empty(dims_half, dtype=dtype)

    # Numpy slice of domain without ghost cells.
    s_int = np.s_[:, n_pad:-n_pad, n_pad:-n_pad]

    # Filter size from meters to `n` grid cells.
    sigma_n = int(np.ceil(sigma_h / (6 * proj_pad.dx)))
    if sigma_n > 0:
        logger.debug(f'Using Gaussian filter with sigma = {sigma_n} grid cells')


    """
    Parse the momentum fields separately if divergence correction is requested.
    """
    if correct_div_h:


        if not all(fld in fields_era for fld in ['u', 'v', 'w']):
            logger.critical('Requested divergence correction, but u, v, or w missing!')

        u = np.empty(dims_full, dtype=dtype)
        v = np.empty(dims_full, dtype=dtype)
        w = np.empty(dims_half, dtype=dtype)

        interpolate(u, 'u')
        interpolate(v, 'v')
        interpolate(w, 'w')

        # Interpolated ERA5 `w_ls` sometimes has strange profiles near surface.
        # Blend linearly to zero. This also insures that w at the surface is 0.0 m/s.
        logger.debug('Blending w to zero at the surface.')
        blend_w_to_zero_at_sfc(w, zh, zmax=500)

        # Correct horizontal divergence of u and v.
        logger.debug('Correcting horizontal divergence of u and v from ERA5')
        correct_div_uv(
            u,
            v,
            w,
            rho,
            rhoh,
            dzi,
            proj_pad.x,
            proj_pad.y,
            proj_pad.xsize,
            proj_pad.ysize,
            n_pad)

        save_field(u, 'u')
        save_field(v, 'v')
        save_field(w, 'w')


    """
    Parse remaining fields.
    """
    exclude_fields = ('u', 'v', 'w') if correct_div_h else ()

    for name, fld_era in fields_era.items():
        if name not in exclude_fields:

            fld = fld_half if name == 'w' else fld_full
            
            # Tri-linear interpolation from ERA5 to LES grid.
            interpolate(fld, name)

            # Save 3D field without ghost cells in binary format.
            save_field(fld, name)