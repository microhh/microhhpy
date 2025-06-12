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

from .global_help_functions import gaussian_filter_wrapper, blend_w_to_zero_at_sfc
from .global_help_functions import correct_div_uv, calc_w_from_uv, check_divergence


def setup_interpolations(
        lon_era,
        lat_era,
        proj_pad,
        dtype):
    """
    Calculate horizontal interpolation factors at all staggered grid locations.
    Horizonal only, so `w` factors equal to scalar factors.

    Arguments:
    ---------
    lon_era : np.ndarray, shape (2,)
        Longitudes of ERA5 grid points.
    lat_era : np.ndarray, shape (2,)
        Latitudes of ERA5 grid points.
    proj_pad : microhhpy.spatial.Projection instance
        Projection including ghost cells.
    dtype : numpy float type, optional
        Data type for output arrays. Defaults to `np.float64`.
    """

    if_u = Rect_to_curv_interpolation_factors(
        lon_era, lat_era, proj_pad.lon_u, proj_pad.lat_u, dtype)

    if_v = Rect_to_curv_interpolation_factors(
        lon_era, lat_era, proj_pad.lon_v, proj_pad.lat_v, dtype)

    if_s = Rect_to_curv_interpolation_factors(
        lon_era, lat_era, proj_pad.lon, proj_pad.lat, dtype)

    return dict(
        u = if_u,
        v = if_v,
        s = if_s
    )


def get_interpolation_factors(
        factors,
        name):
    """
    Get interpolation factors at correct staggered grid location.

    Arguments:
    ---------
    factors : dict
        Dictionary with interpolation factors
    name : str
        Name of field.
    """

    if name == 'u':
        return factors['u']
    elif name == 'v':
        return factors['v']
    else:
        return factors['s']


def process_momentum(
        u,
        v,
        w,
        rho,
        rhoh,
        zh,
        dz,
        dzi,
        domain,
        n_pad
        ):
    """
    Process interpolated momentum fields:
    1. Blend w to zero to surface over a certain (500 m) depth.
    2. Correct horizontal divergence of u and v to match subsidence in LES to ERA5.
    3. Calculate new vertical velocity w to ensure that the fields are divergence free.
    4. Check resulting divergence.

    Arguments:
    ---------
    u : np.ndarray, shape (3,)
        Zonal wind field.
    v : np.ndarray, shape (3,)
        Meridional wind field
    w : np.ndarray, shape (1,)
        Vertical wind field.
    rho : np.ndarray, shape (1,)
        Basestate air density at LES full levels.
    rhoh : np.ndarray, shape (1,)
        Basestate air density at LES half levels.
    zh : np.ndarray, shape (1,)
        LES half-level heights.
    dz : np.ndarray, shape (1,)
        Vertical grid spacing full levels.
    dzi : np.ndarray, shape (1,)
        Inverse of `dz`.
    domain : Domain instance
        microhhpy.domain.Domain instance, needed for spatial transforms.
    n_pad : int
        Number of ghost cells including padding.

    Returns:
        None
    """
    proj_pad = domain.proj_pad
    
    # Interpolated ERA5 `w_ls` sometimes has strange profiles near surface.
    # Blend linearly to zero. This also insures that w at the surface is 0.0 m/s.
    logger.debug(f'Blending w to zero at the surface.')
    blend_w_to_zero_at_sfc(w, zh, zmax=500)

    # Correct horizontal divergence of u and v.
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

    # NOTE: correcting the horizontal divergence only ensures that the _mean_ vertical velocity
    # is correct. We still need to calculate a new vertical velocity to ensure that the wind
    # fields are divergence free at a grid point level.
    logger.debug(f'Calculating new vertical velocity to create divergence free wind fields.')
    calc_w_from_uv(
        w,
        u,
        v,
        rho,
        rhoh,
        dz,
        domain.dxi,
        domain.dyi,
        domain.istart_pad,
        domain.iend_pad,
        domain.jstart_pad,
        domain.jend_pad,
        dz.size)

    # Check! 
    div_max, i, j, k = check_divergence(
        u,
        v,
        w,
        rho,
        rhoh, 
        domain.dxi,
        domain.dyi,
        dzi,
        domain.istart_pad,
        domain.iend_pad,
        domain.jstart_pad,
        domain.jend_pad,
        dz.size)
    logger.debug(f'Maximum divergence in LES domain: {div_max:.3e} at i={i}, j={j}, k={k}')


def create_fields_from_era5(
        fields_era,
        lon_era,
        lat_era,
        z_era,
        z,
        zh,
        dz,
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
    Generate initial LES fields by interpolating ERA5 data onto the LES grid.

    If requested, it also corrects the horizontal divergence of the wind fields,
    to ensure that the domain mean vertical velocity in LES matches the
    subsidence velocity of ERA5, while still being divergence free at the grid point level.

    The generated 3D fields are saved in binary format without ghost cells to `output_dir`.

    Arguments:
    ---------
    fields_era : dict of np.ndarray
        Dictionary with ERA5 fields.
    lon_era : np.ndarray, shape (2,)
        Longitudes of ERA5 grid points.
    lat_era : np.ndarray, shape (2,)
        Latitudes of ERA5 grid points.
    z_era : np.ndarray, shape (3,)
        Heights of ERA5 full levels.
    z : np.ndarray, shape (1,)
        LES full-level heights.
    zh : np.ndarray, shape (1,)
        LES half-level heights.
    dz : np.ndarray, shape (1,)
        LES vertical grid spacing.
    dzi : np.ndarray, shape (1,)
        Inverse of `dz`.
    rho : np.ndarray, shape (1,)
        Basestate air density at LES full levels.
    rhoh : np.ndarray, shape (1,)
        Basestate air density at LES half levels.
    domain : Domain instance
        microhhpy.domain.Domain instance, needed for spatial transforms.
    correct_div_h : bool
        If True, apply horizontal divergence correction to match target subsidence.
    sigma_h : float
        Width of Gaussian filter for smoothing the interpolated fields (in horizontal distance units).
    name_suffix : str, optional
        String to append to output filenames.
    output_dir : str, optional
        Directory to write the output files.
    dtype : numpy float type, optional
        Data type for output arrays. Defaults to `np.float64`.

    Returns:
    -------
    None
    """

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


    def interpolate(fld_les, name):
        """
        Tri-linear interpolation of ERA5 to LES grid,
        """
        logger.debug(f'Interpolating initial field {name} from ERA to LES')

        if_loc = get_interpolation_factors(ip_facs, name)
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
    n_pad = domain.n_pad

    """
    Setup interpolation factors.
    """
    ip_facs = setup_interpolations(lon_era, lat_era, proj_pad, dtype)

    """
    Define fields with ghost cells, needed to make u,v,w, divergence free.
    """
    dims_full = (z.size,  proj_pad.jtot, proj_pad.itot)
    dims_half = (zh.size, proj_pad.jtot, proj_pad.itot)

    fld_full = np.empty(dims_full, dtype=dtype)
    fld_half = np.empty(dims_half, dtype=dtype)

    # Numpy slice with interior of domain.
    s_int = np.s_[:, n_pad:-n_pad, n_pad:-n_pad]

    
    """
    Setup spatial filtering.
    """
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

        process_momentum(
            u,
            v,
            w,
            rho,
            rhoh,
            zh,
            dz,
            dzi,
            domain,
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


def create_lbcs_from_era5(
        fields_era,
        lon_era,
        lat_era,
        z_era,
        time,
        start_date,
        z,
        zh,
        dz,
        dzi,
        rho,
        rhoh,
        domain,
        n_sponge,
        correct_div_h,
        sigma_h,
        output_dir='.',
        dtype=np.float64):
    """
    Docstring TODO
    """

    """
    Inline/lambda help functions.
    """
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
        logger.debug(f'Interpolating field {name} from ERA to LES')

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


    n_pad = domain.n_pad
    proj_pad = domain.proj_pad


    """
    Setup Xarray dataset with correction dimensions/coordinates of lateratal boundary conditions.
    """
    lbc_ds = create_lbc_ds(
        list(fields_era.keys()),
        domain.x,
        domain.y,
        z,
        domain.xh,
        domain.yh,
        zh,
        time,
        start_date,
        domain.n_ghost,
        n_sponge,
        dtype=dtype)


    """
    Numpy slices of all boundary fields.
    """
    slices = dict(
            ss_west = np.s_[:, 1:-1, 1:n_pad],
            ss_east = np.s_[:, 1:-1, -n_pad:-1],
            ss_south = np.s_[:, 1:n_pad, 1:-1],
            ss_north = np.s_[:, -n_pad:-1, 1:-1],

            su_west = np.s_[:, 1:-1, 1:n_pad+1],
            su_east = np.s_[:, 1:-1, -n_pad:-1],
            su_south = np.s_[:, 1:n_pad, 1:-1],
            su_north = np.s_[:, -n_pad:-1, 1:-1],

            sv_west = np.s_[:, 1:-1, 1:n_pad],
            sv_east = np.s_[:, 1:-1, -n_pad:-1],
            sv_south = np.s_[:, 1:n_pad+1, 1:-1],
            sv_north = np.s_[:, -n_pad:-1, 1:-1],

            sw_west = np.s_[:-1, 1:-1, 1:n_pad],
            sw_east = np.s_[:-1, 1:-1, -n_pad:-1],
            sw_south = np.s_[:-1, 1:n_pad, 1:-1],
            sw_north = np.s_[:-1, -n_pad:-1, 1:-1])


    """
    Calculate horizontal interpolation factors at all staggered grid locations.
    Horizonal only, so `w` factors equal to scalar factors.
    """
    if_u = Rect_to_curv_interpolation_factors(
        lon_era, lat_era, proj_pad.lon_u, proj_pad.lat_u, dtype)

    if_v = Rect_to_curv_interpolation_factors(
        lon_era, lat_era, proj_pad.lon_v, proj_pad.lat_v, dtype)

    if_s = Rect_to_curv_interpolation_factors(
        lon_era, lat_era, proj_pad.lon, proj_pad.lat, dtype)

    
    """
    Fields are interpolated over full 3D fields, not just the boundaries.
    This is a bit wasteful, but a lot easier with the spatial filtering and all.
    """
    dims_full = (z.size,  proj_pad.jtot, proj_pad.itot)
    dims_half = (zh.size, proj_pad.jtot, proj_pad.itot)

    fld_full = np.empty(dims_full, dtype=dtype)
    fld_half = np.empty(dims_half, dtype=dtype)

    # Filter size from meters to `n` grid cells.
    sigma_n = int(np.ceil(sigma_h / (6 * proj_pad.dx)))
    if sigma_n > 0:
        logger.debug(f'Using Gaussian filter with sigma = {sigma_n} grid cells')


    """
    Process all time steps.
    """
    for t in range(time.size):
        logger.debug(f'Processing time step {t+1}/{time.size}')