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
import numpy as np
from datetime import datetime

from concurrent.futures import ThreadPoolExecutor

# Local library
from microhhpy.logger import logger
from microhhpy.interp.interp_kernels import Rect_to_curv_interpolation_factors
from microhhpy.interp.interp_kernels import interp_rect_to_curv_kernel
from microhhpy.spatial import calc_vertical_grid_2nd

# Local directory
from .global_help_functions import gaussian_filter_wrapper
from .global_help_functions import correct_div_uv
from .numba_kernels import block_perturb_field, blend_w_to_zero_at_sfc, calc_w_from_uv, check_divergence
from .lbc_help_functions import create_lbc_ds, setup_lbc_slices, lbc_ds_to_binary


def setup_interpolations(
        lon_in,
        lat_in,
        proj_pad,
        float_type):
    """
    Calculate horizontal interpolation factors at all staggered grid locations.
    Horizonal only, so `w` factors equal to scalar factors.

    Parameters:
    ----------
    lon_in : np.ndarray, shape (2,)
        Longitudes of grid points.
    lat_in : np.ndarray, shape (2,)
        Latitudes of grid points.
    proj_pad : microhhpy.spatial.Projection instance
        Spatial projection.
    float_type : numpy float type, np.float32 or np.float64
        Floating point precision.

    Returns:
    -------
    tuple of `Rect_to_curv_interpolation_factors` instances at (u, v, s) locations.
    """

    if_u = Rect_to_curv_interpolation_factors(
        lon_in, lat_in, proj_pad.lon_u, proj_pad.lat_u, float_type)

    if_v = Rect_to_curv_interpolation_factors(
        lon_in, lat_in, proj_pad.lon_v, proj_pad.lat_v, float_type)

    if_s = Rect_to_curv_interpolation_factors(
        lon_in, lat_in, proj_pad.lon, proj_pad.lat, float_type)

    return if_u, if_v, if_s


def save_3d_field(
        fld,
        name,
        name_suffix,
        n_pad,
        output_dir):
    """
    Save 3D field without ghost cells to file.
    """
    if name_suffix == '':
        f_out = f'{output_dir}/{name}.0000000'
    else:
        f_out = f'{output_dir}/{name}_{name_suffix}.0000000'

    fld[:, n_pad:-n_pad, n_pad:-n_pad].tofile(f_out)


def parse_scalar(
    lbc_ds,
    name,
    name_suffix,
    t,
    time,
    fld_in,
    z_in,
    z_les,
    ip_fac,
    lbc_slices,
    sigma_n,
    perturb_size,
    perturb_amplitude,
    perturb_max_height,
    clip_at_zero,
    domain,
    kstart_buffer,
    output_dir,
    float_type):
    """
    Parse a single scalar for a single time step.
    Creates both the initial field (t=0 only) and lateral boundary conditions.

    Parameters
    ----------
    lbc_ds : xarray.Dataset
        Dataset containing lateral boundary condition (LBC) fields.
    name : str
        Name of the scalar field.
    name_suffix : str
        Suffix to append to the output variable name.
    t : int
        Timestep index.
    time : int
        Time in seconds since start of experiment.
    fld_in : np.ndarray, shape (3,)
        Scalar field.
    z_in : np.ndarray, shape (3,)
        Model level heights.
    z_les : np.ndarray, shape (1,)
        Full level heights LES.
    ip_fac : `Rect_to_curv_interpolation_factors` instance.
        Interpolation factors.
    lbc_slices : dict
        Dictionary with Numpy slices for each LBC.
    sigma_n : int
        Width Gaussian filter kernel in LES grid points.
    perturb_size : int
        Perturb 3D fields in blocks of certain size (equal in all dimensions).
    perturb_amplitude : dict
        Dictionary with perturbation amplitudes for each field.
    clip_at_zero : list(str)
        List of fields to clip at >= 0.
    domain : Domain instance
        Domain information.
    kstart_buffer : int
        Start index of the buffer in the vertical direction.
    output_dir : str
        Output directory.
    float_type : np.float32 or np.float64
        Floating point precision.

    Returns
    -------
    None
    """
    logger.debug(f'Processing field {name} at t={time}.')

    # Short-cuts
    n_pad = domain.n_pad

    # Keep creation of 3D field here, for parallel/async exectution..
    fld_les = np.empty((z_les.size, domain.proj_pad.jtot, domain.proj_pad.itot), dtype=float_type)

    # Lazily load data in case xarray data is provided.
    if isinstance(fld_in, xr.DataArray):
        fld_in = fld_in.values

    # Tri-linear interpolation from host to LES grid.
    interp_rect_to_curv_kernel(
        fld_les,
        fld_in,
        ip_fac.il,
        ip_fac.jl,
        ip_fac.fx,
        ip_fac.fy,
        z_les,
        z_in,
        float_type)

    # Apply Gaussian filter.
    if sigma_n > 0:
        gaussian_filter_wrapper(fld_les, sigma_n)

    # Apply perturbation to the field.
    if name in perturb_amplitude.keys() and perturb_size > 0:
        block_perturb_field(fld_les, z_les, perturb_size, perturb_amplitude[name], perturb_max_height)

    # Remove negative values from fields.
    if name in clip_at_zero:
        fld_les[fld_les < 0] = 0.

    # Save 3D field without ghost cells i`n binary format as initial/restart file.
    if t == 0:
        save_3d_field(fld_les, name, name_suffix, n_pad, output_dir)

    # Save lateral boundaries.
    for loc in ('west', 'east', 'south', 'north'):
        lbc_slice = lbc_slices[f's_{loc}']
        lbc_ds[f'{name}_{loc}'][t,:,:,:] = fld_les[lbc_slice]

    # Save 3D buffer field.
    fld_les[kstart_buffer:, n_pad:-n_pad, n_pad:-n_pad].tofile(f'{output_dir}/{name}_buffer.{time:07d}')


def parse_momentum(
    lbc_ds,
    name_suffix,
    t,
    time,
    u_in,
    v_in,
    w_in,
    z_in,
    z,
    zh,
    dz,
    dzi,
    rho,
    rhoh,
    ip_u,
    ip_v,
    ip_s,
    lbc_slices,
    sigma_n,
    domain,
    kstart_buffer,
    kstarth_buffer,
    output_dir,
    float_type):
    """
    Parse all momentum fields for a single time step..
    Creates both the initial field (t=0 only) and lateral boundary conditions.

    Steps:
    1. Blend w to zero to surface over a certain (500 m) depth.
    2. Correct horizontal divergence of u and v to match subsidence in LES to ERA5.
    3. Calculate new vertical velocity w to ensure that the fields are divergence free.
    4. Check resulting divergence.

    Parameters
    ----------
    lbc_ds : xarray.Dataset
        Dataset containing lateral boundary condition (LBC) fields.
    name_suffix : str
        Suffix to append to the output variable name.
    t : int
        Timestep index.
    time : int
        Time in seconds since start of experiment.
    u_in : np.ndarray, shape (3,)
        u-field from host model.
    v_in : np.ndarray, shape (3,)
        v-field from host model.
    w_in : np.ndarray, shape (3,)
        w-field from host model.
    z_in : np.ndarray, shape (3,)
        Model level heights host model.
    z : np.ndarray, shape (1,)
        Full level heights LES.
    zh : np.ndarray, shape (1,)
        Half level heights LES.
    dz : np.ndarray, shape (1,)
        Full level grid spacing LES.
    dzi : np.ndarray, shape (1,)
        Inverse of full level grid spacing LES.
    rho : np.ndarray, shape (1,)
        Full level base state density.
    rhoh : np.ndarray, shape (1,)
        Half level base state density.
    ip_u : `Rect_to_curv_interpolation_factors` instance.
        Interpolation factors at u location.
    ip_v : `Rect_to_curv_interpolation_factors` instance.
        Interpolation factors at v location.
    ip_s : `Rect_to_curv_interpolation_factors` instance.
        Interpolation factors at scalar location.
    lbc_slices : dict
        Dictionary with Numpy slices for each LBC.
    sigma_n : int
        Width Gaussian filter kernel in LES grid points.
    domain : Domain instance
        Domain information.
    kstart_buffer : int
        Start index (full levels) of the buffer in the vertical direction.
    kstarth_buffer : int
        Start index (half levels) of the buffer in the vertical direction.
    output_dir : str
        Output directory.
    float_type : np.float32 or np.float64
        Floating point precision.

    Returns
    -------
    None
    """
    logger.debug(f'Processing momentum at t={time}.')

    # Short-cuts
    n_pad = domain.n_pad

    # Keep creation of 3D field here, for parallel/async exectution..
    u = np.empty((z.size,  domain.proj_pad.jtot, domain.proj_pad.itot), dtype=float_type)
    v = np.empty((z.size,  domain.proj_pad.jtot, domain.proj_pad.itot), dtype=float_type)
    w = np.empty((zh.size, domain.proj_pad.jtot, domain.proj_pad.itot), dtype=float_type)

    # Lazily load data in case xarray data is provided.
    if isinstance(u_in, xr.DataArray):
        u_in = u_in.values
    if isinstance(v_in, xr.DataArray):
        v_in = v_in.values
    if isinstance(w_in, xr.DataArray):
        w_in = w_in.values

    # Tri-linear interpolation from host to LES grid.
    interp_rect_to_curv_kernel(
        u,
        u_in,
        ip_u.il,
        ip_u.jl,
        ip_u.fx,
        ip_u.fy,
        z,
        z_in,
        float_type)

    interp_rect_to_curv_kernel(
        v,
        v_in,
        ip_v.il,
        ip_v.jl,
        ip_v.fx,
        ip_v.fy,
        z,
        z_in,
        float_type)

    interp_rect_to_curv_kernel(
        w,
        w_in,
        ip_s.il,
        ip_s.jl,
        ip_s.fx,
        ip_s.fy,
        zh,
        z_in,
        float_type)

    # Apply Gaussian filter.
    if sigma_n > 0:
        gaussian_filter_wrapper(u, sigma_n)
        gaussian_filter_wrapper(v, sigma_n)
        gaussian_filter_wrapper(w, sigma_n)

    # Host model vertical velocity `w_in` sometimes has strange profiles near surface.
    # Blend linearly to zero. This also insures that w at the surface is 0.0 m/s.
    blend_w_to_zero_at_sfc(w, zh, zmax=500)

    # DEBUG..
    if t == 0:
        print('DEBUG: saving non-fixed velocity fields...')
        u.tofile('u0.0000000')
        v.tofile('v0.0000000')
        w.tofile('w0.0000000')

    # Correct horizontal divergence of u and v.
    proj = domain.proj_pad
    correct_div_uv(
        u,
        v,
        w,
        rho,
        rhoh,
        dzi,
        proj.x,
        proj.y,
        proj.xsize,
        proj.ysize,
        domain.n_pad)

    # NOTE: correcting the horizontal divergence only ensures that the _mean_ vertical velocity
    # is correct. We still need to calculate a new vertical velocity to ensure that the wind
    # fields are divergence free at a grid point level.
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

    # Save 3D field without ghost cells in binary format as initial/restart file.
    if t == 0:
        save_3d_field(u[:  ,:,:], 'u', name_suffix, n_pad, output_dir)
        save_3d_field(v[:  ,:,:], 'v', name_suffix, n_pad, output_dir)
        save_3d_field(w[:-1,:,:], 'w', name_suffix, n_pad, output_dir)

    # Save top boundary condition vertical velocity.
    n = domain.n_pad
    time = int(lbc_ds['time'][t])
    w[-1, n:-n, n:-n].tofile(f'{output_dir}/w_top.{time:07d}')

    # Save lateral boundaries.
    for loc in ('west', 'east', 'south', 'north'):
        lbc_slice = lbc_slices[f'u_{loc}']
        lbc_ds[f'u_{loc}'][t,:,:,:] = u[lbc_slice]

        lbc_slice = lbc_slices[f'v_{loc}']
        lbc_ds[f'v_{loc}'][t,:,:,:] = v[lbc_slice]

        lbc_slice = lbc_slices[f'w_{loc}']
        lbc_ds[f'w_{loc}'][t,:,:,:] = w[lbc_slice]

    # Save 3D buffer field.
    u[kstart_buffer:,    n_pad:-n_pad, n_pad:-n_pad].tofile(f'{output_dir}/u_buffer.{time:07d}')
    v[kstart_buffer:,    n_pad:-n_pad, n_pad:-n_pad].tofile(f'{output_dir}/v_buffer.{time:07d}')
    w[kstarth_buffer:-1, n_pad:-n_pad, n_pad:-n_pad].tofile(f'{output_dir}/w_buffer.{time:07d}')


def parse_pressure(
        p_in,
        z_in,
        zsize,
        ip_s,
        domain,
        time,
        output_dir,
        float_type):
    """
    Interpolate 3D pressure field from host model to top-of-domain (TOD) in LES.

    Parameters:
    ----------
    p_in : np.ndarray, shape (3,)
        Pressure field.
    z_in : np.ndarray, shape (3,)
        Model level heights.
    zsize : float
        Domain height LES.
    ip_s : `Rect_to_curv_interpolation_factors` instance
        Interpolation factors at scalar location.
    domain : Domain instance
        Domain information.
    time : int
        Time in seconds since start of experiment.
    output_dir : str
        Output directory files.
    float_type : np.float32 or np.float64
        Floating point precision.

    returns:
    -------
    None
    """
    logger.debug(f'Processing TOD pressure at t={time}.')

    p_les = np.empty((domain.proj_pad.jtot, domain.proj_pad.itot), dtype=float_type)

    # Lazily load data in case xarray data is provided.
    if isinstance(p_in, xr.DataArray):
        p_in = p_in.values

    interp_rect_to_curv_kernel(
        p_les,
        p_in,
        ip_s.il,
        ip_s.jl,
        ip_s.fx,
        ip_s.fy,
        zsize,
        z_in,
        float_type)

    # Save pressure at top of domain without ghost cells.
    n = domain.n_pad
    p_les[n:-n, n:-n].tofile(f'{output_dir}/phydro_tod.{time:07d}')


def create_era5_input(*args, **kwargs):
    """
    Wrapper to not break old(er) code.
    """
    logger.warning('Function `create_era5_input()` has been renamed to `create_input_from_regular_latlon()`.')
    logger.warning('You can safely rename the function to its new name.')
    create_input_from_regular_latlon(*args, **kwargs)


def create_input_from_regular_latlon(
        fields_in,
        lon_in,
        lat_in,
        z_in,
        p_in,
        time_in,
        z,
        zsize,
        zstart_buffer,
        rho,
        rhoh,
        domain,
        sigma_h,
        perturb_size=0,
        perturb_amplitude={},
        perturb_max_height=0,
        clip_at_zero=(),
        name_suffix='',
        output_dir='.',
        ntasks=8,
        float_type=np.float64):
    """
    Generate all required MicroHH input from large-scale models
    using a regular lat-lon grid (i.e. a grid where lat/lon are 1D arrays).

    Fields that are created, some optional:
    1. Initial fields.
    2. Lateral boundary conditions.
    3. Upper boundary conditions (w).
    4. Sponge layer.
    5. Top-of-domain hydrostatic pressure.

    Parameters
    ----------
    fields_in : dict
        Dictionary containing 4D fields from host model.
    lon_in : np.ndarray, shape (1,)
        Input longitude coordinates in degrees.
    lat_in : np.ndarray, shape (1,)
        Input latitude coordinates in degrees.
    z_in : np.ndarray, shape (4,)
        Input model level heights in meters.
    p_in : np.ndarray, shape (4,)
        Input pressure levels in Pa.
    time_in : np.ndarray, shape (1,)
        Input time coordinates in seconds.
    z : np.ndarray, shape (1,)
        LES full level heights in meters.
    zsize : float
        Vertical domain size in meters.
    zstart_buffer : float
        Vertical start height buffer in meters.
    rho : np.ndarray, shape (1,)
        Base state density profile.
    rhoh : np.ndarray, shape (1,)
        Base state density at half levels.
    domain : Domain instance
        Domain / projection information.
    sigma_h : float
        Width of Gaussian smoothing filter kernel (total size is +/- 3 sigma)
    perturb_size : int, optional
        Perturb 3D fields in blocks of certain size. Default is 0.
    perturb_amplitude : dict, optional
        Dictionary with perturbation amplitudes for each field. Default is {}.
    perturb_max_height : float, optional
        Maximum height for perturbations in meters. Default is 0.
    clip_at_zero : tuple, optional
        Tuple of field names to clip at >= 0. Default is ().
    name_suffix : str, optional
        Suffix to append to output variable names. Default is ''.
    output_dir : str, optional
        Output directory path. Default is '.'.
    ntasks : int, optional
        Number of parallel tasks. Default is 8.
    float_type : np.float32 or np.float64, optional
        Floating point precision. Default is np.float64.

    Returns
    -------
    None
    """
    logger.info(f'Creating MicroHH input in {output_dir}.')

    # Short-cuts.
    proj_pad = domain.proj_pad
    time_in = time_in.astype(np.int32)

    # Setup vertical grid. Definition has to perfectly match MicroHH's vertical grid to get divergence free fields.
    gd = calc_vertical_grid_2nd(z, zsize, remove_ghost=True, float_type=float_type)

    # Setup horizontal interpolations (indexes and factors).
    ip_u, ip_v, ip_s = setup_interpolations(lon_in, lat_in, proj_pad, float_type=float_type)

    # Setup spatial filtering.
    sigma_n = int(np.ceil(sigma_h / proj_pad.dx))
    if sigma_n > 0:
        logger.info(f'Using Gaussian filter with sigma = {sigma_n} grid cells')

    # Setup 3D buffer output.
    kstart_buffer  = int(np.where(gd['z']  >= zstart_buffer)[0][0])
    kstarth_buffer = int(np.where(gd['zh'] >= zstart_buffer)[0][0])

    # Setup lateral boundary fields.
    lbc_ds = create_lbc_ds(
        list(fields_in.keys()),
        time_in,
        domain.x,
        domain.y,
        gd['z'],
        domain.xh,
        domain.yh,
        gd['zh'][:-1],
        domain.n_ghost,
        domain.n_sponge,
        dtype=float_type)

    # Numpy slices of lateral boundary conditions.
    lbc_slices = setup_lbc_slices(domain.n_ghost, domain.n_sponge)

    # Keep track of fields/LBCs that have been parsed.
    fields = []


    """
    Interpolate 3D pressure to domain top LES.
    """
    if p_in is not None:
        args = []
        for t in range(time_in.size):
            args.append(
                (p_in[t,:,:,:],
                 z_in[t,:,:,:],
                 gd['zsize'],
                 ip_s,
                 domain,
                 time_in[t],
                 output_dir,
                 float_type)
            )

        def parse_pressure_wrapper(args):
            return parse_pressure(*args)

        tick = datetime.now()

        with ThreadPoolExecutor(max_workers=ntasks) as executor:
            results = list(executor.map(parse_pressure_wrapper, args))

        tock = datetime.now()
        logger.info(f'Created TOD pressure input in {tock - tick}.')


    """
    Parse scalars.
    This creates the initial fields for t=0 and lateral boundary conditions for all times.
    """
    # Run in parallel with ThreadPoolExecutor for ~10x speed-up.
    args = []
    for name, fld_in in fields_in.items():
        if name not in ('u', 'v', 'w'):
            fields.append(name)

            for t in range(time_in.size):
                args.append((
                    lbc_ds,
                    name,
                    name_suffix,
                    t,
                    time_in[t],
                    fld_in[t,:,:,:],
                    z_in[t,:,:,:],
                    gd['z'],
                    ip_s,
                    lbc_slices,
                    sigma_n,
                    perturb_size,
                    perturb_amplitude,
                    perturb_max_height,
                    clip_at_zero,
                    domain,
                    kstart_buffer,
                    output_dir,
                    float_type))

    def parse_scalar_wrapper(args):
        return parse_scalar(*args)

    tick = datetime.now()

    with ThreadPoolExecutor(max_workers=ntasks) as executor:
        results = list(executor.map(parse_scalar_wrapper, args))

    tock = datetime.now()
    logger.info(f'Created scalar input in {tock - tick}.')


    """
    Parse momentum fields.
    This is treated separately, because it requires some corrections to ensure that the fields are divergence free.
    """
    if any(fld not in fields_in for fld in ('u', 'v', 'w')):
        logger.debug('One or more momentum fields missing! Skipping momentum...')
    else:
        fields.extend(['u', 'v', 'w'])

        # Run in parallel with ThreadPoolExecutor for ~10x speed-up.
        args = []
        for t in range(time_in.size):
            args.append((
                lbc_ds,
                name_suffix,
                t,
                time_in[t],
                fields_in['u'][t,:,:,:],
                fields_in['v'][t,:,:,:],
                fields_in['w'][t,:,:,:],
                z_in[t,:,:,:],
                gd['z'],
                gd['zh'],
                gd['dz'],
                gd['dzi'],
                rho,
                rhoh,
                ip_u,
                ip_v,
                ip_s,
                lbc_slices,
                sigma_n,
                domain,
                kstart_buffer,
                kstarth_buffer,
                output_dir,
                float_type))

        def parse_momentum_wrapper(args):
            return parse_momentum(*args)

        tick = datetime.now()

        with ThreadPoolExecutor(max_workers=ntasks) as executor:
            results = list(executor.map(parse_momentum_wrapper, args))

        tock = datetime.now()
        logger.info(f'Created momentum input in {tock - tick}.')


    """
    Write lateral boundary conditions to file.
    """
    lbc_ds_to_binary(lbc_ds, output_dir, float_type)
