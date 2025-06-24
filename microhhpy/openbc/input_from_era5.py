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
from datetime import datetime

from concurrent.futures import ThreadPoolExecutor
import multiprocessing

# Local library
from microhhpy.logger import logger
from microhhpy.interpolate.interpolate_kernels import Rect_to_curv_interpolation_factors
from microhhpy.interpolate.interpolate_kernels import interpolate_rect_to_curv
from microhhpy.spatial import calc_vertical_grid_2nd

from .global_help_functions import gaussian_filter_wrapper
from .global_help_functions import correct_div_uv
from .numba_kernels import block_perturb_field, blend_w_to_zero_at_sfc, calc_w_from_uv, check_divergence
from .lbc_help_functions import create_lbc_ds, setup_lbc_slices


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
        Spatial projection.
    dtype : numpy float type, optional
        Data type output arrays.

    Returns:
    -------
    tuple of `Rect_to_curv_interpolation_factors` instances at (u, v, s) locations.
    """

    if_u = Rect_to_curv_interpolation_factors(
        lon_era, lat_era, proj_pad.lon_u, proj_pad.lat_u, dtype)

    if_v = Rect_to_curv_interpolation_factors(
        lon_era, lat_era, proj_pad.lon_v, proj_pad.lat_v, dtype)

    if_s = Rect_to_curv_interpolation_factors(
        lon_era, lat_era, proj_pad.lon, proj_pad.lat, dtype)

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
    fld_era,
    z_era,
    z_les,
    ip_fac,
    lbc_slices,
    sigma_n,
    perturb_size,
    perturb_amplitude,
    domain,
    output_dir,
    dtype):
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
    fld_era : np.ndarray, shape (3,)
        Scalar field from ERA5.
    z_era : np.ndarray, shape (3,)
        Model level heights ERA5.
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
    domain : Domain instance
        Domain information.
    output_dir : str
        Output directory.
    dtype : np.float32 or np.float64
        Floating point precision.

    Returns
    -------
    None
    """
    logger.debug(f'Processing field {name} at t={time}.')

    # Keep creation of 3D field here, for parallel/async exectution..
    fld_les = np.empty((z_les.size, domain.proj_pad.jtot, domain.proj_pad.itot), dtype=dtype)

    # Tri-linear interpolation from ERA5 to LES grid.
    interpolate_rect_to_curv(
        fld_les,
        fld_era,
        ip_fac.il,
        ip_fac.jl,
        ip_fac.fx,
        ip_fac.fy,
        z_les,
        z_era,
        dtype)
    
    # Apply Gaussian filter.
    if sigma_n > 0:
        gaussian_filter_wrapper(fld_les, sigma_n)

    # Apply perturbation to the field.
    if name in perturb_amplitude.keys() and perturb_size > 0:
        block_perturb_field(fld_les, perturb_size, perturb_amplitude[name])
    
    # Save 3D field without ghost cells in binary format as initial/restart file.
    if t == 0:
        save_3d_field(fld_les, name, name_suffix, domain.n_pad, output_dir)

    # Save lateral boundaries.
    for loc in ('west', 'east', 'south', 'north'):
        lbc_slice = lbc_slices[f's_{loc}']
        lbc_ds[f'{name}_{loc}'][t,:,:,:] = fld_les[lbc_slice]


def parse_momentum(
    lbc_ds,
    name_suffix,
    t,
    time,
    u_era,
    v_era,
    w_era,
    z_era,
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
    output_dir,
    dtype):
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
    u_era : np.ndarray, shape (3,)
        u-field from ERA5.
    v_era : np.ndarray, shape (3,)
        v-field from ERA5.
    w_era : np.ndarray, shape (3,)
        w-field from ERA5.
    z_era : np.ndarray, shape (3,)
        Model level heights ERA5.
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
    output_dir : str
        Output directory.
    dtype : np.float32 or np.float64
        Floating point precision.

    Returns
    -------
    None
    """
    logger.debug(f'Processing momentum at t={time}.')

    # Keep creation of 3D field here, for parallel/async exectution..
    u = np.empty((z.size,  domain.proj_pad.jtot, domain.proj_pad.itot), dtype=dtype)
    v = np.empty((z.size,  domain.proj_pad.jtot, domain.proj_pad.itot), dtype=dtype)
    w = np.empty((zh.size, domain.proj_pad.jtot, domain.proj_pad.itot), dtype=dtype)

    # Tri-linear interpolation from ERA5 to LES grid.
    interpolate_rect_to_curv(
        u,
        u_era,
        ip_u.il,
        ip_u.jl,
        ip_u.fx,
        ip_u.fy,
        z,
        z_era,
        dtype)

    interpolate_rect_to_curv(
        v,
        v_era,
        ip_v.il,
        ip_v.jl,
        ip_v.fx,
        ip_v.fy,
        z,
        z_era,
        dtype)

    interpolate_rect_to_curv(
        w,
        w_era,
        ip_s.il,
        ip_s.jl,
        ip_s.fx,
        ip_s.fy,
        zh,
        z_era,
        dtype)
    
    # Apply Gaussian filter.
    if sigma_n > 0:
        gaussian_filter_wrapper(u, sigma_n)
        gaussian_filter_wrapper(v, sigma_n)
        gaussian_filter_wrapper(w, sigma_n)

    # ERA5 vertical velocity `w_era` sometimes has strange profiles near surface.
    # Blend linearly to zero. This also insures that w at the surface is 0.0 m/s.
    blend_w_to_zero_at_sfc(w, zh, zmax=500)

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
        save_3d_field(u[:  ,:,:], 'u', name_suffix, domain.n_pad, output_dir)
        save_3d_field(v[:  ,:,:], 'v', name_suffix, domain.n_pad, output_dir)
        save_3d_field(w[:-1,:,:], 'w', name_suffix, domain.n_pad, output_dir)

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


def parse_pressure(
        p_era,
        z_era,
        zsize,
        ip_s,
        domain,
        time,
        output_dir,
        dtype):
    """
    Interpolate 3D pressure field from ERA5 to top-of-domain (TOD) in LES.

    Arguments:
    ---------
    p_era : np.ndarray, shape (3,)
        Pressure field from ERA5.
    z_era : np.ndarray, shape (3,)
        Model level heights ERA5.
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
    dtype : np.float32 or np.float64
        Floating point precision.

    returns:
    -------
    None
    """
    logger.debug(f'Processing TOD pressure at t={time}.')

    p_les = np.empty((domain.proj_pad.jtot, domain.proj_pad.itot), dtype=dtype)

    interpolate_rect_to_curv(
        p_les,
        p_era,
        ip_s.il,
        ip_s.jl,
        ip_s.fx,
        ip_s.fy,
        zsize,
        z_era,
        dtype)

    # Save pressure at top of domain without ghost cells.
    n = domain.n_pad
    p_les[n:-n, n:-n].tofile(f'{output_dir}/phydro_tod.{time:07d}')


def create_era5_input(
        fields_era,
        lon_era,
        lat_era,
        z_era,
        p_era,
        time_era,
        z,
        zsize,
        rho,
        rhoh,
        domain,
        sigma_h,
        perturb_size=0,
        perturb_amplitude={},
        name_suffix='',
        output_dir='.',
        ntasks=8,
        dtype=np.float64):
    """
    Generate all required MicroHH input from ERA5.
    
    1. Initial fields.
    2. Lateral boundary conditions.
    3. ...
    """
    logger.info(f'Creating MicroHH input from ERA5 data in {output_dir}.')

    # Short-cuts.
    proj_pad = domain.proj_pad
    time_era = time_era.astype(np.int32)

    # Setup vertical grid. Definition has to perfectly match MicroHH's vertical grid to get divergence free fields.
    gd = calc_vertical_grid_2nd(z, zsize, remove_ghost=True, dtype=dtype)

    # Setup horizontal interpolations (indexes and factors).
    ip_u, ip_v, ip_s = setup_interpolations(lon_era, lat_era, proj_pad, dtype=dtype)

    # Setup spatial filtering.
    sigma_n = int(np.ceil(sigma_h / proj_pad.dx))
    if sigma_n > 0:
        logger.info(f'Using Gaussian filter with sigma = {sigma_n} grid cells')

    # Setup lateral boundary fields.
    lbc_ds = create_lbc_ds(
        list(fields_era.keys()),
        time_era,
        domain.x,
        domain.y,
        gd['z'],
        domain.xh,
        domain.yh,
        gd['zh'][:-1],
        domain.n_ghost,
        domain.n_sponge,
        dtype=dtype)

    # Numpy slices of lateral boundary conditions.
    lbc_slices = setup_lbc_slices(domain.n_ghost, domain.n_sponge)

    # Keep track of fields/LBCs that have been parsed.
    fields = []


    """
    Interpolate 3D pressure to domain top LES.
    """
    args = []
    for t in range(time_era.size):
        args.append(
            (p_era[t,:,:,:],
             z_era[t,:,:,:],
             gd['zsize'],
             ip_s,
             domain,
             time_era[t],
             output_dir,
             dtype)
        )

    def parse_pressure_wrapper(args):
        return parse_pressure(*args)

    tick = datetime.now()

    with ThreadPoolExecutor(max_workers=ntasks) as executor:
        results = list(executor.map(parse_pressure_wrapper, args))

    tock = datetime.now()
    logger.info(f'Created TOD pressure input from ERA5 in {tock - tick}.')


    """
    Parse scalars.
    This creates the initial fields for t=0 and lateral boundary conditions for all times.
    """
    # Run in parallel with ThreadPoolExecutor for ~10x speed-up.
    args = []
    for name, fld_era in fields_era.items():
        if name not in ('u', 'v', 'w'):
            fields.append(name)

            for t in range(time_era.size):
                args.append((
                    lbc_ds,
                    name,
                    name_suffix,
                    t,
                    time_era[t],
                    fld_era[t,:,:,:],
                    z_era[t,:,:,:],
                    gd['z'],
                    ip_s,
                    lbc_slices,
                    sigma_n,
                    perturb_size,
                    perturb_amplitude,
                    domain,
                    output_dir,
                    dtype))

    def parse_scalar_wrapper(args):
        return parse_scalar(*args)

    tick = datetime.now()

    with ThreadPoolExecutor(max_workers=ntasks) as executor:
        results = list(executor.map(parse_scalar_wrapper, args))

    tock = datetime.now()
    logger.info(f'Created scalar input from ERA5 in {tock - tick}.')


    """
    Parse momentum fields.
    This is treated separately, because it requires some corrections to ensure that the fields are divergence free.
    """
    if any(fld not in fields_era for fld in ('u', 'v', 'w')):
        logger.warning('One or more momentum fields missing! Skipping momentum...')
    else:
        fields.extend(['u', 'v', 'w'])

        # Run in parallel with ThreadPoolExecutor for ~10x speed-up.
        args = []
        for t in range(time_era.size):
            args.append((
                lbc_ds,
                name_suffix,
                t,
                time_era[t],
                fields_era['u'][t,:,:,:],
                fields_era['v'][t,:,:,:],
                fields_era['w'][t,:,:,:],
                z_era[t,:,:,:],
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
                output_dir,
                dtype))

        def parse_momentum_wrapper(args):
            return parse_momentum(*args)

        tick = datetime.now()

        with ThreadPoolExecutor(max_workers=ntasks) as executor:
            results = list(executor.map(parse_momentum_wrapper, args))

        tock = datetime.now()
        logger.info(f'Created momentum input from ERA5 in {tock - tick}.')


        """
        Write lateral boundary conditions to file.
        """
        for fld in fields:
            for loc in ['west', 'east', 'north', 'south']:
                lbc_in = lbc_ds[f'{fld}_{loc}'].values.astype(dtype)
                lbc_in.tofile(f'{output_dir}/lbc_{fld}_{loc}.0000000')