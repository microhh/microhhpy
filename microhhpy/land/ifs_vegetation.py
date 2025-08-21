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
from matplotlib.colors import ListedColormap
from numba import njit
import xarray as xr
import numpy as np

# Local library

def get_ifs_vegetation_lut():
    _raw_data = np.array([
        # vt    z0m      z0h        rsmin cveg  gD         a_r     b_r     L_us  L_st  f_rs   rs  lai, alb
        # 0     1        2          3     4     5          6       7       8     9     10     11  12   13
        [ 0,    0.250,   0.25e-2,   100,  0.90, 0.00/100., 5.558,  2.614,  10.0, 10.0, 0.05,  1,  3 ,  0.16],    # 0  1  Crops, mixed farming | *1
        [ 0,    0.200,   0.2e-2,    100,  0.85, 0.00/100., 10.739, 2.608,  10.0, 10.0, 0.05,  1,  2 ,  0.21],    # 1  2  Short grass | *1
        [ 1,    2.000,   2.0,       250,  0.90, 0.03/100., 6.706,  2.175,  40.0, 15.0, 0.03,  2,  5 ,  0.12],    # 2  3  Evergreen needleleaf | *1
        [ 1,    2.000,   2.0,       250,  0.90, 0.03/100., 7.066,  1.953,  40.0, 15.0, 0.03,  2,  5 ,  0.16],    # 3  4  Deciduous needleleaf | *1
        [ 1,    2.000,   2.0,       175,  0.90, 0.03/100., 5.990,  1.955,  40.0, 15.0, 0.03,  2,  5 ,  0.16],    # 4  5  Deciduous broadleaf | *1
        [ 1,    2.000,   2.0,       240,  0.99, 0.03/100., 7.344,  1.303,  40.0, 15.0, 0.035, 2,  6 ,  0.17],    # 5  6  Evergreen broadleaf | *1
        [ 0,    0.470,   0.47e-2,   100,  0.70, 0.00/100., 8.235,  1.627,  10.0, 15.0, 0.05,  1,  2 ,  0.16],    # 6  7  Tall grass | *1
        [ -1,   0.013,   0.013e-2,  250,  0,    0.00/100., 4.372,  0.978,  15.0, 15.0, 0.00,  1,  0.5, 0.36],    # 7  8  Desert | *1
        [ 0,    0.034,   0.034e-2,  80,   0.50, 0.00/100., 8.992,  8.992,  10.0, 10.0, 0.05,  1,  1 ,  0.17],    # 8  9  Tundra | *1
        [ 0,    0.500,   0.5e-2,    180,  0.90, 0.00/100., 5.558,  2.614,  10.0, 10.0, 0.05,  1,  3 ,  0.175],   # 9  10  Irrigated crops | *3
        [ 0,    0.170,   0.17e-2,   150,  0.10, 0.00/100., 4.372,  0.978,  10.0, 10.0, 0.05,  1,  0.5, 0.265],   # 10 11 Semidesert | *4
        [ -1,   1.3e-10, 1.3e-2,    -1,   -1,   -1,        -1,     -1,     58.0, 58.0, 0.00,  0,  -1,  0.62],    # 11 12 Ice caps and glaciers | *1
        [ 0,    0.830,   0.83e-2,   240,  0.60, 0.00/100., 7.344,  1.303,  10.0, 10.0, 0.05,  1,  0.6, 0.14],    # 12 13 Bogs and marshes | *2
        [ -1,   -1   ,   -1,        -1,   -1,   -1,        -1,     -1,     -1,   -1,   0.00,  0,  -1,  0.08],    # 13 14 Inland water | *2
        [ -1,   -1   ,   -1,        -1,   -1,   -1,        -1,     -1,     -1,   -1,   0.00,  0,  -1,  0.08],    # 14 15 Ocean | *2
        [ 0,    0.100,   0.1e-2,    225,  0.50, 0.00/100., 6.326,  1.567,  10.0, 10.0, 0.05,  1,  3,   0.20],    # 15 16 Evergreen shrubs | *1
        [ 0,    0.250,   0.25e-2,   225,  0.50, 0.00/100., 6.326,  1.567,  10.0, 10.0, 0.05,  1,  1.5, 0.21],    # 16 17 Deciduous shrubs | *1
        [ 1,    2.000,   2.0,       250,  0.90, 0.03/100., 4.453,  1.631,  40.0, 15.0, 0.03,  2,  5,   0.15],    # 17 18 Mixed forest- Wood | *1
        [ 1,    1.100,   1.1,       175,  0.90, 0.03/100., 4.453,  1.631,  40.0, 15.0, 0.03,  2,  2.5, 0.15],    # 18 19 Interrupted forest | *1
        [ 0,    -1,      -1,        150,  0.60, 0.00/100., -1,     -1,     -1,   -1,   0.00,  0,  4,   0.14],    # 19 20 Water-land mixtures | *5
        [ 0,    1.0,     1.0,       100,  0.50, 0.00/100., 10.739, 2.608,  30.0, 30.0, 0.00,  1,  2,   0.18]])   # 20 21 "Urban" --> not part of IFS | *2

    # *1: Coakley, J. A. "Reflectance and albedo, surface." Encyclopedia of atmospheric sciences 12 (2003).
    # *2: Stull, BLM.
    # *3: Most common in NL = wheat, followed by corn and potatoes.
    # *4: Mean of desert and tundra
    # *5: Mean of water and 0.2

    _names = [
        'crops_mixed_farming',
        'short_grass',
        'evergreen_needleleaf',
        'deciduous_needleleaf',
        'deciduous_broadleaf',
        'evergreen_broadleaf',
        'tall_grass',
        'desert',
        'tundra',
        'irrigated_crops',
        'semidesert',
        'ice_caps_glaciers',
        'bogs_marshes',
        'inland_water',
        'ocean',
        'evergreen_shrubs',
        'deciduous_shrubs',
        'mixed_forest_wood',
        'interrupted_forest',
        'water_land_mixtures',
        'urban']

    ifs_vegetation = dict(
        name      = _names,
        veg_type  = _raw_data[:,0].astype(int),
        z0m       = _raw_data[:,1],
        z0h       = _raw_data[:,2],
        rs_min    = _raw_data[:,3],
        c_veg     = _raw_data[:,4],
        gD        = _raw_data[:,5],
        a_r       = _raw_data[:,6],
        b_r       = _raw_data[:,7],
        lambda_us = _raw_data[:,8],
        lambda_s  = _raw_data[:,9],
        f_rs      = _raw_data[:,10],
        rs        = _raw_data[:,11].astype(int),
        lai       = _raw_data[:,12],
        albedo    = _raw_data[:,13])

    return ifs_vegetation


def get_ifs_vegetation_cmap():
    """
    Create matplotlib colormap for IFS vegetation.
    Generated by claude.ai
    """

    _vegetation_colors = [
        '#FFD700',  # crops_mixed_farming - golden yellow
        '#90EE90',  # short_grass - light green
        '#006400',  # evergreen_needleleaf - dark green
        '#228B22',  # deciduous_needleleaf - forest green
        '#32CD32',  # deciduous_broadleaf - lime green
        '#004225',  # evergreen_broadleaf - very dark green
        '#7CFC00',  # tall_grass - lawn green
        '#F4A460',  # desert - sandy brown
        '#D3D3D3',  # tundra - light gray
        '#ADFF2F',  # irrigated_crops - green yellow
        '#DEB887',  # semidesert - burlywood
        '#F0F8FF',  # ice_caps_glaciers - alice blue
        '#8FBC8F',  # bogs_marshes - dark sea green
        '#4169E1',  # inland_water - royal blue
        '#000080',  # ocean - navy blue
        '#556B2F',  # evergreen_shrubs - dark olive green
        '#9ACD32',  # deciduous_shrubs - yellow green
        '#2E8B57',  # mixed_forest_wood - sea green
        '#66CDAA',  # interrupted_forest - medium aquamarine
        '#87CEEB',  # water_land_mixtures - sky blue
        '#696969'   # urban - dim gray
    ]

    return ListedColormap(_vegetation_colors, name='vegetation')


def get_ifs_soil_lut():
    """
    Read the lookup table with van Genuchten parameters.
    """
    def get_data_file(file_name):
        with resources.as_file(resources.files('microhhpy.data') / file_name) as path:
            return path

    nc_file = get_data_file('van_genuchten_parameters.nc')

    return xr.open_dataset(nc_file)


def calc_root_fraction_1d(a_r, b_r, zh):
    """
    Calculate root fraction using the `a_r` and `b_r` coefficients from IFS.
    """
    root_frac = np.zeros(zh.size-1)

    for k in range(1, zh.size-1):
        root_frac[k] = 0.5 * (np.exp(a_r * zh[k+1]) + \
                              np.exp(b_r * zh[k+1]) - \
                              np.exp(a_r * zh[k  ]) - \
                              np.exp(b_r * zh[k  ]))

    # Make sure the profile sums to 1.
    root_frac[0] = 1-root_frac.sum()

    return root_frac


@njit
def calc_root_fraction_3d(root_frac, a_r, b_r, zh):
    """
    Calculate root fraction using the `a_r` and `b_r` coefficients from IFS for 2D input.
    """
    ktot, jtot, itot = root_frac.shape

    for k in range(1, ktot):
        for j in range(jtot):
            for i in range(itot):
                if a_r[j,i] >= 0 and b_r[j,i] >= 0:
                    root_frac[k,j,i] = 0.5 * (np.exp(a_r[j,i] * zh[k+1]) + \
                                              np.exp(b_r[j,i] * zh[k+1]) - \
                                              np.exp(a_r[j,i] * zh[k  ]) - \
                                              np.exp(b_r[j,i] * zh[k  ]))
                else:
                    root_frac[k,j,i] = 0.

    # Make sure the profile sums to 1.
    for j in range(jtot):
        for i in range(itot):
            root_frac[0,j,i] = 1 - root_frac[1:,j,i].sum()