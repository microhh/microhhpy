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
import glob
import os

# Third-party.
import rioxarray as rxr
import xarray as xr

# Local library


def read_hihydrosoil_subtop(geotiff_path, lon0, lon1, lat0, lat1):
    """
    Read HiHydroSoil v2.0 GeoTIFF files from:
    - https://www.futurewater.eu/projects/hihydrosoil/
    - https://www.dropbox.com/sh/iaj2ika2t1pr7lr/AACdn_pqsijYXbPHNSbAcCKba?dl=0

    NOTE: only the sub/top-soil files are currently supported!
    Top soil = 0 - 30 cm
    Sub soil = 30 - 200 cm

    Arguments:
    ----------
    geotiff_path : str
        Path to top/sub-soil GeoTIFF files.
    lon0 : float
        Bounding box longitude west
    lon1 : float
        Bounding box longitude east
    lat0 : float
        Bounding box latitude south
    lat1 : float
        Bounding box latitude north

    Returns:
    --------
    ds : xarray.Dataset
        HiHydroSoil parameters in Xarray Dataset form.
    """
    scale_fac = 0.0001

    tiff_files = glob.glob(f'{geotiff_path}/*.tif')

    variables = {}
    for tiff_file in tiff_files:
        filename = os.path.basename(tiff_file)

        if 'TOPSOIL' in filename:
            var_name = filename.replace('_TOPSOIL.tif', '')
            soil_layer = 'TOPSOIL'
        elif 'SUBSOIL' in filename:
            var_name = filename.replace('_SUBSOIL.tif', '')
            soil_layer = 'SUBSOIL'
        else:
            continue
        
        if var_name not in variables:
            variables[var_name] = {}

        ds = rxr.open_rasterio(tiff_file)
        ds = ds.reindex(y=ds.y[::-1])
        dss = ds.sel(
            x=slice(lon0-0.5, lon1+0.5),
            y=slice(lat0-0.5, lat1+0.5))
        dss = dss.sel(band=1)
        dss = dss.where(dss >= 0) * scale_fac

        variables[var_name][soil_layer] = dss

    data_vars = {}
    for var_name, soil_data in variables.items():

        layers = []
        layer_names = []
        for layer_name in ['TOPSOIL', 'SUBSOIL']:
            if layer_name in soil_data:
                layers.append(soil_data[layer_name])
                layer_names.append(layer_name)

        if layers:
            var_da = xr.concat(layers, dim='soil_layer')
            var_da = var_da.assign_coords(soil_layer=layer_names)
            data_vars[var_name] = var_da

    ds = xr.Dataset(data_vars)

    ds = ds.rename({
        'WCpF2_M_250m': 'theta_fc',
        'WCpF3_M_250m': 'theta_wp',
        'WCres_M_250m': 'theta_res',
        'WCsat_M_250m': 'theta_sat',
        'ALFA_M_250m': 'vg_a',
        'N_M_250m': 'vg_n',
        'Ksat_M_250m': 'ksat'
    })

    return ds