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

    Parameters:
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
    -------
    ds : xarray.Dataset
        HiHydroSoil parameters in Xarray Dataset form,
        with separate variables for top and subsoil.
    """
    scale_fac = 0.0001
    tiff_files = glob.glob(f'{geotiff_path}/*.tif')

    variables = {}
    for tiff_file in tiff_files:
        filename = os.path.basename(tiff_file)

        if 'TOPSOIL' in filename:
            var_name = filename.replace('_TOPSOIL.tif', '')
            suffix = 'top'
        elif 'SUBSOIL' in filename:
            var_name = filename.replace('_SUBSOIL.tif', '')
            suffix = 'bot'
        else:
            continue

        ds = rxr.open_rasterio(tiff_file)
        ds = ds.reindex(y=ds.y[::-1])
        dss = ds.sel(
            x=slice(lon0-0.5, lon1+0.5),
            y=slice(lat0-0.5, lat1+0.5)
        )
        dss = dss.sel(band=1)
        dss = dss.where(dss >= 0) * scale_fac

        variables[f"{var_name}_{suffix}"] = dss

    ds = xr.Dataset(variables)

    # Rename to human-friendly names
    rename_map = {
        'WCpF2_M_250m_top': 'theta_fc_top',
        'WCpF2_M_250m_bot': 'theta_fc_bot',
        'WCpF3_M_250m_top': 'theta_wp_top',
        'WCpF3_M_250m_bot': 'theta_wp_bot',
        'WCres_M_250m_top': 'theta_res_top',
        'WCres_M_250m_bot': 'theta_res_bot',
        'WCsat_M_250m_top': 'theta_sat_top',
        'WCsat_M_250m_bot': 'theta_sat_bot',
        'ALFA_M_250m_top': 'vg_a_top',
        'ALFA_M_250m_bot': 'vg_a_bot',
        'N_M_250m_top': 'vg_n_top',
        'N_M_250m_bot': 'vg_n_bot',
        'Ksat_M_250m_top': 'ksat_top',
        'Ksat_M_250m_bot': 'ksat_bot'
    }
    ds = ds.rename({k: v for k, v in rename_map.items() if k in ds})

    return ds