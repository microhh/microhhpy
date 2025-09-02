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

from pathlib import Path
try:
    from importlib.resources import files
except ImportError:
    from importlib_resources import files

from microhhpy.logger import logger


def get_data_file(file_name):
    """
    Get path to data file with Python version compatibility.

    Parameters:
    ----------
    file_name : str
        Name of the data file in microhhpy.data package

    Returns:
    -------
    path : pathlib.Path
        Path to the data file
    """
    data_files = files('microhhpy.data')
    data_path = data_files / file_name

    if not data_path.is_file():
        logger.critical(f'Cannot find {file_name} in {data_files}')

    return Path(str(data_path))


def check_domain_decomposition(itot, jtot, ktot, npx, npy):
    """
    Check if domain decomposition is valid.
    """
    err = False
    if itot % npx != 0:
        logger.warning('Invalid decomposition: itot % npx != 0')
        err = True

    if itot % npy != 0:
        logger.warning('Invalid decomposition: itot % npy != 0')
        err = True

    if jtot % npx != 0 and npy > 1:
        logger.warning('Invalid decomposition: jtot % npx != 0')
        err = True

    if jtot % npy != 0:
        logger.warning('Invalid decomposition: jtot % npy != 0')
        err = True

    if ktot % npx != 0:
        logger.warning('Invalid decomposition: ktot % npx != 0')
        err = True

    pts_per_core = int(itot*jtot*ktot/(npx*npy))

    if err:
        logger.critical(f'Invalid grid: itot={itot}, jtot={jtot}, ktot={ktot}, npx={npx}, npy={npy}, #/core={pts_per_core}')
    else:
        logger.info(f'Grid okay: itot={itot}, jtot={jtot}, ktot={ktot}, npx={npx}, npy={npy}, #/core={pts_per_core}')
