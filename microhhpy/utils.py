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

try:
    from importlib.resources import files, as_file
except ImportError:
    from importlib_resources import files, as_file


def get_data_file(file_name):
    """
    Get path to data file with Python version compatibility.
    
    Parameters:
    -----------
    file_name : str
        Name of the data file in microhhpy.data package
        
    Returns:
    --------
    path : pathlib.Path
        Path to the data file
    """
    with as_file(files('microhhpy.data') / file_name) as path:
        return path