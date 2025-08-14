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
import os

# Third-party.

# Local library
from microhhpy.logger import logger


def _int_or_float_or_str(value):
    """
    Helper function: convert a string to int/float/str
    """
    try:
        if value == 'true' or value == 'True':
            return True
        elif value == 'false' or value == 'False':
            return False
        elif value == 'None':
            return None
        elif ('.' in value):
            return float(value)
        else:
            return int(float(value))
    except BaseException:
        return value.rstrip()


def _convert_value(value):
    """
    Helper function: convert namelist value or list
    """
    if ',' in value:
        value = value.split(',')
        return [_int_or_float_or_str(val.strip()) for val in value]
    else:
        return _int_or_float_or_str(value.strip())


def read_ini(namelist_file, ducktype=True):
    """
    Read a MicroHH .ini file into a dictionary-like object.

    Arguments:
    ----------
    namelist_file : str
        Full path to the .ini file.
    ducktype : bool
        If True, convert values to appropriate types (int, float, bool, list).
        If False, keep values as strings. Default is True.

    Returns:
    --------
    ini : nested dictionary
        Dictionary containing the groups and variables from the .ini file.
    """

    ini = {}

    with open(namelist_file) as f:
        for line in f:
            lstrip = line.strip()
            if (len(lstrip) > 0 and lstrip[0] != "#"):
                if lstrip[0] == '[' and lstrip[-1] == ']':
                    curr_group_name = lstrip[1:-1]
                    ini[curr_group_name] = {}
                elif ("=" in line):
                    var_name = lstrip.split('=')[0].strip()
                    value    = lstrip.split('=')[1]

                    if ducktype:
                        value = _convert_value(value)

                    ini[curr_group_name][var_name] = value

    return ini


def save_ini(ini, ini_file, clobber=True):
    """
    Write a nested dictionary back to a MicroHH .ini file.

    Arguments:
    ----------
    ini : nested dictionary
        Dictionary containing the groups and variables to write.
    namelist_file : str
        Full path to the .ini file to write.
    clobber : bool
        If True, allow overwriting an existing file. Default is True.

    Returns:
    --------
    None
    """
    
    if os.path.exists(ini_file) and not clobber:
        logger.critical('ini file \"{}\" already exists and clobber=False'.format(ini_file))

    with open(ini_file, 'w') as f:
        for group, sub_dict in ini.items():
            f.write('[{}]\n'.format(group))
            for variable, value in sub_dict.items():
                if isinstance(value, list):
                    if not isinstance(value[0], str):
                        value = [str(v) for v in value]
                    value = ','.join(value)
                elif isinstance(value, bool):
                    value = 'true' if value else 'false'
                f.write('{}={}\n'.format(variable, value))
            f.write('\n')


def check_ini(ini):
    """
    Check if any of the values in the nested `ini` dict are None.

    Arguments:
    ----------
    ini : nested dictionary
        Dictionary containing the groups and variables.
    """

    contains_none = False

    for group, sub_dict in ini.items():
        for variable, value in sub_dict.items():
            if value is None:
                contains_none = True
                logger.warning(f'ini[{group}][{variable}]=None')

    return contains_none
