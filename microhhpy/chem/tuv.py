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
from pathlib import Path
import subprocess
import os

# Third-party.
import pandas as pd
from datetime import datetime, timedelta

# Local library
from microhhpy.logger import logger
from microhhpy.utils import get_data_file


def create_tuv_input(
        input_file,
        output_file,
        name,
        year,
        month,
        day,
        tstart,
        tstop,
        lon,
        lat):
    """
    Create new TUV input file by reading `input_file`,
    updating the date/time/locations,
    and writing it back to a new `output_file`.

    NOTE: only date/time/location can be editted.
          TUV is very picky on input formats (e.g. time has to be 0.000 hours and not 0 or 0.0 or ...).

    Parameters:
    ----------
    TODO

    Returns:
    -------
    TODO
    """

    def fmt(v):
        return f'{v:10.3f}'

    to_replace= {}

    to_replace['inpfil'] = name
    to_replace['outfil'] = name

    to_replace['iyear'] = int(year)
    to_replace['imonth'] = int(month)
    to_replace['iday'] = int(day)

    to_replace['tstart'] = fmt(tstart)
    to_replace['tstop'] = fmt(tstop)
    to_replace['nt'] = int((tstop - tstart)*4+1)

    to_replace['lon'] = fmt(lon)
    to_replace['lat'] = fmt(lat)

    with open(input_file, 'r') as f1:
        with open(output_file, 'w') as f2:
            for l,line in enumerate(f1.readlines()):
                # Only change the header:
                if l > 1 and l < 18:
                    bits = line.split()

                    # Find/replace values.
                    for n in range(3):
                        var = bits[n*3]
                        if var in to_replace.keys():
                            bits[n*3+2] = str(to_replace[var])

                    # Generate back line with correct spacing.
                    line = ''
                    for n in range(3):
                        line += '{0} = {1:>{2}s}   '.format(bits[n*3], bits[n*3+2], 20-len(bits[n*3])-3)
                    line += '\n'

                    f2.write(line)
                else:
                    f2.write(line)


def run_tuv(
        tuv_path,
        name,
        suppress_stdout=True):
    """
    Run TUV code in `tuv_path`, and return to the original working directory.
    """

    cwd = os.getcwd()
    os.chdir(tuv_path)

    if not os.path.exists('tuv'):
        logger.critical('Cannot find tuv executable')

    # Holy fork this is ugly...
    cmd = 'printf \'\n{0}\n\n{0}\n\n\' | ./tuv'.format(name)

    if suppress_stdout:
        subprocess.call(
                cmd, shell=True, executable='/bin/bash', stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    else:
        subprocess.call(cmd, shell=True, executable='/bin/bash')

    os.chdir(cwd)


def read_tuv(
        tuv_file,
        year,
        month,
        day):
    """
    Read photolysis rates from TUV output.
    """

    columns = [
            'time', 'sza', 'jo31d', 'jh2o2', 'jno2',
            'jno3', 'jn2o5', 'jch2or', 'jch2om', 'jch3o2h']

    tuv = pd.read_table(
             tuv_file,
             sep='\s+',
             skiprows=12,
             skipfooter=1,
             engine='python',
             names=columns,
             index_col='time')

    # Create date-time index.
    start_date = datetime(year, month, day)
    dates = [start_date + timedelta(hours=x) for x in tuv.index]
    tuv.index = dates

    return tuv


def calc_tuv_photolysis(
        input_file,
        tuv_path,
        name,
        start_date,
        end_date,
        lon,
        lat,
        suppress_stdout=True):
    """
    Calculate photolysis rates for given time period and location.
        `input_file` is the base input file in `tuv_path/INPUT`.
        `output_file` is `input_file` with date/time/location updated.
        `tuv_path` is path to the `tuv` executable.
        `name` is the experiment name in tuv (has to be SIX characters!).
        `start_date` and `end_date` are `datetime.datetime` instances.
        `lon` and `lat` are the location in degrees.
    """
    logger.info(f'Running TUV for {start_date} to {end_date}')

    path = os.path.dirname(os.path.realpath(__file__))

    # The TUV input is generated from `input_file`, and written to the
    # TUV/INPUTS directory as `name`.
    output_file = f'{tuv_path}/INPUTS/{name}'

    # Empty Pandas dataframe, with correct date index for full period.
    dates = pd.date_range(start_date, end_date, freq='15min')

    columns = [
            'sza', 'jo31d', 'jh2o2', 'jno2',
            'jno3', 'jn2o5', 'jch2or', 'jch2om', 'jch3o2h']

    df = pd.DataFrame(index=dates, columns=columns)

    def is_same_day(d1, d2):
        if d1.year == d2.year and d1.month == d2.month and d1.day == d2.day:
            return True
        return False

    # Loop over days/dates and run TUV!
    date = start_date
    while date < end_date:

        logger.debug(f'Running TUV for {date}')

        tstop = end_date.hour if is_same_day(date, end_date) else 24

        # Create input file for TUV.
        create_tuv_input(
                    input_file,
                    output_file,
                    name,
                    date.year,
                    date.month,
                    date.day,
                    date.hour,
                    tstop,
                    lon,
                    lat)

        # Run TUV. This generates a `name.txt` file with output.
        run_tuv(tuv_path, name, suppress_stdout)

        # The `name.txt` is written one directory up (for convenience? To confuse people? Who knows..)
        txt_file = Path(tuv_path).parent / f'{name}.txt'
        df_sub = read_tuv(txt_file, date.year, date.month, date.day)

        # Copy values to main Dataframe.
        df.loc[df_sub.index] = df_sub

        # Start first day at requested start hour,
        # but following days at 00.00 UTC:
        date += timedelta(hours=24-date.hour)

    return df