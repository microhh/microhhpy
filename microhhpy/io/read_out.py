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

def read_case_out(out_file):
    """
    Parse `case.out` file from MicroHH.
    Some additional columns are calculated:
    - Cumulative cpu time.
    - Simulated days-per-day (SDPD).

    Return list(dict) of runs from .out file.
    """

    cols = ['iter', 'time', 'cpudt', 'dt', 'cfl' , 'dnum', 'div', 'mom', 'tke']
    
    runs = []
    with open(out_file, 'r') as f:
        for line in f.readlines():
            if 'ITER' in line:
                current_run = {}
                for col in cols:
                    current_run[col] = []
                runs.append(current_run)
            else:
                bits = line.split()
                bits = [float(x) for x in bits]
                for i,col in enumerate(cols):
                    current_run[col].append(bits[i])
    
    # Cast to Numpy arrays.
    for i in range(len(runs)):
        for col in cols:
            runs[i][col] = np.array(runs[i][col])
    
    # Cumulative `cpudt`.
    for i in range(len(runs)):
        runs[i]['cput'] = np.cumsum(runs[i]['cpudt'])

    # Simulated days-per-day (SDPD).
    for i in range(len(runs)):
        dtime = runs[i]['time'][1:] - runs[i]['time'][:-1]
        runs[i]['sdpd'] = np.zeros_like(runs[i]['time'])
        runs[i]['sdpd'][1:] = dtime / runs[i]['cpudt'][1:]

    return runs