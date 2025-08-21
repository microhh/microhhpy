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

import logging
from colorlog import ColoredFormatter

logger = logging.getLogger("microhhpy")
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    formatter = ColoredFormatter(
        "[%(asctime)s] [%(name)s] %(log_color)s[%(levelname)s] %(message)s'\033[0m",
        datefmt="%Y/%m/%d %H:%M:%S",
        log_colors={
            "DEBUG": "fg_244",
            "INFO": "",  # No color for INFO
            "WARNING": "fg_208",
            "ERROR": "red",
            "CRITICAL": "red",
        },
    )

    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    def critical_exception(message, *args, **kwargs):
         logger._log(logging.CRITICAL, message, args, **kwargs)
         raise RuntimeError(message)

    logger.critical = critical_exception
