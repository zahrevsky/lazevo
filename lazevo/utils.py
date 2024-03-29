"""
Includes miscelaneous helpers: parse input parameters, calculated squared distance, etc.
"""
import argparse
from typing import Any

import yaml


def parse_params() -> dict[str, Any]:
    """Reads parameters from command line and configuration file.

    Returns:
        A dict with config parameters, that might be nested. For example:

        {'paths': 
            {'input': 
                {'particles_positions': 'input/coord_LCDM.txt',
                 'random_positions': None},
             'output': 
                {'universe_fig': 'output/slice.png',
                 'trajectory_txt_prefix': 'output/coord_LCDM_',
                 'trajectory_fig_prefix': 'output/displacements_'}},
         'dillution_factor': 1,
         'mode': 'periodic',
         'n_realizations': 8,
         'n_iters': 10}
    """
    parser = argparse.ArgumentParser(
        prog = 'LaZeVo',
        description = 'Void finder based on doi:10.1093/mnras/stv043'
    )
    parser.add_argument(
        '--params-filepath', 
        default='input/params.yaml',
        required=False
    )
    params_filepath = parser.parse_args().params_filepath

    # Read config file or use defaults
    #TODO: Check that all params are OK
    #TODO: Also add as CLI args
    with open(params_filepath) as f:
        params = yaml.safe_load(f)

    return params
