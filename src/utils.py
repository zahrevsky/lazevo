import argparse
import random

import yaml


def parse_params():
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


def dist_squared(p1, p2) -> float:
    return (p2[0] - p1[0]) ** 2 \
         + (p2[1] - p1[1]) ** 2 \
         + (p2[2] - p1[2]) ** 2