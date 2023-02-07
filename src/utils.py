import argparse
import random

import yaml

# Encoded coordinates: 0 is for x, 1 is for y, 2 is for z
# Helpful for iterating over list of triplets (x, y, z)
#TODO: move to dataclasses
XYZ = [0, 1, 2]


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


def dist_squared(begin, end) -> float:
    return sum([
        (end[q] - begin[q]) ** 2
        for q in XYZ
    ])