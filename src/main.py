import argparse
import random

import yaml

from piza import Universe, UniverseTrajectory, piza
from plotter import *
from utils import XYZ


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

    # Read coordinates from input file
    #TODO: Should be ASCII by default, binary via flag
    #TODO: Why binary is needed at all?
    if params['coordinates_filetype'] not in ['ascii', 'binary']:
        raise ValueError(
            f"Unexpected output file format:"
            f" {params['coordinates_filetype']}. Supported formats: ascii, binary"
        )

    if params['coordinates_filetype'] == 'binary':
        raise ValueError("Binary format is not implemented yet")

    return params


def read_universe(path):
    with open(path) as f:
        particle_positions = [[float(q) for q in line.rstrip().split()] for line in f]
    return Universe(particle_positions)


def random_init_universe(universe):
    # Initialize origins randomly
    random_particles = [
        tuple(
            random.uniform(universe.min_coords[q], universe.max_coords[q])
            for q in XYZ
        )
        for _ in range(len(universe.particles))
    ]

    return random_particles


if __name__ == '__main__':
    params = parse_params()
    universe = read_universe(params['coordinates_filepath'])
    
    #TODO: implement sample dillution, if needed

    # Plot input universe
    plot_universe_to_file(
        params['output_path'] + params['coords_image_filename'], 
        universe
    )

    #TODO: Calculate coefficients for mode “t” only

    #TODO: Read random fields from file, if provided

    print("Building displacement maps...")

    # Mapping from random origins to a given ends for each realization. 
    # It's arbitrary at first, and the whole job is to make it satisfy 
    # least action principle.
    realizations = [
        UniverseTrajectory(
            universe.particles,
            random_init_universe(universe)
        )
        for _ in range(params['n_realizations'])
    ]

    piza(realizations, n_iters=params['n_iters'])

    # Plot builded displacements
    for idx, realization in enumerate(realizations):
        filename = f"{params['output_path']}" \
            f"{params['displacements_image_filename_prefix']}{idx}"

        plot_universe_trajectory_to_file(
            filename, 
            realization
        )
