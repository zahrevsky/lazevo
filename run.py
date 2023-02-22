from itertools import product

from tqdm.contrib.itertools import product as tproduct

from lazevo.piza import UniverseTrajectory, AveragedUniverseTrajectory, read_universe
from lazevo.plotter import plot_universe_to_file, plot_universe_trajectory_to_file
from lazevo.utils import parse_params


if __name__ == '__main__':
    params = parse_params()

    universe = read_universe(params['paths']['input']['particles_positions'])

    #TODO: implement sample dillution

    #TODO: Calculate coefficients for mode “t” only

    plot_universe_to_file(params['paths']['output']['universe_fig'], universe)

    print("Running PIZA...")
    averaged_universe_trajectory = AveragedUniverseTrajectory(universe, params['n_realizations'])
    averaged_universe_trajectory.piza(params['n_iters'])

    #TODO: implement aut.probe_grid() that utilizes numpy under the hood
    print("Probing 5^3 points...")
    points, vectors = averaged_universe_trajectory.probe_grid(5)

    # Plot universe trajectories
    for idx, realization in enumerate(averaged_universe_trajectory.realizations):
        plot_universe_trajectory_to_file(
            params['paths']['output']['trajectory_fig_prefix'] + str(idx), 
            realization
        )
