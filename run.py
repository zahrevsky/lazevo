from lazevo.piza import UniverseTrajectory, piza, read_universe, random_init_universe
from lazevo.plotter import plot_universe_to_file, plot_universe_trajectory_to_file
from lazevo.utils import parse_params


if __name__ == '__main__':
    params = parse_params()

    universe = read_universe(params['paths']['input']['particles_positions'])

    #TODO: implement sample dillution

    #TODO: Calculate coefficients for mode “t” only

    plot_universe_to_file(params['paths']['output']['universe_fig'], universe)

    # Mapping from random origins to a given ends for each realization. 
    # Mapping is random at first, and the whole job of PIZA is to make it 
    # satisfy least action principle.
    #TODO: Read random positions from file, if provided
    realizations = [
        UniverseTrajectory(universe.particles, random_init_universe(universe))
        for _ in range(params['n_realizations'])
    ]

    print("Running PIZA...")
    piza(realizations, n_iters=params['n_iters'])

    # Plot universe trajectories
    for idx, realization in enumerate(realizations):
        plot_universe_trajectory_to_file(
            params['paths']['output']['trajectory_fig_prefix'] + str(idx), 
            realization
        )
