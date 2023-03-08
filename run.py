from lazevo.piza import Lazevo
from lazevo.plotter import plot_universe_to_file, plot_universe_trajectory_to_file
from lazevo.utils import parse_params

import matplotlib.pyplot as plt
import numpy as np
import humanize


if __name__ == '__main__':
    params = parse_params()

    #TODO: implement sample dillution

    #TODO: Calculate coefficients for mode “t” only

    # with open(params['paths']['input']['particles_positions']) as f:
    #     #TODO: use numpy method for reading files
    #     particles = np.array([[float(q) for q in line.rstrip().split()] for line in f])

    # if params['paths']['input']['reconstructions'] is not None:
    #     with open(params['paths']['input']['reconstructions'][0]) as f:
    #         #TODO: use numpy method for reading files
    #         reconstruction = np.array([[float(q) for q in line.rstrip().split()] for line in f])
    #     lazevo = Lazevo(
    #         particles, 
    #         [reconstruction],
    #         sigma=params['sigma']
    #     )
    # else:
    #     lazevo = Lazevo.from_particles(
    #         particles, 
    #         params['n_reconstructions'],
    #         sigma=params['sigma']
    #     )

    lazevo = Lazevo.load_piza_execution('output/piza_dump.json')
    
    x, y, z = lazevo.universe.sizes
    n_particles = len(lazevo.universe.particles)
    mean_squared_displacement = humanize.scientific(lazevo.reconstructions[0].mean_squared_displacement)
    print(
        f"Input universe parameters\n"
        f"  Sizes\n"
        f"    X: {x}\n"
        f"    Y: {y}\n"
        f"    Z: {z}\n"
        f"  Particles: {n_particles}\n"
        f"\n"
        f"Mean squared displacement (reconstruction #1): {mean_squared_displacement}\n"
    )

    # for idx, reconstruction in enumerate(lazevo.reconstructions):
    #     squared_displacements = np.linalg.norm(reconstruction.displacements, axis=1) ** 2
    #     plt.clf()
    #     plt.hist(squared_displacements, bins=20)
    #     plt.xlabel('Vector length')
    #     plt.ylabel('Frequency')
    #     plt.title('Distribution of vector lengths')
    #     plt.savefig(f'output/squared_displacements_{idx}.png')

    #     # Compute the pairwise distances between particles
    #     distances = np.sqrt(((reconstruction.particles[:, None, :] - reconstruction.particles) ** 2).sum(axis=2))

    #     # Define the distance bins
    #     r_min = 0
    #     r_max = np.max(distances)
    #     num_bins = 20
    #     bins = np.linspace(r_min, r_max, num_bins + 1)

    #     # Count the number of pairs in each bin
    #     counts, bin_edges = np.histogram(distances, bins=bins)

    #     # Compute the bin centers and normalize the counts
    #     bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    #     norm_counts = counts / (len(reconstruction.particles) * (len(reconstruction.particles) - 1))

    #     # Calculate the two-point correlation function
    #     delta_r = bin_centers[1] - bin_centers[0]
    #     r_sq = bin_centers ** 2
    #     xi = (norm_counts - 1) / (4 * np.pi * r_sq * delta_r)

    #     # Plot the two-point correlation function
    #     plt.clf()
    #     plt.plot(bin_centers, xi, 'o-')
    #     plt.xlabel('Distance (Mpc)')
    #     plt.ylabel('Two-point correlation function')
    #     plt.title('Two-point correlation function of particles')
    #     plt.savefig(f'output/autocorrelation_{idx}.png')

    plot_universe_to_file(
        params['paths']['output']['universe_fig'], 
        lazevo.universe,
        start=params['visualization']['start'],
        end=params['visualization']['end']
    )

    # lazevo.piza(params['n_iters'])
    # lazevo.dump_piza_execution('output/piza_dump.json')

    # Plot universe trajectories
    for idx, reconstruction in enumerate(lazevo.reconstructions):
        plot_universe_trajectory_to_file(
            params['paths']['output']['trajectory_fig_prefix'] + str(idx),
            reconstruction,
            start=params['visualization']['start'],
            end=params['visualization']['end']
        )

    particles, displacement_quiver = lazevo.visualization(
        start=params['visualization']['start'],
        end=params['visualization']['end'],
        grid_step=params['visualization']['grid_step'],
    )
    plt.clf()
    plt.plot(*particles, ',k')
    plt.quiver(*displacement_quiver, color='red', width=0.002)
    plt.savefig('output/lazevo_results.png')
