"""
This module contains functions for plotting and saving 2D slices of Universes using matplotlib.

Those methods could be a part of Universe and UniverseTrajectory classes, but factoring them out
helps to distribute responsibility, as plotting and minimizing action are two distinct tasks.
"""

import matplotlib.pyplot as plt


def plot_projection_to_file(filename, particles, displacements=None):
    """Plots the 2D projection of the given particles and their displacements, and saves the plot to a file.
    
    Args:
        filename (str): the name of the file to save the plot to.
        particles (list of tuples): a list of 2D tuples representing the particles to be plotted.
        displacements (list of tuples, optional): a list of 2D tuples representing the displacements of the particles. If not provided, only the particles will be plotted.
    """
    plt.clf()

    xs, ys = list(zip(*particles))
    plt.plot(xs, ys, 'k,')

    if displacements is not None:
        # Scale for visual clarity. plt.quiver scale argument
        # doesn't work for some reason. 
        # TODO: switch to plt.quiver scale
        displacements = [
            (0.05*q for q in displacement) 
            for displacement in displacements
        ]
        us, vs = list(zip(*displacements))
        plt.quiver(
            xs, ys, us, vs, 
            color='red', angles='xy', scale_units='xy', scale=1
        )
    
    plt.savefig(filename)


def project(item_3d, axis=0):
    """Projects a 3D point onto a 2D plane, perpendicular to the given axis.
    
    Args:
        item_3d (tuple): a 3D tuple to be projected.
        axis (int, optional): the axis to drop. The default is 0.
    
    Returns:
        tuple: a 2D tuple representing the projection of the 3D item.
    """
    if axis == 0:
        return (item_3d[1], item_3d[2])
    elif axis == 1:
        return (item_3d[0], item_3d[2])
    elif axis == 2:
        return (item_3d[0], item_3d[1])


def plot_universe_trajectory_to_file(filename, universe_trajectory, axis=0, start=0.1, end=0.1625):
    """Plots the 2D projection of a slice of a universe trajectory, and saves the plot to a file.

    Similar to `plot_universe_to_file` but also draws displacements.
    
    Args:
        filename (str): the name of the file to save the plot to.
        universe_trajectory (lazevo.piza.UniverseTrajectory): the universe trajectory to plot.
        axis (int, optional): the axis to drop. The default is 0.
        start (float, optional): the start of the slice. The default is 0.1.
        end (float, optional): the end of the slice. The default is 0.1625.
    """
    universe_trajectory_slice = universe_trajectory.slice(axis, start, end)

    particles_proj = [project(p) for p in universe_trajectory_slice.particles]
    displacements_proj = [project(d) for d in universe_trajectory_slice.displacements]
    
    plot_projection_to_file(
        filename,
        particles_proj,
        displacements=displacements_proj
    )


def plot_universe_to_file(filename, universe, axis=0, start=0.1, end=0.1625):
    """
    Plots the 2D projection of a slice of a universe, and saves the plot to a file.

    Args:
        filename (str): the name of the file to save the plot to.
        universe (lazevo.piza.Universe): the universe to plot.
        axis (int, optional): the axis to drop. The default is 0.
        start (float, optional): the start of the slice. The default is 0.1.
        end (float, optional): the end of the slice. The default is 0.1625.
    """
    particles = universe.slice(axis, start, end).particles
    #TODO: rewrite project() as Universe method: 
    # universe.slice(axis, start, end).project('x').particles
    particles_proj = [project(p) for p in particles]
    
    plot_projection_to_file(
        filename,
        particles_proj
    )