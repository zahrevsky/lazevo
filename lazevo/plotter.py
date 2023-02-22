"""
This module contains functions for plotting and saving 2D slices of Universes using matplotlib.

Those methods could be a part of Universe and UniverseTrajectory classes, but factoring them out
helps to distribute responsibility, as plotting and minimizing action are two distinct tasks.
"""

import matplotlib.pyplot as plt
import numpy as np


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

    # obj is a number of column to remove, which is the axis along which to project
    particles_proj = np.delete(universe_trajectory_slice.particles, obj=axis, axis=1)
    displacements_proj = np.delete(universe_trajectory_slice.displacements, obj=axis, axis=1)
    
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
    #TODO: rewrite projection using np.delete to a Universe method, that returns 2D slice
    # universe.slice(axis, start, end).project('x').particles
    particles_proj = np.delete(universe.slice(axis, start, end).particles, obj=axis, axis=1)
    
    plot_projection_to_file(
        filename,
        particles_proj
    )