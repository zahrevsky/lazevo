import matplotlib.pyplot as plt


def plot_projection_to_file(filename, particles, displacements=None):
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
    if axis == 0:
        return (item_3d[1], item_3d[2])
    elif axis == 1:
        return (item_3d[0], item_3d[2])
    elif axis == 2:
        return (item_3d[0], item_3d[1])


def plot_universe_trajectory_to_file(filename, universe_trajectory, axis=0, start=0.1, end=0.1625):
    universe_trajectory_slice = universe_trajectory.slice(axis, start, end)

    particles_proj = [project(p) for p in universe_trajectory_slice.particles]
    displacements_proj = [project(d) for d in universe_trajectory_slice.displacements]
    
    plot_projection_to_file(
        filename,
        particles_proj,
        displacements=displacements_proj
    )


def plot_universe_to_file(filename, universe, axis=0, start=0.1, end=0.1625):
    particles = universe.slice(axis, start, end).particles
    #TODO: rewrite project() as Universe method: 
    # universe.slice(axis, start, end).project('x').particles
    particles_proj = [project(p) for p in particles]
    
    plot_projection_to_file(
        filename,
        particles_proj
    )