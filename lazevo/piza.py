"""
Provides PIZA algorithm implementation, as well as classes to represent data structures.
"""
from typing import Optional, Union
import json
from datetime import datetime

import numpy as np
from tqdm import trange, tqdm
import humanize


class Universe:
    """Universe class represents the universe of particles.
    
    Primarily purpose is to store data: particle positions and boundaries.
    It also has a method slice, to extract a sub-universe from the current universe.
    """
    def __init__(self, particles):
        self.particles = np.asarray(particles)
        self.min_coords = np.min(self.particles, axis=0)
        self.max_coords = np.max(self.particles, axis=0)
        self.sizes = self.max_coords - self.min_coords

    def slice(self, axis: int, start: float, end: float):
        """Slices a universe along the given axis.

        Extract a sub-universe, containing all particles with value 
        from a range (start, end) for a given axis.
        
        Args:
            axis: Axis to slice on, either 0 (x-axis), 1 (y-axis), or 2 (z-axis).
            start: Starting coordinate of the slice, normalized to the length universe takes along this coordinate. So, 0.0 corresponds to the minimal coordinate value, and 1.0 corresponds to maximal coordinate value.
            end: End coordinate of the slice. Same as with start: the value is in the rage from 0 to 1.
        
        Returns:
            A new Universe instance, representing the sub-universe.
        """
        mask = np.logical_and(
            start * self.sizes[axis] <= self.particles[:, axis] - self.min_coords[axis],
            end * self.sizes[axis] >= self.particles[:, axis] - self.min_coords[axis]
        )
        return Universe(self.particles[mask])
    
    def pairwise_distances(self):
        # calculate pairwise distances between each row of X
        for i in range(self.particles.shape[0]):
            for j in range(i+1, self.particles.shape[0]):
                yield np.sqrt(((self.particles[i] - self.particles[j]) ** 2).sum())

    @property
    def mean_distance(self):
        # calculate mean distance between all pairs of particles
        distances = self.pairwise_distances()
        return sum(distances) / (len(self.particles) * (len(self.particles) - 1) / 2)


class Reconstruction:
    """Does PIZA and represents the trajectory of all the particles in the universe.
    
    It holds information about the initial and actual positions of the particles 
    as well as an action of the system
    
    Unlike Universe, it not only holds information, but also allows to perform
    PIZA algorithm step.

    Attributes:
        pairs: A dictionary where keys are initial positions of particles and values are their final positions.
        min_coords: Minimum values of x, y and z coordinates.
        max_coords: Maximum values of x, y and z coordinates.
        sizes: Sizes of the universe along each axis.
        action: The sum of each point's displacements squared, a. k. a. ∑ψᵢ².

    Args:
        particles: List of particles in the trajectory.
        init_positions: List of initial positions of the particles. 
    """
    def __init__(self, particles, init_positions):
        self.particles = np.asarray(particles)
        self.init_positions = np.asarray(init_positions)
        self.pairs = np.stack((particles, init_positions), axis=1)

        # Boundaries
        self.min_coords = np.min(particles, axis=0)
        self.max_coords = np.max(particles, axis=0)
        self.sizes = self.max_coords - self.min_coords

        # Not an action, actually, but action is proportional to it. This is
        # a sum of each point's displacements squared, a. k. a. ∑ψᵢ²
        # By minimizing it, you minimize an action of a system.
        self.action = np.sum(np.linalg.norm(self.init_positions - self.particles, axis=1)**2)

        # Used to prioritize displacements with high deviation from mean
        # when doing PIZA
        self.mean_squared_displacement = np.mean(np.linalg.norm(self.displacements, axis=1) ** 2)

    @property
    def displacements(self) -> np.ndarray:
        """
        Returns:
            Displacement of each particle from its initial position to final position.
        """
        return self.particles - self.init_positions

    def slice(self, axis: int, start: float, end: float):
        """See Universe.slice."""
        mask = np.logical_and(
            start * self.sizes[axis] <= self.particles[:, axis] - self.min_coords[axis],
            end * self.sizes[axis] >= self.particles[:, axis] - self.min_coords[axis]
        )
        return Reconstruction(self.particles[mask], self.init_positions[mask])

    def do_piza_step(self) -> bool:
        """Performs a step of PIZA algorithm.
        
        Picks randomly two particles and swaps their initial positions, if it 
        decreases total action.

        Returns:
            bool: True, if swapping was done, False otherwise
        
        """
        np_rnd = np.random.default_rng()
        # weights = np.abs(
        #     self.mean_squared_displacement \
        #     - np.linalg.norm(self.displacements, axis=1)
        # )
        # norm_weights = weights / np.sum(weights)
        idx1, idx2 = np_rnd.choice(len(self.pairs), size=2, replace=False)
        trajectory1, trajectory2 = self.pairs[idx1], self.pairs[idx2]
        end1, origin1 = trajectory1
        end2, origin2 = trajectory2

        action_delta = np.linalg.norm(origin1 - end2) ** 2 \
            + np.linalg.norm(origin2 - end1) ** 2 \
            - np.linalg.norm(origin1 - end1) ** 2 \
            - np.linalg.norm(origin2 - end2) ** 2

        if action_delta < 0:
            self.pairs[[idx1,idx2],1] = self.pairs[[idx2,idx1],1]
            self.init_positions[[idx1, idx2]] = self.init_positions[[idx2, idx1]]
            self.action = self.action + action_delta
            return True
        return False


class Lazevo:
    """Lagrangian-Zeldovich Void Finder.

    Multiple reconstructions of Universe. Initial positions 
    in each Reconstruction are random at first, and the whole job of PIZA 
    is to make them satisfy least action principle.
    """
    def __init__(self, 
                 particles: Union[np.ndarray, Universe], 
                 reconstructions: Union[np.ndarray, list[Reconstruction]],
                 sigma: float):
        if isinstance(particles, Universe):
            self.universe = particles
        else:
            self.universe = Universe(particles)
        
        if isinstance(reconstructions, list) and isinstance(reconstructions[0], Reconstruction):
            self.reconstructions = reconstructions
        else:
            self.reconstructions = [
                Reconstruction(self.universe.particles, reconstruction)
                for reconstruction in reconstructions
            ]
        
        self.sigma = sigma
    
    @classmethod
    def from_file(cls, filename, n_reconstructions: int, sigma: float):
        """Read the universe from the given file path.
        
        Args:
            path: Path to the file
        
        Returns:
            A Universe instance, representing the universe read from the file
        """
        with open(filename) as f:
            #TODO: use numpy method for reading files
            particles = np.array([[float(q) for q in line.rstrip().split()] for line in f])
        return cls.from_particles(particles, n_reconstructions, sigma)
    
    @classmethod
    def from_particles(cls, 
                       particles: np.ndarray, 
                       n_reconstructions: int, 
                       sigma: float):
        particles = np.asarray(particles)
        # sorted_particles = particles[np.lexsort((particles[:,2], particles[:,1], particles[:,0]))]
        # universe = Universe(sorted_particles)
        universe = Universe(particles)

        np_rnd = np.random.default_rng()
        unordered_init_positions = [
            np_rnd.uniform(
                low=universe.min_coords,
                high=universe.max_coords,
                size=universe.particles.shape
            )
            for _ in range(n_reconstructions)
        ]
        init_positions = [
            min(
                unordered_init_positions[i:len(unordered_init_positions)], 
                key=lambda p: np.linalg.norm(p - particles[i]) ** 2
            )
            for i in range(len(unordered_init_positions))
        ]
        # sorted_init_positions = [
        #     init_pos[np.lexsort((init_pos[:,2], init_pos[:,1], init_pos[:,0]))]
        #     for init_pos in unsorted_init_positions
        # ]

        return cls(universe, init_positions, sigma)
    
    @classmethod
    def load_piza_execution(cls, filename):
        with open(filename) as inp:
            piza_exec_data = json.load(inp)
        
        return Lazevo(
            piza_exec_data['universe'],
            piza_exec_data['reconstructions'],
            piza_exec_data['sigma']
        )
    
    def dump_piza_execution(self, filename):
        piza_exec_data = {
            'universe': self.universe.particles.tolist(),
            'reconstructions': [r.init_positions.tolist() for r in self.reconstructions],
            'n_iterations': self.n_iters, # See Lazevo.piza for details
            'started_at': self.started_at,
            'ended_at': self.ended_at,
            'start_actions': self.start_actions,
            'end_actions': [r.action for r in self.reconstructions],
            # Technically, sigma is not a part of PIZA. See Lazevo.piza comment
            'sigma': self.sigma
        }
        with open(filename, 'w') as out:
            json.dump(piza_exec_data, out, default=str)

    def piza(self, n_iters: int = 10):
        """Minimizing action by Path Interchange Zeldovich Approximation (PIZA).

        Runs PIZA for several Reconstruction objects

        Args:
            n_iters: Number of iterations to perform in the minimization process
        """

        # Needed in Lazevo.dump_piza_execution. It is a strong code smell, 
        # because, ideally, there should be a separate PizaExecution class, that
        # is __init__ed with n_iters and stores all those variables.
        self.n_iters = n_iters
        self.started_at = datetime.now()
        self.start_actions = [r.action for r in self.reconstructions]

        n_swaps = 0
        pbar = trange(n_iters, desc="PIZA")
        for iteration_idx in pbar: #TODO: Implement more quit strategies
            for reconstruction in self.reconstructions:
                was_swapped = reconstruction.do_piza_step()
                if was_swapped:
                    n_swaps += 1/ len(self.reconstructions)
            mean_swapping_rate = n_swaps / (iteration_idx + 1) * 100
            avg_psi = np.average([r.action for r in self.reconstructions])
            pbar.set_description(
                f"PIZA (avg ψ = {humanize.scientific(avg_psi)}, "
                f"swapping rate: {mean_swapping_rate:.2f}%)"
            )
        
        self.ended_at = datetime.now()

    
    def visualization(self, 
        start: float, end: float, grid_step: float, 
        axis: int = 0, field_at: Optional[float] = None) -> tuple:
        """Representation of the universe and algorithm results for plotting.

        Returns particles and, optionally, the displacement field from 
        a predefined slice of the universe. Returned values are 
        matplotlib-friendly — plug them into plt.plot and plt.quiver.

        Displacement field is available only after running ``Lazevo.piza`` 
        method, otherwise it's None.

        Examples:
            Example 1::
                >>> particles, _ = lazevo.visualization()
                >>> plt.plot(*particles) # Plots a slice of the universe
            
            Example 2::
                >>> # After running PIZA algorightm
                >>> particles, displacements = lazevo.visualization()
                >>> plt.plot(*particles) # Plots a slice of the universe
                >>> plt.quiver(*displacements) # Plots the displacement field

        Args:
            axis (int): (not working) the axis along which to slice.
                Possible values are 0, 1 and 2, meaning 'x', 'y' and 'z'. 
                Currently only default value 0 is working correctly.
            start (float): the start of the slice. 
                In fractions of size along the axis, so value is in range [0, 1]
            end (float): the end of the slice.
                In fractions of size along the axis, so value is in range [0, 1]
            grid_step (float):
                Distance between neighbour points in a square grid, which will 
                be used to calculate displacement. Given in fractions of size 
                along the axis, so this value is in range [0, 1].
            field_at (float, optional): 
                The value of `axis` coordinate, at which displacement field 
                grid will be calculated. If None, will be the middle of 
                (start, end). Note, that displacement grid will be None, if
                PIZA wasn't runned.
    
        Returns:
            A tuple, containing two items: 
            - particles(np.ndarray): a tuple of xs and ys of the particles
            - displacements(tuple): a tuple, ready to be plugged into
              ``matplotlib.pyplot.quiver``. Represents an equally-spaced 2D grid 
              and displacements vectors, calculated for every point on the grid, 
              using ``Lazevo.avg_displacement_at``. Has 4 items:
                - x(np.ndarray): x values of every point in the grid
                - y(np.ndarray): y values of every point in the grid
                - u(np.ndarray): x-components of every displacement vector
                - v(np.ndarray): y-components of every displacement vector
        """
        if field_at is None:
            field_at = (start + end) / 2

        universe_slice = self.universe.slice(axis, start, end)
        #TODO: explain this $N - axis$ trick
        slice_particles = universe_slice.particles[:, [1 - axis, 2 - axis]]

        # Early return if there are no displacements
        if self.reconstructions[0].displacements is None:
            return slice_particles, None

        # Construct a grid of 3D points. Each point's `axis` axis has 
        # a value `field_at` and other two axes are constrcted from 
        # an np.meshgrid from start*size to end*size with 
        # distance grid_step*size.
        #
        # xs and ys are meant in newly constructed 2D plot and have 
        # nothing to do with `lazevo.universe` coordinate system
        xs, ys = np.meshgrid(
            np.arange(0, 1, grid_step) * self.universe.sizes[1 - axis] + self.universe.min_coords[1 - axis],
            np.arange(0, 1, grid_step) * self.universe.sizes[2 - axis] + self.universe.min_coords[2 - axis]
        )
        grid = np.stack((
            #TODO: `axis` must determine the order of those 3 arrays
            np.full(xs.shape, field_at * self.universe.sizes[axis] + self.universe.min_coords[axis]),
            xs,
            ys
        ), axis=2).reshape(-1, 3)

        # Calculate displacements at every point in the grid
        displacements = np.array([
            self.avg_displacement_at(point)
            for point in tqdm(grid, desc="Displacement field")
        ])
        # displacements = np.stack([
        #     self.avg_displacement_at(point)
        #     for point in tqdm(grid, desc="Bulding displacement grid for a given slice...")
        # ], axis=0)

        particle_xs, particle_ys = slice_particles.T
        xs, ys = grid[:, [1 - axis, 2 - axis]].T
        u, v = displacements[:, [1 - axis, 2 - axis]].T
        return (particle_xs, particle_ys), (xs, ys, u, v)


    def avg_displacement_at(self, point) -> np.ndarray:
        """
        Averaged displacement at arbitrary point is given by:
        1 / V(point) * Σ( a(point, p) * p.displacement )
        [Σ is over all particles `p` in all reconstructions]
        Where:
            V(p) = Σ a(point, p)
            a(p, point) = e^( - d(p, point)^2 / (2 * self.sigma^2) )
            d(p, point) - distacne between points p and point
        
        Args:
            point: numpy 1d array of 3 floats
        """
        point = np.asarray(point)

        #TODO: check if point is in universe boundaries

        all_init_positions = np.concatenate([r.init_positions for r in self.reconstructions], axis=0)
        all_particles = np.concatenate([r.particles for r in self.reconstructions], axis=0)
        all_displacements = all_particles - all_init_positions

        a = np.exp(-1 * np.linalg.norm(point - all_particles, axis=1) ** 2 / (2 * self.sigma ** 2))
        V = np.sum(a)
        avg_displacement_at_point = np.sum(a[:, np.newaxis] * all_displacements, axis=0) / V
        return avg_displacement_at_point

    def avg_displacements_on_grid(self, n_steps):
        # Create the 3D grid
        min_coords = self.universe.min_coords
        max_coords = self.universe.max_coords

        x = np.linspace(min_coords[0], max_coords[0], n_steps)
        y = np.linspace(min_coords[1], max_coords[1], n_steps)
        z = np.linspace(min_coords[2], max_coords[2], n_steps)
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        grid = np.vstack([xx.flatten(), yy.flatten(), zz.flatten()]).T

        def calc_avg_and_upd_progres_bar(point, bar):
            bar.update(1)
            return self.avg_displacement_at(point)

        with tqdm(total=len(grid)) as bar:
            avgs_on_grid = np.apply_along_axis(
                lambda p: calc_avg_and_upd_progres_bar(p, bar), 
                axis=1, arr=grid
            )

        return grid, avgs_on_grid
