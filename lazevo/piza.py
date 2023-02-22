"""
Provides PIZA algorithm implementation, as well as classes to represent data structures.
"""
import random

import numpy as np
from tqdm import trange, tqdm


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


class UniverseTrajectory:
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
        self.action = np.sum(np.linalg.norm(self.init_positions - self.particles, axis=1))

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
        return UniverseTrajectory(self.particles[mask], self.init_positions[mask])

    def do_piza_step(self):
        """Picks randomly two particles and swaps their initial positions, if it decreases total action."""
        np_rnd = np.random.default_rng()

        idx1, idx2 = random.sample(range(len(self.pairs)), 2)
        trajectory1, trajectory2 = self.pairs[idx1], self.pairs[idx2]
        end1, origin1 = trajectory1
        end2, origin2 = trajectory2

        action_delta = np.linalg.norm(origin1 - end2) \
            + np.linalg.norm(origin2 - end1) \
            - np.linalg.norm(origin1 - end1) \
            - np.linalg.norm(origin2 - end2)

        if action_delta < 0:
            self.pairs[[idx1, idx2]] = self.pairs[[idx2, idx1]]
            self.particles[[idx1, idx2]] = self.particles[[idx2, idx1]]
            self.init_positions[[idx1, idx2]] = self.init_positions[[idx2, idx1]]
            self.action = self.action + action_delta


class AveragedUniverseTrajectory:
    """
    Multiple reconstructions of UniverseTrajectory. Initial positions 
    of each trajectory is random at first, and the whole job of PIZA 
    is to make it satisfy least action principle.
    """
    def __init__(self, universe: Universe, n_realizaitons: int):
        self.universe = universe

        #TODO: Read random positions from file, if provided
        self.realizations = [
            UniverseTrajectory(universe.particles, np.asarray(random_init_universe(universe)))
            for _ in range(n_realizaitons)
        ]


    def piza(self, n_iters: int = 10):
        """Minimizing action by Path Interchange Zeldovich Approximation (PIZA).

        Runs PIZA for several UniverseTrajectory objects

        Args:
            realizations: Each realization is a UniverseTrajectory with a unique initial positions
            n_iters: Number of iterations to perform in the minimization process
        """

        for curr_iter in trange(n_iters): #TODO: Implement more quit strategies
            for realization in self.realizations:
                realization.do_piza_step()

    def probe(self, point) -> np.ndarray:
        """
        Averaged displacement at arbitrary point is given by:
        probe(q) = 1 / V(q) * Σ( a(q, p) * p.displacement )
        Where:
            q - arbitrary point in the Universe
            p - one of particles, Σ is over all particles in all the realizations
            V(p) = Σ a(point, p)
            a(p, q) = e^( - d(p, q)^2 / (2σ^2) )
            d(p, q) - distacne between points p and q
            σ - kernel smoothing parameter
        """
        point = np.asarray(point)

        #TODO: check if point is in universe boundaries

        all_init_positions = np.concatenate([r.init_positions for r in self.realizations], axis=0)
        all_particles = np.concatenate([r.particles for r in self.realizations], axis=0)
        all_displacements = all_particles - all_init_positions

        # 1.5 Mpc/h, assuming 80 Mpc/h = 250 000 in internal units (this is the size of the universe)
        sigma = 4687.5

        a = np.exp(-1 * np.linalg.norm(point - all_particles, axis=1) / (2 * sigma ** 2))
        V = np.sum(a)
        return a[:, np.newaxis] * all_displacements / V

    def probe_grid(self, n_steps):
        # Create the 3D grid
        min_coords = self.universe.min_coords
        max_coords = self.universe.max_coords

        x = np.linspace(min_coords[0], max_coords[0], n_steps)
        y = np.linspace(min_coords[1], max_coords[1], n_steps)
        z = np.linspace(min_coords[2], max_coords[2], n_steps)
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        grid = np.vstack([xx.flatten(), yy.flatten(), zz.flatten()]).T

        def probe_and_upd_progressbar(point, bar):
            bar.update(1)
            return self.probe(point)

        with tqdm(total=len(grid)) as bar:
            grid_probes = np.apply_along_axis(
                lambda p: probe_and_upd_progressbar(p, bar), 
                axis=1, arr=grid
            )

        return grid, grid_probes

def read_universe(path: str) -> Universe:
    """Read the universe from the given file path.
    
    Args:
        path: Path to the file.
    
    Returns:
        A Universe instance, representing the universe read from the file.
    """
    with open(path) as f:
        particle_positions = [[float(q) for q in line.rstrip().split()] for line in f]
    return Universe(particle_positions)


def random_init_universe(universe: Universe) -> np.ndarray:
    """Generates random initial positions for each particle from the given Universe object.

    Returns:
        List[Tuple[float, float, float]]: List, containing the same number 
        of elements, as universe.particles does. Each element is a 3-item 
        tuple with random coordinates (x, y, z). Generated coordinates are 
        from range (universe.min_coords, universe.max_coords). So, no random 
        coordinate is larger or smaller than any given coordinate from Universe.

    """
    random_positions = [
        [
            random.uniform(universe.min_coords[0], universe.max_coords[0]),
            random.uniform(universe.min_coords[1], universe.max_coords[1]),
            random.uniform(universe.min_coords[2], universe.max_coords[2])
        ]
        for _ in range(len(universe.particles))
    ]

    return random_positions
