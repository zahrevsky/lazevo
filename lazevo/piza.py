"""
Provides PIZA algorithm implementation, as well as classes to represent data structures.
"""


import random

from tqdm import trange

from lazevo.utils import dist_squared


class Universe:
    """Universe class represents the universe of particles.
    
    Primarily purpose is to store data: particle positions and boundaries.
    It also has a method slice, to extract a sub-universe from the current universe.
    """
    def __init__(self, particles):
        self.particles = particles
        self.min_coords = [
            min([p[0] for p in self.particles]),
            min([p[1] for p in self.particles]),
            min([p[2] for p in self.particles])
        ]
        self.max_coords = [
            max([p[0] for p in self.particles]),
            max([p[1] for p in self.particles]),
            max([p[2] for p in self.particles])
        ]
        self.sizes = [
            self.max_coords[0] - self.min_coords[0],
            self.max_coords[1] - self.min_coords[1],
            self.max_coords[2] - self.min_coords[2]
        ]

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
        return Universe([
            p for p in self.particles 
            if start * self.sizes[axis] <= p[axis] - self.min_coords[axis] <= end * self.sizes[axis]
        ])


class UniverseTrajectory:
    """Does PIZA and represents the trajectory of all the particles in the universe.
    
    It holds information about the initial and actual positions of the particles 
    as well as an action of the system
    
    Unlike Universe, it not only holds information, but also allows to perform
    PIZA algorithm step.

    Attributes:
        pairs (Dict[Tuple[float, float, float], Tuple[float, float, float]]): A dictionary where keys are initial positions of particles and values are their final positions.
        min_coords (List[float, float, float]): Minimum values of x, y and z coordinates.
        max_coords (List[float, float, float]): Maximum values of x, y and z coordinates.
        sizes (List[float, float, float]): Sizes of the universe along each axis.
        action (int): The sum of each point's displacements squared, a. k. a. ∑ψᵢ².

    Args:
        particles: List of particles in the trajectory.
        init_positions: List of initial positions of the particles. 
    """
    def __init__(self, particles, init_positions):
        #TODO: either swap keys and values in pairs, or even consider avoiding ambiguity, by storing them separately
        self.pairs = {
            init_position: particle
            for init_position, particle in zip(init_positions, particles)
        }

        # Boundaries
        self.min_coords = [
            min([p[0] for p in self.particles]),
            min([p[1] for p in self.particles]),
            min([p[2] for p in self.particles])
        ]
        self.max_coords = [
            max([p[0] for p in self.particles]),
            max([p[1] for p in self.particles]),
            max([p[2] for p in self.particles])
        ]
        self.sizes = [
            self.max_coords[0] - self.min_coords[0],
            self.max_coords[1] - self.min_coords[1],
            self.max_coords[2] - self.min_coords[2],
        ]

        # Not an action, actually, but action is proportional to it. This is
        # a sum of each point's displacements squared, a. k. a. ∑ψᵢ²
        # By minimizing it, you minimize an action of a system.
        self.action = sum([
            dist_squared(origin, end) for origin, end in self.pairs.items()
        ])

    @property
    def particles(self):
        """
        Returns:
            List[Tuple[float, float, float]]: The final positions of particles.
        """
        return self.pairs.values()

    @property
    def init_positions(self):
        """
        Returns:
            List[Tuple[float, float, float]]: The initial positions of particles.
        """
        return self.pairs.keys()

    @property
    def displacements(self):
        """
        Returns:
            List[Tuple[float, float, float]]: Displacement of each particle from its initial position to final position.
        """
        return [
            (
                position[0] - init_position[0],
                position[1] - init_position[1],
                position[2] - init_position[2]
            )
            for init_position, position in self.pairs.items()
        ]

    def slice(self, axis, start, end):
        """See Universe.slice."""
        slice_raw = {
            init_position: particle
            for init_position, particle in self.pairs.items()
            if start * self.sizes[axis] <= particle[axis] - self.min_coords[axis] <= end * self.sizes[axis]   
        }
        return UniverseTrajectory(slice_raw.values(), slice_raw.keys())

    def do_piza_step(self):
        """Picks randomly two particles and swaps their initial positions, if it decreases total action."""
        trajectory1, trajectory2 = random.sample(list(self.pairs.items()), k=2)
        origin1, end1 = trajectory1
        origin2, end2 = trajectory2

        action_delta = dist_squared(origin1, end2) \
            + dist_squared(origin2, end1) \
            - dist_squared(origin1, end1) \
            - dist_squared(origin2, end2)

        if action_delta < 0:
            self.pairs[origin1], self.pairs[origin2] = end2, end1
            self.action = self.action + action_delta


def piza(realizations: list[UniverseTrajectory], n_iters=10):
    """Minimizing action by Path Interchange Zeldovich Approximation (PIZA).

    Runs PIZA for several UniverseTrajectory objects

    Args:
        realizations: Each realization is a UniverseTrajectory with a unique initial positions
        n_iters: Number of iterations to perform in the minimization process
    """

    for curr_iter in trange(n_iters): #TODO: Implement more quit strategies
        for realization in realizations:
            realization.do_piza_step()


def read_universe(path):
    """Read the universe from the given file path.
    
    Args:
        path: Path to the file.
    
    Returns:
        A Universe instance, representing the universe read from the file.
    """
    with open(path) as f:
        particle_positions = [[float(q) for q in line.rstrip().split()] for line in f]
    return Universe(particle_positions)


def random_init_universe(universe):
    """Generates random initial positions for each particle from the given Universe object.

    Returns:
        List[Tuple[float, float, float]]: List, containing the same number 
        of elements, as universe.particles does. Each element is a 3-item 
        tuple with random coordinates (x, y, z). Generated coordinates are 
        from range (universe.min_coords, universe.max_coords). So, no random 
        coordinate is larger or smaller than any given coordinate from Universe.

    """
    random_positions = [
        (
            random.uniform(universe.min_coords[0], universe.max_coords[0]),
            random.uniform(universe.min_coords[1], universe.max_coords[1]),
            random.uniform(universe.min_coords[2], universe.max_coords[2])
        )
        for _ in range(len(universe.particles))
    ]

    return random_positions
