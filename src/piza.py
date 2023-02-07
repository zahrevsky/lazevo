import random

from tqdm import trange

from utils import XYZ, dist_squared


class Universe:
    def __init__(self, particles):
        self.particles = particles
        self.min_coords = [
            min(self.particles, key=lambda p: p[q])[q] for q in XYZ
        ]
        self.max_coords = [
            max(self.particles, key=lambda p: p[q])[q] for q in XYZ
        ]
        self.sizes = [
            self.max_coords[q] - self.min_coords[q] for q in XYZ
        ]

    def slice(self, axis, start, end):
        return Universe([
            p for p in self.particles 
            if start * self.sizes[axis] <= p[axis] - self.min_coords[axis] <= end * self.sizes[axis]
        ])


class UniverseTrajectory:
    def __init__(self, particles, init_positions):
        #TODO: either swap keys and values in pairs, or even consider avoiding ambiguity, by storing them separately
        self.pairs = {
            init_position: particle
            for init_position, particle in zip(init_positions, particles)
        }

        # Boundaries
        self.min_coords = [
            min(self.particles, key=lambda p: p[q])[q] for q in XYZ
        ]
        self.max_coords = [
            max(self.particles, key=lambda p: p[q])[q] for q in XYZ
        ]
        self.sizes = [
            self.max_coords[q] - self.min_coords[q] for q in XYZ
        ]

        # Not an action, actually, but action is proportional to it. This is
        # a sum of each point's displacements squared, a. k. a. ∑ψᵢ²
        # By minimizing it, you minimize an action of a system.
        self.action = sum([
            dist_squared(origin, end) for origin, end in self.pairs.items()
        ])

    @property
    def particles(self):
        return self.pairs.values()

    @property
    def init_positions(self):
        return self.pairs.keys()

    @property
    def displacements(self):
        return [
            (
                position[0] - init_position[0],
                position[1] - init_position[1],
                position[2] - init_position[2]
            )
            for init_position, position in self.pairs.items()
        ]

    def slice(self, axis, start, end):
        slice_raw = {
            init_position: particle
            for init_position, particle in self.pairs.items()
            if start * self.sizes[axis] <= particle[axis] - self.min_coords[axis] <= end * self.sizes[axis]   
        }
        return UniverseTrajectory(slice_raw.values(), slice_raw.keys())

    def do_piza_step(self):
        """Swap two origins if it decreases total action"""
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


def piza(realizations, n_iters=10):
    """Minimizing action by Path Interchange Zeldovich Approximation (PIZA)"""

    for curr_iter in trange(n_iters): #TODO: Implement more quit strategies
        for realization in realizations:
            realization.do_piza_step()


def read_universe(path):
    with open(path) as f:
        particle_positions = [[float(q) for q in line.rstrip().split()] for line in f]
    return Universe(particle_positions)


def random_init_universe(universe):
    random_positions = [
        tuple(
            random.uniform(universe.min_coords[q], universe.max_coords[q])
            for q in XYZ
        )
        for _ in range(len(universe.particles))
    ]

    return random_positions
