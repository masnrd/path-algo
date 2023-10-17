import numpy as np
import h3

from pathfinder.interface import PathFinder
from utils.hex import *

class TestFramework:
    def __init__(self, name: str, waypoint: tuple[float, float], res: int, f: float = None):
        self.name = name
        self.probability_map = self.initialize_probability_map()
        self.waypoint = waypoint
        self.res = res
        self.f = f
        self.pathfinder = None
        self.output = dict()
        self.steps = None

    def run(self, steps: int, update_map: bool = False):
        if not self.pathfinder:
            raise ValueError("Please Register your Pathfinder first")

        for i in range(steps):
            if update_map:
                self.update_probability_map()

            self.waypoint = self.pathfinder.find_next_step(self.waypoint, self.probability_map)
            self.output[h3.geo_to_h3(self.waypoint[0], self.waypoint[1], self.res)] = (steps-i)/steps

        self.steps = steps
        return self.output

    def register_pathfinder(self, pathfinder: PathFinder):
        self.pathfinder = pathfinder(self.res, self.waypoint)

    def initialize_probability_map(self):

        fake_map = np.random.rand(7, 7, 7, 7)
        fake_map /= fake_map.sum()
        return fake_map

    def update_probability_map(self):
        """Update the probability map using Baye's theorem.

        Args:
            prob_map (np.ndarray): A numpy array of (7,7,7,7) representing the probability in each hexagon.
            waypoint (tuple[float, float]): Current search position as a tuple of (latitude, longitude).
            res (_type_): The H3 resolution of the hexagon associated with the position.
            f (_type_): The probability of detecting a person.

        Returns:
            np.ndarray: The updated probability map.
        """
        hex_waypoint = h3.geo_to_h3(self.waypoint[0], self.waypoint[1], self.res)
        array_index_waypoint = hex_to_array_index(hex_waypoint, self.probability_map)

        # Prior
        prior = self.probability_map[array_index_waypoint]

        # Posterior
        posterior = prior*(1-self.f) / (1-prior*self.f)
        self.probability_map[array_index_waypoint] = posterior

        # Disrtribute
        self.probability_map /= self.probability_map.sum()

    def evaluate(self):
        path_coverage = round(len(self.output) / self.steps * 100, 2)
        print(f"{self.name}'s Path Coverage: {path_coverage}%")