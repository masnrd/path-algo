import numpy as np
import h3
import random

from pathfinder.interface import PathFinder
from utils.hex import *


class TestFramework:
    def __init__(self, name: str, centre: tuple[float, float], res: int, f: float = None):
        self.name = name
        self.centre = centre
        self.res = res
        # TODO: Decide what is the ideal initialized map size
        self.probability_map = self.initialize_probability_map(res)
        self.waypoint = centre
        self.f = f
        self.pathfinder = None
        self.output = []
        self.steps = None
        self.update_probability_map()

    def run(self, steps: int, update_map: bool = False):
        if not self.pathfinder:
            raise ValueError("Please Register your Pathfinder first")

        for i in range(steps):
            if update_map:
                self.update_probability_map()

            self.waypoint = self.pathfinder.find_next_step(
                self.waypoint, self.probability_map)
            self.output.append({"hex_idx": h3.geo_to_h3(
                self.waypoint[0], self.waypoint[1], self.res), "step_count": i})

        self.steps = steps
        return self.output

    def register_pathfinder(self, pathfinder: PathFinder):
        self.pathfinder = pathfinder(self.res, self.centre)

    def initialize_probability_map(self, n_rings):

        # Dictionary for the hexagons
        probability_map = {}
        all_hex = h3.k_ring(h3.geo_to_h3(
            self.centre[0], self.centre[1], self.res), n_rings)
        for hex in all_hex:
            probability_map[hex] = random.random()
        return probability_map

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
        hex_waypoint = h3.geo_to_h3(
            self.waypoint[0], self.waypoint[1], self.res)

        # Prior
        prior = self.probability_map[hex_waypoint]

        # Posterior
        posterior = prior*(1-self.f) / (1-prior*self.f)
        self.probability_map[hex_waypoint] = posterior

        # Distribute
        total_prob = sum(self.probability_map.values())
        if total_prob != 0:
            self.probability_map = {
                key: value / total_prob for key, value in self.probability_map.items()}
        else:
            print("Entire probability map is zero")

    def evaluate(self):
        path_coverage = round(len(self.output) / self.steps * 100, 2)
        print(f"{self.name}'s Path Coverage: {path_coverage}%")
