import numpy as np
import h3
import random
from datetime import datetime

from pathfinder.interface import PathFinder
from utils.hex import *


class TestFramework:
    def __init__(self, name: str, centre: tuple[float, float], res: int, f: float = None):
        self.name = name
        self.centre = centre
        self.res = res
        # TODO: Decide what is the ideal initialized map size
        self.probability_map = self.initialize_probability_map(res)
        self.add_hotspot()

        self.waypoint = centre
        self.f = f
        self.pathfinder = None
        self.output = []
        self.casualty_locations = set()
        self.casualty_detected = dict()
        self.minimum_time_captured = None

    def run(self, steps: int, update_map: bool = False) -> list[dict]:
        """
        Simulate the drone path

        :param steps: Number of steps to simulate.
        :param update_map: Boolean to decide if probability map should be updated at each step.
        :return: A list containing dictionary of hexagon index and step count.
        """
        if not self.pathfinder:
            raise ValueError("Please Register your Pathfinder first")

        start_time = datetime.now()
        end_time = None
        for i in range(steps):
            if update_map:
                self.update_probability_map()

            self.waypoint = self.pathfinder.find_next_step(
                self.waypoint, self.probability_map)

            hex_idx = h3.geo_to_h3(self.waypoint[0], self.waypoint[1], self.res)
            if hex_idx in self.casualty_locations:
                coin = random.randint(1, 10)
                if coin == 1:
                    self.casualty_detected[hex_idx] = False
                else:
                    self.casualty_detected[hex_idx] = True
                if len(self.casualty_detected) == len(self.casualty_locations):
                    end_time = datetime.now()

            self.output.append({"hex_idx": h3.geo_to_h3(
                self.waypoint[0], self.waypoint[1], self.res), "step_count": i})
        if end_time:
            self.minimum_time_captured = end_time - start_time
            self.minimum_time_captured = self.minimum_time_captured.total_seconds()

        return self.output

    def register_pathfinder(self, pathfinder: PathFinder):
        """
        Register the pathfinder.

        :param pathfinder: An instance of PathFinder.
        """
        self.pathfinder = pathfinder(self.res, self.centre)

    def initialize_probability_map(self, n_rings: int) -> dict[str, float]:
        """
        Initialize the probability map.

        :param n_rings: Number of rings around the center hexagon.
        :return: Dictionary containing hex index as key and its probability as value.
        """
        probability_map = {}
        all_hex = h3.k_ring(h3.geo_to_h3(
            self.centre[0], self.centre[1], self.res), n_rings)
        for hex in all_hex:
            probability_map[hex] = 0
        return probability_map

    def add_hotspot(self, sigma: float = 0.00003, r_range: int = 10):
        """
        Update the probability map based on a given hotspot.

        Parameters:
        - prob_en: numpy array representing the probability map
        - hotspot: tuple containing latitude and longitude of the hotspot
        - sigma: standard deviation for the gaussian probability distribution
        - r_range: range for hex_ring (default is 10)

        Returns:
        - Updated prob_en
        """
        def gaussian_probability(distance, sigma=0.01):
            return np.exp(-distance**2 / (2 * sigma**2))

        hex_hotspot = h3.geo_to_h3(self.centre[0], self.centre[1], self.res)
        for i in range(0, r_range):
            hex_at_r = h3.hex_ring(hex_hotspot, i)
            distance = euclidean(h3.h3_to_geo(hex_hotspot), h3.h3_to_geo(next(iter(hex_at_r))))
            probability = gaussian_probability(distance, sigma)
            for hex_idx in hex_at_r:
                if hex_idx in self.probability_map.keys():
                    self.probability_map[hex_idx] += probability

        # Distribute
        total_prob = sum(self.probability_map.values())
        if total_prob != 0:
            self.probability_map = {
                key: value / total_prob for key, value in self.probability_map.items()}
        else:
            print("Entire probability map is zero")

    def generate_casualty(self, num_casualty: int):
        """
        Generate casualty locations.

        :param num_casualty: Number of casualties to be generated.
        """
        all_non_zero_cells = {key for key, value in self.probability_map.items() if value != 0}
        while len(self.casualty_locations) < num_casualty:
            probability = random.uniform(0, 1)
            random_cell = random.choice(list(all_non_zero_cells))
            if probability < self.probability_map[random_cell]:
                self.casualty_locations.add(random_cell)


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
        """
        Evaluate the performance of the pathfinder.
        """
        path_coverage = self.check_path_coverage()
        print(f"{self.name}'s Path Coverage: {path_coverage}%")

        num_casualty = len(self.casualty_locations)
        guaranteed_capture, false_negative = self.check_guaranteed_capture()
        if guaranteed_capture and false_negative == 0:
            print(f"{self.name}'s Guaranteed Capture: YES, captured {num_casualty}/{num_casualty}")
        else:
            print(f"{self.name}'s Guaranteed Capture: NO, captured {len(self.casualty_detected)}/{num_casualty}")
            if false_negative:
                print(f"{self.name}'s False Negative: {false_negative}")
        if self.check_minimum_time_captured():
            print(f"{self.name}'s Minimum Time Capture: {self.minimum_time_captured} seconds")
        else:
            print(f"{self.name}'s Minimum Time Capture: NA")


    def check_path_coverage(self) -> float:
        """
        Check the path coverage percentage.

        :return: Coverage percentage.
        """
        all_non_zero_cells = {key for key, value in self.probability_map.items() if value != 0}
        covered_cells = {item["hex_idx"] for item in self.output}
        path_covered =  all_non_zero_cells & covered_cells
        path_coverage = round(len(path_covered) / len(all_non_zero_cells) * 100, 2)
        return path_coverage

    def check_minimum_time_captured(self):
        """
        Check the minimum time for capturing all casualties.

        :return: Minimum time in seconds or False if not captured.
        """
        return self.minimum_time_captured if self.minimum_time_captured else None

    def check_guaranteed_capture(self) -> tuple[bool, int]:
        """
        Check if the pathfinder guaranteed the capture of all casualties.

        :return: Tuple containing boolean value for guaranteed capture and the number of false positives.
        """
        false_negative = 0
        for key, value in self.casualty_detected.items():
            if not value:
                false_negative += 1
        if false_negative == 0 and len(self.casualty_detected) == len(self.casualty_locations):
            guaranteed_capture = True
        else:
            guaranteed_capture = False

        return (guaranteed_capture, false_negative)
