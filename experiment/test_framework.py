import numpy as np
import h3
import random
from datetime import datetime

from pathfinder.interface import PathFinder
from utils.hex import *


class TestFramework:
    def __init__(self, name: str, centre: tuple[float, float], res: int):
        self.name = name
        self.centre = centre
        self.res = res
        # TODO: Decide what is the ideal initialized map size
        self.probability_map = self.initialize_probability_map(res)
        self.add_hotspot()

        self.waypoint = centre
        self.visited = set()
        centre_hex = h3.geo_to_h3(self.centre[0], self.centre[1], self.res)
        self.visited.add(centre_hex)
        self.pathfinder = None
        self.output = []
        self.casualty_locations = set()
        self.casualty_detected = dict()
        self.minimum_time_captured = None

    def run(self, step: int, update_map: bool = False) -> tuple[int, list[dict]]:
        """
        Simulate the drone path

        :param steps: Number of steps to simulate.
        :param update_map: Boolean to decide if probability map should be updated at each step.
        :return: A list containing dictionary of hexagon index and step count.
        """
        if not self.pathfinder:
            raise ValueError("Please Register your Pathfinder first")

        # start_time = datetime.now()
        # end_time = None
        # for i in range(steps):
        self.waypoint = self.pathfinder.find_next_step(
            self.waypoint, self.probability_map)
        zero = 1
        if update_map:
            zero, self.probability_map = self.update_probability_map(self.waypoint)
        if zero == 0:
            return True, self.output

        hex_idx = h3.geo_to_h3(self.waypoint[0], self.waypoint[1], self.res)
        self.visited.add(hex_idx)
        if hex_idx in self.casualty_locations:
            coin = random.randint(1, 10)
            if coin == 1:
                self.casualty_detected[hex_idx] = False
            else:
                self.casualty_detected[hex_idx] = True
            if len(self.casualty_detected) == len(self.casualty_locations):
                end_time = datetime.now()

        self.output.append({"hex_idx": h3.geo_to_h3(
            self.waypoint[0], self.waypoint[1], self.res), "step_count": step})
        # if end_time:
        #     self.minimum_time_captured = end_time - start_time
        #     self.minimum_time_captured = self.minimum_time_captured.total_seconds()

        return False, self.output

    def register_pathfinder(self, pathfinder: PathFinder):
        """
        Register the pathfinder.

        :param pathfinder: An instance of PathFinder.
        """
        self.pathfinder = pathfinder(self.res, self.centre)

    def initialize_probability_map(self, n_rings: int=10) -> dict[str, float]:
        """
        Initialize the probability map.

        :param n_rings: Number of rings around the center hexagon.
        :return: Dictionary containing hex index as key and its probability as value.
        """
        prob_map = {}
        h3_indices = h3.k_ring(
            h3.geo_to_h3(self.centre[0], self.centre[1], self.res),
            n_rings,
        )
        for h3_index in h3_indices:
            prob_map[h3_index] = 0
        return prob_map

    def add_hotspot(self, sigma: float = 0.00003, r_range: int = 20):
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
        def gaussian_probability(dist, sig=0.3):
            return np.exp(-dist ** 2 / (2 * sig ** 2))

        def euclidean(point1, point2):
            return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

        delta_probability_map = {}
        hotspots = random.choices(list(self.probability_map.keys()), k=4)
        hotspots = ['8f6526ac34230a1', '8f6526ac34235ad', '8f6526ac3423c08', '8f6526ac3423cb0']
        # hex_hotspot = h3.geo_to_h3(self.centre[0], self.centre[1], self.res)
        # hotspots = [hex_hotspot]

        for hex_hotspot in hotspots:
            # NOTE: Sanity check to see if hex_hotspot being added to the map is within the map size
            if hex_hotspot not in self.probability_map: print(f'Hex hotspot {hex_hotspot} not in prob_map')

            for i in range(0, r_range):
                hex_at_r = h3.hex_ring(hex_hotspot, i)
                if hex_at_r:
                    distance = euclidean(h3.h3_to_geo(hex_hotspot),
                                         h3.h3_to_geo(next(iter(hex_at_r))))
                    probability = gaussian_probability(distance, sigma)
                    for hex_idx in hex_at_r:
                        if hex_idx in self.probability_map:
                            delta_probability_map[hex_idx] = probability + delta_probability_map.get(hex_idx, 0)
                        else:
                            pass
        # Normalize the delta_probability_map
        total_delta_prob = sum(delta_probability_map.values())
        if total_delta_prob != 0:
            delta_probability_map = {key: (value / total_delta_prob) for key, value in delta_probability_map.items()}
        # Update the original probability map with the delta_probability_map
        for hex_idx, value in delta_probability_map.items():
            if hex_idx in self.probability_map:
                self.probability_map[hex_idx] += value
            else:
                print("Delta not in original prob")
                pass
        # Normalize the updated probability map
        total_prob = sum(self.probability_map.values())
        if total_prob != 0:
            self.probability_map = {key: (value / total_prob) for key, value in self.probability_map.items()}
        else:
            print("Entire probability map is zero")

        return self.probability_map

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
        self.casualty_locations = {'8f6526ac3423c91', '8f6526ac3423c0d', '8f6526ac3423c2a', '8f6526ac3422269', '8f6526ac3423502', '8f6526ac3423ca4', '8f6526ac3423516', '8f6526ac342359c', '8f6526ac3423180', '8f6526ac3423085'}


    def update_probability_map(self, curr_hex, f=1):
        hex_centre = h3.geo_to_h3(
            curr_hex[0], curr_hex[1], self.res)

        if hex_centre not in self.probability_map:
            print("Has not reached cluster hex map yet")
            return # When it is traveling to prob map

        # Prior
        prior = self.probability_map[hex_centre]
        # Posterior
        posterior = prior*(1-f) / (1-prior*f)
        posterior = 0

        self.probability_map[hex_centre] = posterior

        # Distribute
        sum_after_update = sum(self.probability_map.values())
        if sum_after_update != 0:
            probability_map = {
                key: value / sum_after_update for key, value in self.probability_map.items()}
            return 1, probability_map
        else:
            print("Entire probability map is zero")
            return 0, self.probability_map

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
