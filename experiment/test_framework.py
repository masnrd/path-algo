from typing import Optional

import numpy as np
import h3
import folium
import random
from datetime import datetime

from clusterfinder.point import Point
from clusterfinder.interface import ClusterFinder
from pathfinder.interface import PathFinder
from utils.hex import *
from utils.angle import *

N_RINGS_CLUSTER = 16  # 7.5m * 16 = 240m radius
MAIN_MAP_RADIUS = 0.5  # km


class TestFramework:
    def __init__(self, name: str, res: int):
        self.name = name

        # Main map
        self.res = res
        self.centre = None
        self.main_map = None
        self.bounds = None
        self.num_hotspot = None
        self.num_casualty = None
        self.hotspots = None            # list[tuple[lat, lng]]
        self.casualty_locations = None

        # Cluster Finder
        self.cluster_finder = None
        self.cluster_results = dict()

        # Path Finder Map
        self.path_finder = None
        self.path_finder_object = None

        self.all_centres = dict()
        self.all_probability_map = dict()
        self.all_casualty_locations = dict()
        self.all_search_outputs = dict()

        # Metrics
        self.evaluation_metrics = list()

    def init_mission(self, main_map: folium.Map, centre: tuple[float, float], num_hotspot: int, num_casualty: int):
        self.num_hotspot = num_hotspot
        self.num_casualty = num_casualty
        self.centre = centre
        self.main_map = main_map
        self.bounds = self.add_markers_get_bounds()
        self.hotspots = self.add_hotspots(num_hotspot)
        self.casualty_locations = self.add_casualty(num_casualty)
        self.casualty_locations = set(h3.geo_to_h3(lat, lng, self.res) for lat, lng in self.casualty_locations)

    def add_markers_get_bounds(self, radius_km=MAIN_MAP_RADIUS) -> list[list[float, float], list[float, float]]:
        """
        Get bounds of the Folium map by adding feature group with radius radius_km
        :param radius_km:
        :return: SW and NE bounding corners in lat, lng
        """
        from math import radians, degrees, cos, sin, asin, atan2

        def calculate_offset(lat, lon, d_km, bearing):
            R = 6371.0  # Radius of the Earth in km
            bearing = radians(bearing)  # Convert bearing to radians
            lat1 = radians(lat)  # Current lat point converted to radians
            lon1 = radians(lon)  # Current long point converted to radians

            lat2 = asin(sin(lat1) * cos(d_km / R) + cos(lat1) * sin(d_km / R) * cos(bearing))
            lon2 = lon1 + atan2(sin(bearing) * sin(d_km / R) * cos(lat1), cos(d_km / R) - sin(lat1) * sin(lat2))

            lat2 = degrees(lat2)
            lon2 = degrees(lon2)

            return lat2, lon2
        # Calculate the NSEW bounds
        offsets = [
            calculate_offset(self.centre[0], self.centre[1], radius_km, bearing)
            for bearing in [0, 90, 180, 270]
        ]

        # Create a feature group
        fg = folium.FeatureGroup(name='NSEW markers')
        for offset in offsets:
            folium.Marker(offset).add_to(fg)
        fg.add_to(self.main_map)
        return fg.get_bounds()

    def add_hotspots(self, num_hotspots):
        hotspots = []
        for i in range(num_hotspots):
            lat = random.uniform(self.bounds[0][0], self.bounds[1][0])
            lng = random.uniform(self.bounds[0][1], self.bounds[1][1])
            hotspots.append((lat, lng))
        return hotspots

    def add_casualty(
            self, num_casualty: int, casualty_distance_std_dev=0.00005
    ) -> list:
        casualty_locations = []
        num_hotspots = len(self.hotspots)

        # Calculate the number of casualties per hotspot
        casualties_per_hotspot = max(num_casualty // num_hotspots, 1)  # Ensure at least one casualty per hotspot
        casualties_remainder = num_casualty % num_hotspots

        for idx, hotspot in enumerate(self.hotspots):
            hotspot_lat, hotspot_lng = hotspot
            # Assign an extra casualty to some hotspots to account for remainder
            extra_casualty = 1 if idx < casualties_remainder else 0
            for _ in range(casualties_per_hotspot + extra_casualty):
                casualty_lat = np.random.normal(hotspot_lat, casualty_distance_std_dev)
                casualty_lng = np.random.normal(hotspot_lng, casualty_distance_std_dev)

                # Ensure that the latitude and longitude are valid
                casualty_lat = max(min(casualty_lat, 90), -90)
                casualty_lng = max(min(casualty_lng, 180), -180)

                casualty_locations.append((casualty_lat, casualty_lng))

        return casualty_locations

    def run(
            self, steps: int, update_map: bool = False, f: Optional[float] = None
    ) -> dict[int: list]:
        # Stage 1: Region Segmentation - Clustering
        self.cluster_results = self.cluster_finder.fit()
        self.cluster_finder.print_outputs()

        # TODO Stage 2: Region Allocation - Task Assignment

        # Stage 3: Search
        for cluster_id, cluster in self.cluster_results.items():
            print(f"\nCluster:", cluster_id)
            # Step 1: Find centre for probability map
            centre = self.find_search_centre(cluster)
            self.all_centres[cluster_id] = centre

            # Step 2: Initialize probability map based on all mini hotspot
            probability_map = self.initialize_probability_map(centre, N_RINGS_CLUSTER)
            self.all_probability_map[cluster_id] = probability_map

            # Step 3: Update probability map based on mini hotspots:
            for mini_hotspot in cluster:
                probability_map = self.add_mini_hotspot(probability_map, mini_hotspot)

            # Step 4: Search
            self.path_finder_object = self.path_finder(self.res, centre)
            output, casualty_detected, minimum_time_captured, accumulated_angle = self.search(
                probability_map, centre, self.casualty_locations, steps, update_map, f
            )
            self.all_search_outputs[cluster_id] = output

            # Step 5: Evaluate
            metrics = self.evaluate_search(
                probability_map, self.casualty_locations, casualty_detected, output, minimum_time_captured, accumulated_angle
            )
            self.evaluation_metrics.append(metrics)
            self.print_individual_metrics(metrics)

        self.print_evaluation_averages()

    @staticmethod
    def find_search_centre(cluster: list[Point]) -> tuple[float, float]:
        """
        Find the geographic center of a cluster of Points.

        :param cluster: A list of Point objects.
        :return: A Point object representing the geographic center of the cluster.
        """

        if not cluster:
            raise ValueError("The cluster is empty")

        # Convert all points to Cartesian coordinates
        x, y, z = 0.0, 0.0, 0.0

        for point in cluster:
            latitude = np.radians(point.coordinates[0])
            longitude = np.radians(point.coordinates[1])

            x += np.cos(latitude) * np.cos(longitude)
            y += np.cos(latitude) * np.sin(longitude)
            z += np.sin(latitude)

        # Compute average coordinates
        total_points = len(cluster)
        x /= total_points
        y /= total_points
        z /= total_points

        # Convert average coordinates back to latitude and longitude
        central_longitude = np.arctan2(y, x)
        central_square_root = np.sqrt(x * x + y * y)
        central_latitude = np.arctan2(z, central_square_root)

        # Convert radians back to degrees
        central_latitude = np.degrees(central_latitude)
        central_longitude = np.degrees(central_longitude)

        # Create a new Point object for the center
        centre_point = (central_latitude, central_longitude)

        return centre_point

    def initialize_probability_map(self, centre: tuple[float, float], n_rings: int) -> dict[str, float]:
        """
        Initialize the probability map.

        :param centre: centre of the probability map to be searched
        :param n_rings: Number of rings around the center hexagon.
        :return: Dictionary containing hex index as key and its probability as value.
        """
        probability_map = {}
        all_hex_idx = h3.k_ring(h3.geo_to_h3(
            centre[0], centre[1], self.res), n_rings)
        for hex_idx in all_hex_idx:
            probability_map[hex_idx] = 0
        return probability_map

    def search(
            self, probability_map: dict[str, float], waypoint: tuple[float, float], casualty_locations: set,
            steps: int, update_map: bool = False, f: Optional[float] = None
    ) -> tuple:
        """
        Simulate the drone path

        :param casualty_locations:
        :param probability_map:
        :param waypoint: current position
        :param f: probability of detecting a person
        :param steps: Number of steps to simulate.
        :param update_map: Boolean to decide if probability map should be updated at each step.
        :return: A list containing dictionary of hexagon index and step count.
        """
        if not self.path_finder_object:
            raise ValueError("Please Register your Pathfinder first")

        output = list()
        casualty_detected = dict()
        accumulated_angle = 0.0
        start_time, end_time = datetime.now(), None

        for i in range(steps):
            if update_map:
                probability_map = self.update_probability_map(probability_map, waypoint, f)

            waypoint = self.path_finder_object.find_next_step(waypoint, probability_map)
            hex_idx = h3.geo_to_h3(waypoint[0], waypoint[1], self.res)

            casualty_detected, end_time = self.handle_detection(
                hex_idx, casualty_locations, casualty_detected, end_time
            )
            output.append({"hex_idx": hex_idx, "step_count": i})
            accumulated_angle = self.handle_angle(output, accumulated_angle)

        minimum_time_captured = self.calculate_metrics(start_time, end_time)
        return output, casualty_detected, minimum_time_captured, accumulated_angle

    @staticmethod
    def handle_detection(hex_idx: str, casualty_locations: set, casualty_detected: dict, end_time) -> tuple:
        # Probability of discovering casualty
        if hex_idx in casualty_locations:
            coin = random.randint(1, 10)
            if coin == 1:
                casualty_detected[hex_idx] = False
            else:
                casualty_detected[hex_idx] = True
            if len(casualty_detected) == len(casualty_locations):
                end_time = datetime.now()
        return casualty_detected, end_time

    @staticmethod
    def handle_angle(output: list[dict], accumulated_angle: float) -> float:
        if len(output) >= 3:
            a = h3.h3_to_geo(output[-3]["hex_idx"])
            b = h3.h3_to_geo(output[-2]["hex_idx"])
            c = h3.h3_to_geo(output[-1]["hex_idx"])
            accumulated_angle += get_angle_3_pts(a, b, c)
        return accumulated_angle

    @staticmethod
    def calculate_metrics(start_time, end_time) -> int:
        minimum_time_captured = None
        if end_time:
            minimum_time_captured = end_time - start_time
            minimum_time_captured = round(minimum_time_captured.total_seconds(), 2)
        return minimum_time_captured

    def register_cluster_finder(self, cluster_finder: ClusterFinder, *args, **kwargs):
        # Convert hotspots to Point
        hotspots = [Point(i, self.hotspots[i]) for i in range(self.num_hotspot)]
        self.cluster_finder = cluster_finder(hotspots, *args, **kwargs)

    def register_path_finder(self, path_finder: PathFinder):
        """
        Register the path_finder.

        :param path_finder: An instance of PathFinder.
        """
        self.path_finder = path_finder

    def add_mini_hotspot(
            self, probability_map: dict[str, float], hotspot: Point,
            sigma: float = 0.03, r_range: int = 100
    ) -> dict[str, float]:
        """
        Update the probability map based on a given hotspot.

        Parameters:
        - prob_en: numpy array representing the probability map
        - hotspot: tuple containing latitude and longitude of the hotspot
        - sigma: standard deviation for the gaussian probability distribution
        - r_range: range for hex_ring (default is 100)

        Returns:
        - Updated prob_en
        """
        def gaussian_probability(dist, sig=0.01):
            return np.exp(-dist**2 / (2 * sig**2))

        hex_hotspot = h3.geo_to_h3(hotspot.coordinates[0], hotspot.coordinates[1], self.res)

        delta_probability_map = {}
        for i in range(0, r_range):
            hex_at_r = h3.hex_ring(hex_hotspot, i)
            distance = euclidean(h3.h3_to_geo(hex_hotspot),
                                 h3.h3_to_geo(next(iter(hex_at_r))))
            probability = gaussian_probability(distance, sigma)
            for hex_idx in hex_at_r:
                delta_probability_map[hex_idx] = probability

        # Distribute
        total_prob = sum(delta_probability_map.values())
        if total_prob != 0:
            delta_probability_map = {
                key: (value / total_prob) for key, value in delta_probability_map.items()
            }

        for hex_idx in probability_map:
            if hex_idx in delta_probability_map:
                probability_map[hex_idx] += delta_probability_map[hex_idx]

        # Distribute
        total_prob = sum(probability_map.values())
        if total_prob != 0:
            probability_map = {
                key: (value / total_prob) for key, value in probability_map.items()}
            return probability_map
        else:
            print("Entire probability map is zero")
            return dict()

    def update_probability_map(
            self, probability_map: dict[str, float], centre: tuple[float, float], f: float
    ) -> dict[str, float]:
        """Update the probability map using Bayes theorem.

        Args:
            :param f: Probability of finding a person
            :param centre:
            :param probability_map:
        """
        hex_centre = h3.geo_to_h3(
            centre[0], centre[1], self.res)

        # Prior
        prior = probability_map[hex_centre]

        # Posterior
        posterior = prior*(1-f) / (1-prior*f)
        probability_map[hex_centre] = posterior

        # Distribute
        total_prob = sum(probability_map.values())
        if total_prob != 0:
            probability_map = {
                key: value / total_prob for key, value in probability_map.items()}
            return probability_map
        else:
            print("Entire probability map is zero")
            return dict()

    def evaluate_search(
            self, probability_map: dict[str, float],
            casualty_locations: set, casualty_detected: dict,
            output: list, minimum_time_captured: int, accumulated_angle: float
    ):
        """
        Evaluate the performance of the path_finder.
        """
        return {
            'path_coverage': self.check_path_coverage(probability_map, output),
            'angle_curvature': self.calculate_average_angle_curvature(output, accumulated_angle),
            'casualties_captured': len(casualty_detected),
            'casualties_count': len(casualty_locations),
            'minimum_time_captured': minimum_time_captured if minimum_time_captured else None,
            'false_negatives': self.check_guaranteed_capture(casualty_locations, casualty_detected)[1]
        }

    @staticmethod
    def calculate_average_angle_curvature(output, accumulated_angle):
        if len(output) >= 3:
            return round(accumulated_angle / (len(output) - 2), 2)
        else:
            return None

    def print_individual_metrics(self, metrics: dict):
        """
        Prints the individual metrics for a single evaluation.
        """
        print(f"{self.name}'s Path Coverage: {metrics['path_coverage']}%")
        angle_curvature = metrics['angle_curvature']
        if angle_curvature is not None:
            print(f"{self.name}'s Average Angle Curvature: {angle_curvature} degrees")
        else:
            print(f"{self.name}'s Average Angle Curvature: NA")

        print(f"{self.name}'s Casualties Captured: {metrics['casualties_captured']}")
        # if metrics['false_negatives']:
        #     print(f"{self.name}'s False Negatives: {metrics['false_negatives']}")

        # minimum_time = metrics['minimum_time_captured']
        # if minimum_time is not None:
        #     print(f"{self.name}'s Minimum Time Capture: {minimum_time} seconds")
        # else:
        #     print(f"{self.name}'s Minimum Time Capture: NA")

    def print_evaluation_averages(self):
        # Initialize sums for each metric
        sums = {
            'path_coverage': 0,
            'angle_curvature': 0,
            'casualties_captured': 0,
            'casualties_count': 0,
            'minimum_time_captured': 0,
            'false_negatives': 0,
        }
        counts = {key: 0 for key in sums}

        # Sum values from each evaluation
        for metrics in self.evaluation_metrics:
            for key, value in metrics.items():
                if value is not None:
                    sums[key] += value
                    counts[key] += 1

        # Calculate and print averages
        print("\nAverage Evaluation Metrics:")
        for key, total in sums.items():
            if counts[key] > 0:
                average = round(total / counts[key], 2)
                if key == 'casualties_captured':
                    # These are counts, not averages, so handle them differently
                    print(f"Total {key.replace('_', ' ').title()}: {total}/{self.num_casualty}")
                elif key in {'path_coverage', 'angle_curvature'}:
                    print(f"Average {key.replace('_', ' ').title()}: {average}")
                else:
                    pass
            else:
                print(f"Average {key.replace('_', ' ').title()}: NA")

    @staticmethod
    def check_path_coverage(probability_map: dict[str, float], output) -> float:
        """
        Check the path coverage percentage.

        :return: Coverage percentage.
        """
        all_cells = {
            key for key, value in probability_map.items()}
        covered_cells = {item["hex_idx"] for item in output}
        path_covered = all_cells & covered_cells
        path_coverage = round(len(path_covered) /
                              len(all_cells) * 100, 2)
        return path_coverage

    @staticmethod
    def check_guaranteed_capture(casualty_locations, casualty_detected) -> tuple[bool, int]:
        """
        Check if the path_finder guaranteed the capture of all casualties.

        :return: Tuple containing boolean value for guaranteed capture and the number of false positives.
        """
        false_negative = 0
        for key, value in casualty_detected.items():
            if not value:
                false_negative += 1
        if false_negative == 0 and len(casualty_detected) == len(casualty_locations):
            guaranteed_capture = True
        else:
            guaranteed_capture = False

        return guaranteed_capture, false_negative
