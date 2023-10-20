import numpy as np
import h3

from pathfinder.interface import PathFinder
from utils.hex import *


class OutwardSpiralPathFinder(PathFinder):
    def __init__(self, res: int, centre: tuple):
        super().__init__(res, centre)
        self.segment_start_ij_coord = None
        self.next_path_segment = []
        self.k_ring = 1

        center_ij_coord = h3.experimental_h3_to_local_ij(
            self.centre_hexagon, self.centre_hexagon)
        self.next_path_segment.append(center_ij_coord)
        self.segment_start_ij_coord = center_ij_coord

    # Create path over an entire circle
    def ring_edge_traversal(self, repetitions, current_ij_coord, i_increment, j_increment):
        for i in range(repetitions):
            current_ij_coord = (
                current_ij_coord[0]+i_increment, current_ij_coord[1]+j_increment)
            self.next_path_segment.append(current_ij_coord)
        return current_ij_coord

    # Implementation of abstract method that returns next waypoint
    def find_next_step(self, current_position: tuple[int, int], prob_map: dict) -> tuple[int, int]:
        current_position_ij = h3.experimental_h3_to_local_ij(self.centre_hexagon, h3.geo_to_h3(
            current_position[0], current_position[1], resolution=self.res))
        # Waypoints are calculated based on ring
        if len(self.next_path_segment) == 1 and self.segment_start_ij_coord == current_position_ij:
            self.segment_start_ij_coord = self.ring_edge_traversal(
                1, self.segment_start_ij_coord, 0, -1)
            self.segment_start_ij_coord = self.ring_edge_traversal(
                self.k_ring-1, self.segment_start_ij_coord, 1, 0)
            self.segment_start_ij_coord = self.ring_edge_traversal(
                self.k_ring, self.segment_start_ij_coord, 1, 1)
            self.segment_start_ij_coord = self.ring_edge_traversal(
                self.k_ring, self.segment_start_ij_coord, 0, 1)
            self.segment_start_ij_coord = self.ring_edge_traversal(
                self.k_ring, self.segment_start_ij_coord, -1, 0)
            self.segment_start_ij_coord = self.ring_edge_traversal(
                self.k_ring, self.segment_start_ij_coord, -1, -1)
            self.segment_start_ij_coord = self.ring_edge_traversal(
                self.k_ring, self.segment_start_ij_coord, 0, -1)
            self.k_ring += 1

        if current_position_ij == self.next_path_segment[0]:
            self.next_path_segment.pop(0)
            return h3.h3_to_geo(h3.experimental_local_ij_to_h3(self.centre_hexagon, self.next_path_segment[0][0], self.next_path_segment[0][1]))
        else:
            print("Previous waypoint may not be correct")
            return None


class BayesianHexSearch(PathFinder):
    """A pathfinding algorithm in the H3 hexagonal grid system using probability.
    """

    def __init__(self, res: int, center: tuple) -> None:
        """Initializes with given resolution and starting position

        Args:
            res (int): The H3 resolution for the hexagonal grid.
            center (tuple[float, float]): Starting position as a tuple of (latitude, longitude).
        """
        super().__init__(res, center)
        self.trajectory = []

    def find_next_step(self, current_position: tuple[float, float], prob_map: dict) -> tuple[int, int]:
        """Determines the next waypoint based on current position and a probability map.

        Args:
            current_position (tuple[float, float]): Current position as a tuple of (latitude, longitude).
            prob_map (dict): A numpy array of (7,7,7,7) representing the probability in each hexagon./ Dictionary of hexagons 

        Returns:
            tuple[int, int]: Next waypoint as a tuple of (latitude, longitude).
        """
        # Initialise current position
        curr_hexagon = h3.geo_to_h3(
            current_position[0], current_position[1], resolution=self.res)

        # Hex index of the highest probability
        max_hex_index = max(prob_map, key=lambda key: prob_map[key])

        # Get neighbours
        neighbours = h3.k_ring(curr_hexagon, 1)

        # Initialise variables to find the nest best neighbour
        best_neighbour = None
        highest_score = 0

        for neighbour in neighbours:
            if neighbour not in prob_map:
                continue
            dist = distance_between_2_hexas(neighbour, max_hex_index)
            neighbour_prob = prob_map[neighbour]
            # TODO: Test different parameters for this
            score = dist * 100 + neighbour_prob * 10
            # score = neighbour_prob * 10

            if score > highest_score:
                best_neighbour = neighbour
                highest_score = score

        return h3.h3_to_geo(best_neighbour)
