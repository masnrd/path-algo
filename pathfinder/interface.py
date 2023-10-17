from abc import ABC, abstractmethod
import numpy as np
import h3


# Define an interface using an abstract base class
class PathFinder(ABC):

    def __init__(self, res: int, center: tuple):
        self.res = res
        self.center_hexagon = h3.geo_to_h3(center[0], center[1], resolution=self.res)

    @abstractmethod
    def find_next_step(self, current_position: tuple[int, int], prob_map: np.ndarray) -> tuple[int, int]:
        """
        Find the next step and waypoints to follow.

        Args:
            current_position (tuple[int, int]): Current coordinates (x, y) lat, long
            prob_map: np.ndarray - of size 7*..7 equal to size of dimension
        Returns:
            tuple[int, int] - Next step coordinates in long lat (x, y).
        """
        pass
