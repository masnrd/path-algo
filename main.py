import folium

from clusterfinder.clusterfinder import (
    DIANAClusterFinder
)
from pathfinder.algo import (
    OutwardSpiralPathFinder,
)
from experiment.test_framework import TestFramework

# Run and validate
RES=15
NUM_HOTSPOT = 50
NUM_CASUALTY = 50
STEPS = 200
CENTRE = (1.3392911509416838, 103.95958286190708)
MAP = folium.Map(location=CENTRE, zoom_start=16, tiles='cartodb positron', max_zoom=24)


if __name__ == "__main__":
    print("\nDIANA-Spiral")
    test_object = TestFramework(name="DIANA", res=RES)
    test_object.init_mission(MAP, CENTRE, NUM_HOTSPOT, NUM_CASUALTY)
    test_object.register_cluster_finder(DIANAClusterFinder, threshold=0.1)
    test_object.register_path_finder(OutwardSpiralPathFinder)
    test_object.run(STEPS, only_path=True, print_output=True)
