from abc import ABC, abstractmethod


class ClusterFinder(ABC):
    """DBScan Clustering Object"""

    def __init__(self, dataset):
        self.dataset = dataset  # List of points to do clustering
        self.cluster_count = 0  # Total number of clusters
        # Dictionary with keys(cluster_id), values (list of Points)
        self.clusters = {}

    @abstractmethod
    def fit(self):
        """
        Run the DBSCAN clustering algorithm.
        """
        return self.clusters
