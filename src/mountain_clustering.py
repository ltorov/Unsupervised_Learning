import numpy as np
from plotting import scatter_to_surface

class MountainClustering:
    def __init__(self, sigma: float = 0.5, beta: float = 0.625, max_iterations: int = 100, centers: np.array = None):
        """
        Initialize the MountainClustering class.

        Args:
            sigma (float): Sigma parameter for the mountain function.
            beta (float): Beta parameter for the mountain function.
            max_iterations (int): Maximum number of iterations for clustering.
            centers (list): List to store cluster centers.
        """
        self.sigma = sigma
        self.beta = beta
        self.max_iterations = max_iterations
        self.centers = centers if centers else []

    @staticmethod
    def mountain_function(data: np.array, G: np.array, sigma: float) -> np.array:
        """
        Compute the mountain function for data points and grid points.

        Args:
            data (np.array): Data points.
            G (np.array): Grid points.
            sigma (float): Sigma parameter for the mountain function.

        Returns:
            np.array: Array of mountain values for each grid point.
        """
        mv = []
        for i in range(len(G)):
            m = 0
            v = G[i]
            for xi in data:
                m += np.exp(-(np.linalg.norm(v - xi) ** 2) / (2 * sigma ** 2))
            mv.append(m)
        return np.array(mv)

    @staticmethod
    def new_mountain_function(mv: np.array, G: np.array, center: int, beta: float) -> np.array:
        """
        Compute the new mountain function based on an existing mountain function.

        Args:
            mv (np.array): Existing mountain function.
            G (np.array): Grid points.
            center (int): Index of the current center.
            beta (float): Beta parameter for the mountain function.

        Returns:
            np.array: Array of new mountain values.
        """
        mci = []
        for i in range(len(G)):
            v = G[i]
            mci.append(mv[i] - mv[center] * np.exp(-(np.linalg.norm(v - G[center]) ** 2) / (2 * beta ** 2)))
        return np.array(mci)

    def select_center(self, data: np.array, G: np.array) -> (int, np.array):
        """
        Select the initial center for clustering.

        Args:
            data (np.array): Data points.
            G (np.array): Grid points.

        Returns:
            int: Index of the selected center.
            np.array: Mountain values for all grid points.
        """
        mv = self.mountain_function(data, G, sigma=self.sigma)
        max_mountain_function = [np.max(mv), np.argmax(mv)]
        return max_mountain_function[1], mv

    def select_new_center(self, mv: np.array, G: np.array, center: int) -> (int, np.array):
        """
        Select a new center based on the existing center and mountain values.

        Args:
            mv (np.array): Existing mountain function.
            G (np.array): Grid points.
            center (int): Index of the current center.

        Returns:
            int: Index of the new center.
            np.array: New mountain values.
        """
        m_new = self.new_mountain_function(mv, G, center, beta=self.beta)
        new_center = np.argmax(m_new)
        return new_center, m_new

    @staticmethod
    def assign_to_centers(data: np.array, G: np.array, centers: np.array, sigma: float) -> np.array:
        """
        Assign data points to cluster centers based on mountain values.

        Args:
            data (np.array): Data points.
            G (np.array): Grid points.
            centers (np.array): Cluster centers.
            sigma (float): Sigma parameter for the mountain function.

        Returns:
            dict: Dictionary with cluster assignments.
        """
        clusters = {}
        for i, d in enumerate(data):
            mv = MountainClustering.mountain_function(np.array([d]), G[centers], sigma=sigma)

            if centers[np.argmax(mv)] not in clusters.keys():
                clusters[centers[np.argmax(mv)]] = []
            clusters[centers[np.argmax(mv)]].append(i)

        return clusters
    

    def cluster(self, data: np.array, G: np.array, show: bool = True) -> dict:
        """
        Perform mountain clustering on the data points.

        Args:
            data (np.array): Data points.
            G (np.array): Grid points.

        Returns:
            dict: Dictionary with cluster assignments.
        """
        center, mv = self.select_center(data, G)
        self.centers.append(center)
        if show:
            scatter_to_surface(G[:, 0], G[:, 1], mv, title = "Mountain Function")
        iteration_count = 0

        while True and iteration_count < self.max_iterations:
            center, mv = self.select_new_center(mv, G, center)
            if self.centers[-1] == center:
                break
            else:
                self.centers.append(center)
                if show:
                    scatter_to_surface(G[:, 0], G[:, 1], mv, title = "Mountain Function revised without center "+str(center))
            iteration_count += 1

        clusters = self.assign_to_centers(data, G, self.centers, sigma=self.sigma)

    
        new_clusters = {}

        for cluster_center, points in clusters.items():
            cluster_name = f'cluster {cluster_center}'
            center = G[cluster_center]
            new_clusters[cluster_name] = {'center': np.array(center), 'points': np.array(points)}

        return new_clusters