import numpy as np
from plotting import scatter_to_surface

class SubtractiveClustering:
    def __init__(self, alpha_radius: float = 0.5, beta_radius: float = None, max_iterations: int = 100, centers: np.array = None, metric = 2):
        """
        Initialize the SubtractiveClustering class.

        Args:
            alpha_radius (float): alpha_radius parameter for the density measure.
            beta_radius (float): beta_radius parameter for the density measure.
            max_iterations (int): Maximum number of iterations for clustering.
            centers (list): List to store cluster centers.
        """
        self.alpha_radius = alpha_radius
        if beta_radius != None: self.beta_radius = beta_radius
        else: self.beta_radius = self.alpha_radius * 1.5
        self.max_iterations = max_iterations
        self.centers = centers if centers else []
        self.metric = metric

    def density_measure(self, data: np.array) -> np.array:
        """
        Compute the density measure for data points and grid points.

        Args:
            data (np.array): Data points.

        Returns:
            np.array: Array of density values for each grid point.
        """
        D = []
        for xi in data:
            Di = 0
            for xj in data:
                Di += np.exp(-(np.linalg.norm(xi - xj, ord = self.metric) ** 2) / ((self.alpha_radius/2) ** 2))
            D.append(Di)
        return np.array(D)

    def new_density_measure(self, data: np.array, D: np.array, center: int) -> np.array:
        """
        Compute the new density measure based on an existing mountain function.

        Args:
            data (np.array): Data points.
            D (np.array): Existing density measures.
            center (int): Index of the current center.

        Returns:
            np.array: Array of revised density measures.
        """
        Dci = []
        for i, xi in enumerate(data):
            Dci.append(D[i] - D[center] * np.exp(-(np.linalg.norm(xi - data[center], ord = self.metric) ** 2) / ((self.beta_radius/2) ** 2)))
        return np.array(Dci)

    def select_center(self, data: np.array) -> (int, np.array):
        """
        Select the initial center for clustering.

        Args:
            data (np.array): Data points.

        Returns:
            int: Index of the selected center.
            np.array: Density values for all data points.
        """
        D = self.density_measure(data)
        max_density_measure = [np.max(D), np.argmax(D)]
        return max_density_measure[1], D

    def select_new_center(self,  data: np.array, D: np.array, center: int) -> (int, np.array):
        """
        Select a new center based on the existing center and density measures.

        Args:
            D (np.array): Existing density measures.
            center (int): Index of the current center.

        Returns:
            int: Index of the new center.
            np.array: New density measures.
        """
        D_new = self.new_density_measure(data, D, center)
        return np.argmax(D_new), D_new

    def assign_to_centers(self, data: np.array) -> np.array:
        """
        Assign data points to cluster centers based on mountain values.

        Args:
            data (np.array): Data points.

        Returns:
            dict: Dictionary with cluster assignments.
        """
    
        clusters = {}
        for i, xi in enumerate(data):
            D = []
            
            for cj in self.centers:
                Di = np.exp(-(np.linalg.norm(xi - data[cj], ord = self.metric) ** 2) / ((self.alpha_radius/2) ** 2))
                D.append(Di)

            if self.centers[np.argmax(D)] not in clusters.keys():
                clusters[self.centers[np.argmax(D)]] = []
            clusters[self.centers[np.argmax(D)]].append(i)

        return clusters
    
    def stop_criteria(self, D_center: np.array, D_center_new: np.array, epsilon: float = 0.2, criteria = 'center_densities') -> bool:
        """
        Decide when clustering algorithm should stop.

        Args:
            D_center (np.array): Data points.
            D_center_new (np.array): Grid points.
            epsilon (float): alpha_radius parameter for the mountain function.

        Returns:
            dict: Dictionary with cluster assignments.
        """
        if criteria == 'center_densities':
            return (D_center_new < epsilon * D_center)
        
        return False

    def cluster(self, data: np.array, show: bool = True, colors = "YlGnBu") -> dict:
        """
        Perform mountain clustering on the data points.

        Args:
            data (np.array): Data points.
            G (np.array): Grid points.

        Returns:
            dict: Dictionary with cluster assignments.
        """
        center, D = self.select_center(data)
        self.centers.append(center)
        if show:
            scatter_to_surface(data[:, 0], data[:, 1], D, title = "Subtractive Density Measure", colors = colors)
        iteration_count = 0

        while True and iteration_count < self.max_iterations:
            center, D_new = self.select_new_center(data, D, center)

            if self.centers[-1] == center:
                break
            elif self.stop_criteria(D[self.centers[-1]], D_new[center]):
                break
            else:
                self.centers.append(center)
                D = D_new
                if show:
                    scatter_to_surface(data[:, 0], data[:, 1], D, title = "Subtractive Density Measure revised without center "+str(center), colors = colors)
            iteration_count += 1
        clusters = self.assign_to_centers(data)
        
        new_clusters = {}

        for cluster_center, points in clusters.items():
            cluster_name = f'cluster {cluster_center}'
            center = data[cluster_center].tolist()
            new_clusters[cluster_name] = {'center': np.array(center), 'points': np.array(points)}

        return new_clusters