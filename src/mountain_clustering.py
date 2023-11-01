import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata



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
    
    def scatter_to_surface(x, y, z, title = "Surface plot", xlabel = 'X Axis',  ylabel = 'Y Axis', zlabel = 'Z Axis'):
        # Create a 3D figure
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Define the grid for interpolation
        xi, yi = np.meshgrid(np.linspace(min(x), max(x), 100), np.linspace(min(y), max(y), 100))

        # Interpolate Z-values to create a surface
        zi = griddata((x, y), z, (xi, yi), method='cubic')

        # Create the surface plot
        ax.plot_surface(xi, yi, zi, cmap='viridis')

        # Set labels for the axes
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)

        # Set title
        plt.title(title)

        # Show the plot
        plt.show()

    def cluster(self, data: np.array, G: np.array) -> dict:
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
        # self.scatter_to_surface(G[:, 0], G[:, 1], mv)
        iteration_count = 0

        while True and iteration_count < self.max_iterations:
            center, mv = self.select_new_center(mv, G, center)
            if self.centers[-1] == center:
                break
            else:
                self.centers.append(center)
                # self.scatter_to_surface(G[:, 0], G[:, 1], mv)
            iteration_count += 1

        clusters = self.assign_to_centers(data, G, self.centers, sigma=self.sigma)
        return clusters