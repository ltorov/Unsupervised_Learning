from kmeans_clustering import KMeansClustering
import numpy as np
from distances import distance, sort_distances
class SpectralClustering:
    def __init__(self, k: int,  max_iterations: int = 100, epsilon: float = 0.1, sigma: float = 0.9):
        """
        Initialize the FuzzyCMeansClustering class.

        Args:
            k (int): Number of clusters to create.
            max_iterations (int): Maximum number of iterations for clustering. Default is 100.
            epsilon (float): Convergence threshold for stopping the iterations. Default is 0.1.
        """
        self.k = k
        self.max_iterations = max_iterations
        self.epsilon = epsilon
        self.sigma = sigma
        self.similarity = []
        self.centers = []


    def cluster(self, data: np.array) -> dict:
        """
        Cluster the data using the K-Means algorithm.

        Args:
            data (np.array): Data points to cluster.

        Returns:
            clusters (dict): Dictionary of clusters, where keys are cluster indices, and values are lists of data indices.
        """
        clusters = {}
        distances = distance(data, metric = "euclidean")
        self.similarity = np.exp(-distances**2/ (2 * self.sigma**2))

        diagonal = np.diag(np.sum(self.similarity, axis=0))
        laplacian = diagonal - self.similarity
        
        eigenvalues, eigenvectors = np.linalg.eig(laplacian)
        min_eigenvalues = np.argsort(eigenvalues)
        
        self.centers = min_eigenvalues[:self.k]
        print(self.centers)

        kmeans = KMeansClustering(k = self.k)
        clusters = kmeans.cluster(eigenvectors[:, min_eigenvalues[:self.k]])
        return clusters