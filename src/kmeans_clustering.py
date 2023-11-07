import numpy as np
import random
from distances import distance, sort_distances

class KMeansClustering:
    def __init__(self, k: int, centers: np.array = None, max_iterations: int = 100, epsilon: float = 0.1):
        """
        Initialize the KMeansClustering class.

        Args:
            k (int): Number of clusters to create.
            centers (np.array): Initial cluster centers. Default is None.
            max_iterations (int): Maximum number of iterations for clustering. Default is 100.
            epsilon (float): Convergence threshold for stopping the iterations. Default is 0.1.
        """
        self.k = k
        self.centers = centers if centers != None else []
        self.max_iterations = max_iterations
        self.epsilon = epsilon
        self.clusters = {}

    def dissimilarity_measure(self, data: np.array) -> (float, np.array):
        """
        Calculate the dissimilarity measure of the clustering.

        Args:
            data (np.array): Data points to cluster.

        Returns:
            J (float): Dissimilarity measure of the clustering.
            U (np.array): Membership matrix.
        """
        J = 0
        U = self.membership_matrix(data)
        for i, ci in enumerate(self.centers):
            for k, xk in enumerate(data):
                J += U[i][k] * np.linalg.norm(xk - ci, ord=1)

        return J, U

    def membership_matrix(self, data) -> np.array:
        """
        Calculate the membership matrix for the data.

        Args:
            data (np.array): Data points to cluster.

        Returns:
            U (np.array): Membership matrix.
        """
        U = np.zeros((self.k, len(data)))

        for i, ci in enumerate(self.centers):
            for j, xj in enumerate(data):
                if i == j:
                    continue
                uij = 1
                for ck in self.centers:
                    if not (np.linalg.norm(xj - ci) <= np.linalg.norm(xj - ck)):
                        uij = 0
                U[i, j] = uij
        return U
    
    def update_centers(self, data) -> np.array:
        new_centers = np.zeros((len(self.clusters.keys()), data.shape[1]))
        for i, key in enumerate(self.clusters.keys()):
            new_centers[i,:] = 1/len(self.clusters[key])* np.sum(data[self.clusters[key]], axis =0)
        self.centers = new_centers

    def update_clusters(self, U):
        clusters = {}
        for i, row in enumerate(U):
                indices = np.where(row == 1)[0]
                if indices.size > 0:
                    clusters[i] = indices.tolist()
        self.clusters = clusters
        


    def cluster(self, data: np.array) -> dict:
        """
        Cluster the data using the K-Means algorithm.

        Args:
            data (np.array): Data points to cluster.

        Returns:
            clusters (dict): Dictionary of clusters, where keys are cluster indices, and values are lists of data indices.
        """
        dissimilarity_measures = []
        iter = 0
        if self.centers == []:
            centers_indexes = []
            centers_indexes.append(np.random.choice(np.arange(len(data)), size = 1)[0])
            for k_i in range(self.k-1):
                distances = distance(data, data[centers_indexes], metric = "euclidean")
                nearest_distances = np.min(distances, axis=0)**2
                selected_index = random.choices(range(len(nearest_distances)), weights=nearest_distances)[0]
                centers_indexes.append(selected_index)
            self.centers = data[centers_indexes]

        J, U = self.dissimilarity_measure(data)
        dissimilarity_measures.append(J)
        self.update_clusters(U)
        self.update_centers(data)
        while True and iter < self.max_iterations:

            J, U = self.dissimilarity_measure(data)
            if (dissimilarity_measures[-1] - J) < self.epsilon:
                break
            else:
                dissimilarity_measures.append(J)
                self.update_clusters(U)
                self.update_centers(data)
                iter += 1

        new_clusters = {}

        for cluster_center, points in self.clusters.items():
            cluster_name = f'cluster {cluster_center}'
            center = data[cluster_center].tolist()
            new_clusters[cluster_name] = {'center': np.array(center), 'points': np.array(points)}

        return new_clusters