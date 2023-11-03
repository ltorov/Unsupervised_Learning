import numpy as np

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
        self.centers = centers if centers else []
        self.max_iterations = max_iterations
        self.epsilon = epsilon

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
                J += U[i][k] * np.linalg.norm(xk - data[int(ci)], ord=1)

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
                    if not (np.linalg.norm(xj - data[int(ci)]) <= np.linalg.norm(xj - data[int(ck)])):
                        uij = 0
                U[i, j] = uij
        return U

    def cluster(self, data: np.array) -> dict:
        """
        Cluster the data using the K-Means algorithm.

        Args:
            data (np.array): Data points to cluster.

        Returns:
            clusters (dict): Dictionary of clusters, where keys are cluster indices, and values are lists of data indices.
        """
        clusters = {}
        dissimilarity_measures = []
        iter = 0
        if self.centers == []:
            self.centers = np.random.choice(np.linspace(0, len(data) - 1, len(data), replace=False))

        J, _ = self.dissimilarity_measure(data)
        dissimilarity_measures.append(J)
        while True and iter < self.max_iterations:

            J, U = self.dissimilarity_measure(data)
            if (dissimilarity_measures[-1] - J) < self.epsilon:
                break
            else:
                dissimilarity_measures.append(J)
                iter += 1

        for i, row in enumerate(U):
            indices = np.where(row == 1)[0]
            if indices.size > 0:
                clusters[i] = indices.tolist()

        return clusters