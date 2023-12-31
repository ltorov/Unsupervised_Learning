import numpy as np

class FuzzyCMeansClustering:
    def __init__(self, k: int, m: float = 2.1, max_iterations: int = 100, epsilon: float = 0.1, metric = 2):
        """
        Initialize the FuzzyCMeansClustering class.

        Args:
            k (int): Number of clusters to create.
            max_iterations (int): Maximum number of iterations for clustering. Default is 100.
            epsilon (float): Convergence threshold for stopping the iterations. Default is 0.1.
        """
        self.k = k
        self.m = m
        self.centers = []
        self.max_iterations = max_iterations
        self.epsilon = epsilon
        self.U = []
        self.metric = metric

    def dissimilarity_measure(self, data: np.array) -> float:
        """
        Calculate the dissimilarity measure of the clustering.

        Args:
            data (np.array): Data points to cluster.

        Returns:
            J (float): Dissimilarity measure of the clustering.
        """
        J = 0
        for i, ui in enumerate(self.U):
            for j, xj in enumerate(data):
                J += ui[j]**self.m * np.linalg.norm(xj- self.centers[i], ord = self.metric)
            
        return J

    def update_membership_matrix(self, data) -> None:
        """
        Update the membership matrix for the data.

        Args:
            data (np.array): Data points to cluster.
        """
        for i, ci in enumerate(self.centers):
            for j, xj in enumerate(data):
                uij = 0
                for ck in self.centers:
                    uij+= (np.linalg.norm(ci- xj, ord = self.metric)/ np.linalg.norm(ck- xj, ord = self.metric))**(2/(self.m-1))
                self.U[i, j] = 1/uij

    def find_centers(self, data: np.array) -> None:
        """
        Finds the centroid of each all clusters.

        Args:
            data (np.array): Data points to cluster.
        """
        self.centers = []
        for i, ui in enumerate(self.U):
            ci = np.zeros(data.shape[1])
            for j, xj in enumerate(data):
                ci += xj* (ui[j]**self.m)
            self.centers.append(ci/np.sum(ui**self.m))

    def cluster(self, data: np.array) -> dict:
        """
        Cluster the data using the K-Means algorithm.

        Args:
            data (np.array): Data points to cluster.

        Returns:
            new_clusters (dict): Dictionary of clusters, where keys are cluster names 'cluster _', and values are a dictionary with keys 
                'center', with the coordinates of the center and 'points', an array of data point indexes.
        """
        clusters = {}
        dissimilarity_measures = []
        iter = 0

        self.U = np.random.rand(self.k, len(data))
        self.U /= np.sum(self.U, axis=0)
        
        self.find_centers(data)
        J = self.dissimilarity_measure(data)
        dissimilarity_measures.append(J)
        self.update_membership_matrix(data)

        
        while True and iter < self.max_iterations:
            self.find_centers(data)
            J = self.dissimilarity_measure(data)

            if (dissimilarity_measures[-1] - J) < self.epsilon:
                break
            else:
                dissimilarity_measures.append(J)
                self.update_membership_matrix(data)
                iter += 1

            J = self.dissimilarity_measure(data)

        for i, xi in enumerate(data):
            if np.argmax(self.U[:,i]) not in clusters.keys():
                clusters[np.argmax(self.U[:,i])] = []
            clusters[np.argmax(self.U[:,i])].append(i)

        new_clusters = {}

        for cluster_center, points in clusters.items():
            cluster_name = f'cluster {cluster_center}'
            center = data[cluster_center].tolist()
            new_clusters[cluster_name] = {'center': np.array(center), 'points': np.array(points)}

        return new_clusters