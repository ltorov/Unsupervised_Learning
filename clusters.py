import numpy as np
import itertools
from typing import Dict, List

from sklearn.metrics import silhouette_score,  davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score, normalized_mutual_info_score,fowlkes_mallows_score

def intra_cluster_indices(data, clusters, show = True):
    labels = np.zeros(len(data))

    for idx, cluster in enumerate(clusters.values()):
        labels[cluster['points']] = idx
        
    silhoutte = silhouette_score(data, labels)
    davies_bouldin = davies_bouldin_score(data, labels)
    calinski_harabasz = calinski_harabasz_score(data, labels)
    
    if show:
        print("Silhoutte Score : ", silhoutte)
        print("Davies Bouldin Score : ", davies_bouldin)
        print("Calinski Harabasz Score : ", calinski_harabasz)

    return [silhoutte, davies_bouldin, calinski_harabasz]

def extra_cluster_indices(data, clusters, target_clusters, show = True):
    labels = np.zeros(len(data))

    for idx, cluster in enumerate(clusters.values()):
        labels[cluster['points']] = idx

    target_labels = np.zeros(len(data))

    for idx, cluster in enumerate(target_clusters.values()):
        target_labels[cluster['points']] = idx

    ari = adjusted_rand_score(target_labels, labels)
    nmi = normalized_mutual_info_score(target_labels, labels)
    fmi = fowlkes_mallows_score(target_labels, labels)

    if show:
        print("Adjusted Rand Index : ", ari)
        print("Normalized Mutual Information : ", nmi)
        print("Fowlkes-Mallows Index : ", fmi)
    return [ari, nmi, fmi]

def grid(dim: int, size: float, a: float = 0, b: float = 1) -> np.ndarray:
    """
    Divide a data points into grids based on the distance to other data points.

    Args:
        dim (int): The dimension of the data.
        size (float): The size of each grid cell.
        a (float, optional): The minimum value of the data (default is 0).
        b (float, optional): The maximum value of the data (default is 1).

    Returns:
        numpy.ndarray: A grid of points.
    """
    
    n = int((b-a)/(size) + 1)
    return np.array(list(itertools.product(np.linspace(a,b,n), repeat = dim)))

def similarity_index(clusters: dict, target_clusters: dict) -> dict:
    similarities = {}
    for key, target_cluster in target_clusters.items():
        max_similarity = max((len(np.intersect1d(cluster, target_cluster))/len(target_cluster), key) for key, cluster in clusters.items())
        similarities[key] = max_similarity
    return similarities

def boxes(D: np.ndarray, div: int = 4) -> Dict[str, List[List[float]]]:
    """
    Divide a data points into boxes based on the distance to other data points.

    Parameters:
        D (numpy.ndarray): The input 2D array.
        div (int, optional): The number of divisions (default is 4).

    Returns:
        dict: A dictionary containing lists of data points in each box.
              Keys are in the format 'Box <index>', and values are lists of data points [i, j, D[i, j]].
    """
    n, m = D.shape
    min_val = np.min(D); max_val = np.max(D)
    boxes_limits = np.linspace(min_val, max_val, div + 1)
    boxes_limits[-1] += 0.01
    boxes = {f'Box {i}': [] for i in range(div)}
    for i in range(n):
        for j in range(m):
            for b in range(div):
                if boxes_limits[b] <= D[i, j] < boxes_limits[b + 1]:
                    boxes[f'Box {b}'].append([i, j, D[i, j]])
    return boxes

def get_strs(str1):
  index = str1.find(" ") + 2
  return str1[:index]
  
def box_clusters(b, D):
    dists = {}
    for key, values in b.items():
        for point in values:
            if key + ' Point '+ str(point[0]) not in dists.keys():
                dists[key + ' Point '+ str(point[0])] = []
            dists[key + ' Point '+ str(point[0])].append(point[2])
    average_dists = {}
    for key, values in dists.items():
        if key not in average_dists.keys():
            average_dists[key] = len(dists[key])
    
    n, m = D.shape
    B = []
    for i in range(m):
        point = 'Point '+str(i)
        max = 0
        box = ''
        for key, value in average_dists.items():
            if point in key:
                if value > max:
                    max = value
                    box = get_strs(key)
        
        B.append([box, max])


    # Extract unique box names
    box_names = sorted(set(entry[0] for entry in B))

    # Create a dictionary to map box names to row indices in the matrix
    box_indices = {box: idx for idx, box in enumerate(box_names)}

    # Create the matrix with zeros
    matrix = np.zeros((len(box_names), len(B)))

    # Fill the matrix with 1s where the box is present at that index
    for i, (box, _) in enumerate(B):
        matrix[box_indices[box]][i] = 1

    clusters = {}
    for i, cluster in enumerate(matrix):
        points = []
        for j in range(len(cluster)):
            if j == 1:
                points.append(j)
        clusters['cluster '+ str(i)] = {'points': np.array(points)}
    return clusters

def neighbors(D: np.ndarray, c: int, epsilon: float) -> np.ndarray:
    """
    Perform DBSCAN clustering on the given distance matrix D.

    Args:
        D (numpy.ndarray): The input distance matrix.
        c (int): The minimum number of points required to form a cluster.
        epsilon (float): The maximum distance between two points for them to be considered neighbors.

    Returns:
        numpy.ndarray: A cluster matrix, where each row represents a point and each column represents a cluster.
    """
    n = len(D)
    labels = [-1] * n  # Initialize labels as unassigned (-1)
    cluster_id = 0

    for i in range(n):
        if labels[i] != -1:
            continue  # Skip already assigned points
        
        neighbors = [j for j in range(n) if D[i][j] <= epsilon]
        
        if len(neighbors) < c:
            labels[i] = 0  # Label point as noise
        else:
            cluster_id += 1  # Start a new cluster
            labels[i] = cluster_id  # Assign cluster ID to the current point
            expand_cluster(D, labels, i, cluster_id, epsilon, c)

    # Merge extra clusters if needed
    merge_clusters(D, labels, c, epsilon)

    n = len(labels)
    cluster_matrix = np.zeros((len(np.unique(labels)), n))
    for i in range(n):
        cluster_matrix[labels[i]][i] = 1

    clusters = {}
    for i, cluster in enumerate(cluster_matrix):
        points = []
        for j in range(len(cluster)):
            if j == 1:
                points.append(j)
        clusters['cluster '+ str(i)] = {'points': np.array(points)}
    return clusters


def expand_cluster(D: np.ndarray, labels: np.ndarray, i: int, cluster_id: int, epsilon: float, c: int) -> None:
    """
    Expand a cluster by assigning neighboring points to the cluster.

    Args:
        D (numpy.ndarray): The input distance matrix.
        labels (numpy.ndarray): A cluster matrix, where each row represents a point and each column represents a cluster.
        i (int): The index of the current point.
        cluster_id (int): The ID of the current cluster.
        epsilon (float): The maximum distance between two points for them to be considered neighbors.
        c (int): The minimum number of points required to form a cluster.

    Returns:
        None
    """
    n = len(D)
    stack = [i]

    while stack:
        current_point = stack.pop()
        neighbors = [j for j in range(n) if D[current_point][j] <= epsilon]

        if len(neighbors) >= c:
            for neighbor in neighbors:
                if labels[neighbor] == -1:
                    labels[neighbor] = cluster_id
                    stack.append(neighbor)

def merge_clusters(D: np.ndarray, labels: np.ndarray, max_clusters: int, epsilon: float) -> None:
    """
    Merge clusters if the number of clusters is greater than the specified maximum number.

    Args:
        D (numpy.ndarray): The input distance matrix.
        labels (numpy.ndarray): A cluster matrix, where each row represents a point and each column represents a cluster.
        max_clusters (int): The maximum number of clusters.
        epsilon (float): The maximum distance between two points for them to be considered neighbors.

    Returns:
        None
    """
    n = len(D)
    clusters = np.unique(labels).tolist()  # Convert to a Python list
    while len(clusters) > max_clusters:
        cluster_sizes = [np.sum(np.array(labels) == cluster) for cluster in clusters]
        smallest_cluster_idx = np.argmin(cluster_sizes)
        smallest_cluster = clusters[smallest_cluster_idx]
        clusters.pop(smallest_cluster_idx)  # Remove the smallest cluster ID

        # Calculate the Jaccard index between the smallest cluster and all other clusters
        jaccard_indices = []
        for other_cluster in clusters:
            intersection = np.sum(np.array(labels) == smallest_cluster) & np.sum(np.array(labels) == other_cluster)
            union = np.sum(np.array(labels) == smallest_cluster) | np.sum(np.array(labels) == other_cluster)
            jaccard_index = intersection / union
            jaccard_indices.append(jaccard_index)

        # Merge the smallest cluster with the cluster that has the highest Jaccard index
        best_cluster_idx = np.argmax(jaccard_indices)
        best_cluster = clusters[best_cluster_idx]
        for i in range(n):
            if labels[i] == smallest_cluster:
                labels[i] = best_cluster