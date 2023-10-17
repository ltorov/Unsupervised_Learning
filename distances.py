import numpy as np
import itertools
from clusters import boxes

def euclidean(data: np.ndarray) -> np.ndarray:
    """
    Calculates the Euclidean distance matrix between all points in the data set.

    Args:
        data (numpy.ndarray): A 2D array of data points.

    Returns:
        numpy.ndarray: A 2D array of Euclidean distances between all points.
    """
    n = len(data)
    D = np.zeros([n, n])
    for i in range(n):
        for j in range(i,n):
            if i==j:
                continue
            d = np.sqrt(np.sum((data[i] - data[j])**2))
            D[i,j] = d; D[j,i] = d
    return D

def euclidean_distances(data: np.ndarray, vertices: np.ndarray = None) -> np.ndarray:
    """
    Calculates the Euclidean distance matrix between all points in the data set and the given vertices.

    Args:
        data (numpy.ndarray): A 2D array of data points.
        vertices (numpy.ndarray): A 2D array of vertices.

    Returns:
        numpy.ndarray: A 2D array of Euclidean distances between all points and the given vertices.
    """
    if not isinstance(vertices, np.ndarray):
        return euclidean(data)
    
    n = len(vertices); m = len(data)
    D = np.zeros([n, m])
    for i in range(n):
        for j in range(m):
            D[i,j] = np.sqrt(np.sum((vertices[i] - data[j])**2))
    return D

def manhattan(data: np.ndarray) -> np.ndarray:
    """
    Calculates the Manhattan distance matrix between all points in the data set.

    Args:
        data (numpy.ndarray): A 2D array of data points.

    Returns:
        numpy.ndarray: A 2D array of Manhattan distances between all points.
    """
    n = len(data)
    D = np.zeros([n, n])
    for i in range(n):
        for j in range(i,n):
            if i==j:
                continue
            d = np.sum(np.abs(data[i]-data[j]))
            D[i,j] = d; D[j,i] = d
    return D

def manhattan_distances(data: np.ndarray, vertices: np.ndarray = None) -> np.ndarray:
    """
    Calculates the Manhattan distance matrix between all points in the data set and the given vertices.

    Args:
        data (numpy.ndarray): A 2D array of data points.
        vertices (numpy.ndarray): A 2D array of vertices.

    Returns:
        numpy.ndarray: A 2D array of Manhattan distances between all points and the given vertices.
    """
    if not isinstance(vertices, np.ndarray):
        return manhattan(data)
    
    n = len(vertices); m = len(data)
    D = np.zeros([n, m])
    for i in range(n):
        for j in range(m):
            D[i,j] = np.sum(np.abs(vertices[i]-data[j]))
    return D

def lp(data: np.ndarray, p=1) -> np.ndarray:
    """
    Calculates the Lp distance matrix between all points in the data set, where p is a positive integer.

    Args:
        data (numpy.ndarray): A 2D array of data points.
        p (int): The Lp norm.

    Returns:
        numpy.ndarray: A 2D array of Lp distances between all points.
    """
    n = len(data)
    D = np.zeros([n, n])
    if p != "inf":
        for i in range(n):
            for j in range(i,n):
                if i==j:
                    continue
                d = (np.sum(np.abs(data[i]-data[j])**p))**(1/p)
                D[i,j] = d; D[j,i] = d
    else:
        for i in range(n):
            for j in range(i,n):
                if i==j:
                    continue
                d = np.max(np.abs(data[i]-data[j]))
                D[i,j] = d; D[j,i] = d
    return D

def lp_distances(data, vertices = None, p = 1):
    if not isinstance(vertices, np.ndarray):
        return lp(data, p )
    
    n = len(vertices); m = len(data)
    D = np.zeros([n, m])

    if p != "inf":
        for i in range(n):
            for j in range(m):
                D[i,j] = np.sum(np.abs(vertices[i]-data[j]))
    else:
        for i in range(n):
            for j in range(m):
                D[i,j] = np.max(np.abs(vertices[i]-data[j]))
    return D


def mahalanobis(data, robust = True):
    n,m = data.shape
    Cov = np.zeros((m, m))

    if robust:
        for i in range(m):
            for j in range(i, m):
                cov = np.sum((data[:,i]-np.median(data[:,i]))*(data[:,j]-np.median(data[:,j])))/(n-1)
                Cov[i,j] = cov
                if i!=j:
                    Cov[j,i] = cov
    else:
        for i in range(m):
            for j in range(i,m):
                cov = np.sum((data[:,i]-np.mean(data[:,i]))*(data[:,j]-np.mean(data[:,j])))/(n-1)
                Cov[i,j] = cov
                if i!=j:
                    Cov[j,i] = cov

    D = np.zeros([n, n])
    
    for i in range(n):
        for j in range(n):
            xi_xj =  (data[i]-data[j])

            Cov_inv = np.linalg.inv(Cov)

            D[i,j] = np.sqrt(np.dot(np.dot(xi_xj.reshape(1, -1), Cov_inv), xi_xj))

    return D, Cov


def mahalanobis_distances(data, vertices = None, robust = True):
    #not complete yet
    if not isinstance(vertices, np.ndarray):
        return mahalanobis(data, robust = robust)
    
    n = len(vertices); m = len(data)
    D = np.zeros([n, m])
    for i in range(n):
        for j in range(m):
            D[i,j] = np.sqrt(np.sum((vertices[i] - data[j])**2))
    return D

def cosine(data):
    n = len(data)
    D = np.zeros([n, n])
    for i in range(n):
        for j in range(i, n):
            if i == j:
                continue
            d = np.sum(np.dot(data[i],data[j]))/(np.sum(data[i]**2)*np.sum(data[j]**2))
            D[i, j] = d; D[j, i] = d
    return D

def cosine_distances(data, vertices = None):
    if not isinstance(vertices, np.ndarray):
        return cosine(data)
    
    n = len(vertices); m = len(data)
    D = np.zeros([n, m])
    for i in range(n):
        for j in range(m):
            D[i,j] = np.sum(np.dot(vertices[i],data[j]))/(np.sum(vertices[i]**2)*np.sum(data[j]**2))
    return D

def distance(data, vertices = None, metric = 'euclidean', p = 'inf', robust = True):
    if metric == 'euclidean': return euclidean_distances(data, vertices)
    elif metric == 'manhattan': return manhattan_distances(data, vertices)
    elif metric == 'lp': return lp_distances(data, vertices, p)
    elif metric == 'mahalanobis': return mahalanobis_distances(data, vertices, robust)
    elif metric == 'cosine': return cosine_distances(data, vertices)
    else: raise ValueError(f"Unsupported metric: {metric}, try 'euclidean' instead.") 

def unique_boxes(B, D, div = 4):
    n, m = D.shape
    # counts_rows = {f'Point {i}': [] for i in range(n)}
    # counts_cols = {f'Point {i}': [] for i in range(m)}
    M = []
    new_B = {f'Box {i}': [] for i in range(div)}
    for key in B.keys():
        counts_rows = {i: 0 for i in range(n)}
        counts_cols = {i: 0 for i in range(m)}
        for b in B[key]:

            counts_rows[b[0]]+=1
            counts_cols[b[1]]+=1
        M.append(counts_rows)
    maxes = np.zeros(n, dtype = int)
    maxis = np.zeros(n, dtype = int)
    for i in range(div):
        m = M[i]
        for j in range(n):
            # print('box',i, 'point',j,'freq',m[j])
            if maxes[j] < m[j]:
                maxes[j] = m[j]
                maxis[j] = i

    for i in range(len(maxis)): 
        new_B['Box '+str(maxis[i])].append(i)
        
    n, m = D.shape

    new_B2 = {f'Box {i}': [] for i in range(div)}
    maxes = np.zeros(m, dtype = int)
    maxis = np.zeros(m, dtype = int)
    for i in range(div):
        m_ = M[i]
        for j in range(m):
            # print('box',i, 'point',j,'freq',m[j])
            if maxes[j] < m_[j]:
                maxes[j] = m_[j]
                maxis[j] = i

    for i in range(len(maxis)): 
        new_B2['Box '+str(maxis[i])].append(i)
    return new_B, new_B2

def sort_distances(D, div = 4):
    box = boxes(D, div)
    for key in box.keys():
        box[key] = sorted(box[key], key = lambda x: x[2])

    # Iterate through the dictionary and filter out arrays where index 2 is zero
    for key, value in box.items():
        box[key] = [b for b in value if b[2] != 0]

    n,m = D.shape
    row = np.zeros(n, dtype=int); column = np.zeros(m, dtype=int)
    contr = 0; contc = 0
    for key in box.keys():
        b = box[key]
        for bi in b:
            if int(bi[0]) not in row:
                row[contr] = int(bi[0])
                contr+=1
            if int(bi[1]) not in column:
                column[contc] = int(bi[1])
                contc+=1

        # Rearrange the matrix D based on the specified order
    
    
    order, order_cols = unique_boxes(box, D, div)
    # Concatenate all the arrays into a single list
    order_ = []
    for key, value in order.items():
        order_.extend(value)

    order_cols_ = []
    for key, value in order_cols.items():
        order_cols_.extend(value)

    new_rows = D[order_, :]
    new_D = new_rows[:, order_cols_]
    
    return new_D, row, column