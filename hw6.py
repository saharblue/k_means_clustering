import numpy as np
from numpy import float64


def get_random_centroids(X, k):
    '''
    Each centroid is a point in RGB space (color) in the image. 
    This function should uniformly pick `k` centroids from the dataset.
    Input: a single image of shape `(num_pixels, 3)` and `k`, the number of centroids. 
    Notice we are flattening the image to a two dimentional array.
    Output: Randomly chosen centroids of shape `(k,3)` as a numpy array. 
    '''
    centroids = []
    #np.random.seed(42)
    num_samples = X.shape[0]

    # generate k unique random indices in the range of the number of samples
    random_indices = np.random.choice(num_samples, size=k, replace=False)

    # select the random lines from your dataset
    centroids = X[random_indices, :]

    # make sure you return a numpy array
    return np.asarray(centroids).astype(np.float)


def lp_distance(X, centroids, p=2):
    '''
    Inputs: 
    A single image of shape (num_pixels, 3)
    The centroids (k, 3)
    The distance parameter p

    output: numpy array of shape `(k, num_pixels)` thats holds the distances of 
    all points in RGB space from all centroids
    '''
    distances = []
    k = len(centroids)
    # Subtract centroids from X using broadcasting.
    diff = X[:, np.newaxis] - centroids

    # Compute Minkowski distance
    distances = np.empty((k, len(X)), dtype=float)
    for i in range(k):
        distances[i] = calc_minkowski(np.zeros_like(diff[:, i]), diff[:, i], p)

    return distances


def calc_minkowski(x, y, p=2):
    """
    Calculate the minkowski distance between two vectors.
    Inputs:
    - x: a numpy array
    - y: a numpy array
    - p: the parameter governing the distance measure.
    Output:
    - The minkowski distance between the two vectors.
    """
    return np.sum(np.abs(x - y) ** p, axis=-1) ** (1 / p)


def kmeans(X, k, p, max_iter=100):
    """
    Inputs:
    - X: a single image of shape (num_pixels, 3).
    - k: number of centroids.
    - p: the parameter governing the distance measure.
    - max_iter: the maximum number of iterations to perform.

    Outputs:
    - The calculated centroids as a numpy array.
    - The final assignment of all RGB points to the closest centroids as a numpy array.
    """
    classes = []
    centroids = get_random_centroids(X, k)

    return general_kmeans(X, k, p, centroids, max_iter)


def kmeans_pp(X, k, p, max_iter=100):
    """
    Your implementation of the kmeans++ algorithm.
    Inputs:
    - X: a single image of shape (num_pixels, 3).
    - k: number of centroids.
    - p: the parameter governing the distance measure.
    - max_iter: the maximum number of iterations to perform.

    Outputs:
    - The calculated centroids as a numpy array.
    - The final assignment of all RGB points to the closest centroids as a numpy array.
    """
    data = X.copy()
    centroids = np.zeros((k, X.shape[1]))

    #np.random.seed(42)
    random_index = np.random.choice(X.shape[0], size=1, replace=False)
    centroids[0] = data[random_index]
    data = np.delete(data, random_index, axis=0)

    for i in range(1, k):
        diff = data[:, np.newaxis] - centroids
        # Compute Minkowski distance
        distances = np.empty((k, len(data)), dtype=float64)
        for j in range(k):
            distances[j] = (calc_minkowski(np.zeros_like(diff[:, j]), diff[:, j], p)) ** 2

        mins = np.min(distances[:i], axis=0)
        probs = mins / mins.sum()

        # Choose the next centroid
        random_index = np.random.choice(data.shape[0], size=1, replace=False, p=probs)
        centroids[i] = data[random_index]
        data = np.delete(data, random_index, axis=0)

    return general_kmeans(X, k, p, centroids, max_iter)


def general_kmeans(X, k, p, centroids, max_iter=100):
    """
    Inputs:
    - X: a single image of shape (num_pixels, 3).
    - k: number of centroids.
    - p: the parameter governing the distance measure.
    - centroids: the initial centroids as a numpy array.
    - max_iter: the maximum number of iterations to perform.

    Outputs:
    - The calculated centroids as a numpy array.
    - The final assignment of all RGB points to the closest centroids as a numpy array.
    """
    classes = []

    for i in range(max_iter):
        current_centroids = np.zeros_like(centroids)
        distances = lp_distance(X, centroids, p)
        classes = np.argmin(distances, axis=0)

        for j in range(k):
            current_centroids[j] = np.mean(X[classes == j], axis=0)

        if np.array_equal(current_centroids, centroids):
            break
        centroids = current_centroids
    return centroids, classes


def calc_inertia(data, centroids, classes, p=2):
    """
    Calculate the inertia of the given data with the given centroids and classes.
    Inputs:
    - data: a numpy array of shape (num_pixels, 3).
    - centroids: a numpy array of shape (k, 3).
    - classes: a numpy array of shape (num_pixels,).
    Output:
    - The inertia of the given data with the given centroids and classes.
    """
    data = np.hstack((data, classes.reshape(-1, 1)))
    distances = np.apply_along_axis(lambda x: calc_minkowski(x[:-1], centroids[x[-1]], p), axis=1, arr=data)
    distances = distances ** 2
    return distances.sum()
