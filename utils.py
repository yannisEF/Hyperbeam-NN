import pathlib
import numpy as np

from functools import wraps
from time import time

# ------------------------------- Others -------------------------------

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:%r took: %2.9f sec' % \
        (f.__name__, te-ts))
        return result
    return wrap

def get_timestamp():
    return int(time())

def make_path(path):
    pathlib.Path.mkdir(pathlib.Path(path), parents=True, exist_ok=True)

# ------------------------------- Vectors -------------------------------

def distance_euclidian(x, y):
    """Returns the euclidian distance between x and y."""
    return np.linalg.norm(np.array(x)-np.array(y))

def order_neighbours(list_vectors):
    """
    Orders a list of vectors by inserting them between their average nearest neighbours.
    Returns the ordered list and the list of indices to reproduce the order.
    """
    
    indices = [0]
    ordered_vectors = [list_vectors[0]]
    for vector in list_vectors[1:]:

        # Compute the distance to the already ordered vectors
        distance_to_others = list(map(
            lambda other: distance_euclidian(vector, other),
            ordered_vectors
        ))

        # Get the average distance 
        #   ... (using the minimum distance can easily break the already ordered vectors)
        mean_distance = [distance_to_others[0]] + list(map(
            lambda i: np.mean([distance_to_others[i], distance_to_others[i+1]]),
            range(1, len(distance_to_others) - 1)
        )) + [distance_to_others[-1]]

        # Insert it where it minimizes its average distance to the others
        indices.append(np.argmin(mean_distance))
        ordered_vectors.insert(indices[-1], vector)

    return ordered_vectors, indices

def insert_with_indices(A, indices):
    """Orders A by inserting items one by one according to the input indices."""

    assert len(A) == len(indices), "Size of list and insert indices don't match."

    B = []
    for i in range(len(A)):
        B.insert(indices[i], A[i])

    return B
