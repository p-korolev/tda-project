"""
The following module enables projection for points in $R^d$ to a point in $R^1$.
"""

import numpy as np
import pandas as pd

from numpy.typing import ArrayLike

# conversion for following functions
def _to_vector(x: ArrayLike) -> np.ndarray:
    '''
    Convert input to a 1D numpy array of dtype float64.
    '''
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"Expected a 1D vector, got shape {arr.shape}")
    return arr

def mean(x: ArrayLike) -> np.float64:
    return np.float64(np.mean(x))

def max(x: ArrayLike) -> np.float64:
    return np.float64(np.max(x))

def min(x: ArrayLike) -> np.float64:
    return np.float64(np.min(x))

def smoothmax(x: ArrayLike) -> np.float64:
    s = 0
    for i in range(len(x)):
        s += np.e**x[i]
    return np.float64(np.log(s))

def project_point(point: ArrayLike, method: str) -> np.float64:
    '''
    :param point: Point in $R^d$ with np.ArrayLike structure
    :param method: Projection method

    **Projection Methods**
    avg: computes mean of coordinates
    max: computes maximum of coordinates
    min: computes minimum of coordinates
    norm2: computes the 2-norm of point
    sum: computes sum of coordinates
    sdfo: computes squared distance from origin
    smoothmax: computes smooth max of coordinates
    '''
    vec = _to_vector(x=point)
    if method=="avg":
        return mean(vec)
    if method=="max":
        return max(vec)
    if method=="min":
        return min(vec)
    if method=="norm2":
        return np.float64(np.linalg.norm(vec))
    if method=="sum":
        return np.float64(np.sum(vec))
    if method=="sdfo":
        return np.float64(np.dot(vec, vec))
    if method=="smoothmax":
        return smoothmax(vec)
    else:
        raise ValueError("Method must be valid.")

test = [1,2,3,4,5]
print(project_point(test, "sdfo"))