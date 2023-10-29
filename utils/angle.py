import numpy as np
import math


def angle_between_vectors_degrees(u, v):
    """Return the angle between two vectors in any dimension space,
    in degrees."""
    # Clamp the value to the range [-1, 1] to prevent math domain errors
    dot_product = np.dot(u, v)
    norms_product = np.linalg.norm(u) * np.linalg.norm(v)
    value = dot_product / norms_product
    value = max(-1.0, min(1.0, value))

    return np.degrees(math.acos(value))


def get_angle_3_pts(A: tuple, B: tuple, C: tuple):
    """Return the angle between three points in long lat."""
    # Convert the points to numpy latitude/longitude radians space
    a = np.radians(np.array(A))
    b = np.radians(np.array(B))
    c = np.radians(np.array(C))

    # Vectors in latitude/longitude space
    avec = a - b
    cvec = c - b

    # Adjust vectors for changed longitude scale at given latitude into 2D space
    lat = b[0]
    avec[1] *= math.cos(lat)
    cvec[1] *= math.cos(lat)

    # Find the angle between the vectors in 2D space
    angle2deg = angle_between_vectors_degrees(avec, cvec)

    return 180-angle2deg
