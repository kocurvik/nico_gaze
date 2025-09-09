import numpy as np


def addDicertionVectorToPoint(point, dVector, length):
    x, y, z = point
    dx, dy, dz = dVector
    x += length * dx
    y += length * dy
    z += length * dz
    return np.array([x, y, z]).astype(np.int32)


def vectors_angle(v1, v2):
    dot_product = np.dot(v1, v2)
    dot_product = np.clip(dot_product, -1.0, 1.0)

    angle_radians = np.arccos(dot_product)
    angle_degrees = np.degrees(angle_radians)
    return angle_degrees


def add_dVector_3D(point, dVector, length):
    point = point + (dVector * length)
    return point


def get_direction_vector_from_line(point1, point2):
    dVector = point2 - point1
    return dVector / np.linalg.norm(dVector)


def distance_3D(point1, point2):
    distance = np.linalg.norm(np.array(point2) - np.array(point1))
    return distance
