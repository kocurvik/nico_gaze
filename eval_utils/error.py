import numpy as np


def vectors_angle(v1, v2):
    dot_product = np.dot(v1 / np.linalg.norm(v1), v2 / np.linalg.norm(v2))
    dot_product = np.clip(dot_product, -1.0, 1.0)

    angle_radians = np.arccos(dot_product)
    angle_degrees = np.degrees(angle_radians)
    return angle_degrees


def get_direction_vector_from_line(point1, point2):
    dVector = point2 - point1
    return dVector / np.linalg.norm(dVector)


def distance_3D(point1, point2):
    distance = np.linalg.norm(np.array(point2) - np.array(point1))
    return distance

def get_result_dict(view_point, intersection_point, gt_direction, est_direction, head_point):
    d = {}
    d['distance_error_3d'] = distance_3D(view_point, intersection_point)
    d['angular_error_3d'] = vectors_angle(gt_direction, est_direction)
    d['angular_error_2d'] = vectors_angle(gt_direction[:2], est_direction[:2])

    d['intersection_point'] = intersection_point.tolist()
    d['view_point'] = view_point.tolist()
    d['gt_direction'] = gt_direction.tolist()
    d['est_direction'] = est_direction.tolist()
    d['head_point'] = head_point.tolist()

    return d