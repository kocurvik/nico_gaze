import cv2
from eval_utils.image import load_calib_data
import numpy as np


# frameSize = (640, 480)
# frameSize = (1280, 720)

def detect_plane(img, calib_dir, chessboard_dim=35.0, chessboard_size=(7, 4)):
    mtx = calib_dir['new_K_l']
    dist = np.zeros_like(calib_dir['D_l'])

    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * chessboard_dim

    rvec, tvec = estimate_pose(img, objp, chessboard_size, mtx, dist)
    plane = compute_plane_equation(rvec, tvec)

    x = np.linspace(0, (chessboard_size[0] - 1) * chessboard_dim, chessboard_size[0])
    y = np.linspace(0, (chessboard_size[1] - 1) * chessboard_dim, chessboard_size[1])

    x_grid, y_grid = np.meshgrid(x, y)

    x_flat = x_grid.flatten()
    y_flat = y_grid.flatten()
    z_flat = np.zeros_like(x_flat)  # Z coordinates are all zero

    chessboard_corners = np.stack([x_flat, y_flat, z_flat], axis=1)

    chessboard_corners_3D = project_points_to_plane(chessboard_corners, tvec, rvec)

    chessboard_middle_points = calculate_chessboard_middle_points(chessboard_size, chessboard_dim)

    # points_to_project = np.array(chessboard_middle_points[:, :3], dtype=np.float32)
    # imgpts, _ = cv2.projectPoints(points_to_project, rvec, tvec, mtx, dist)
    # drawPoints(img, imgpts, (0, 0, 255))

    chessboard_middle_points = project_points_to_plane(chessboard_middle_points, tvec, rvec)

    # frameSize = (1280, 720)
    # resized_frame_R = cv2.resize(img, frameSize)
    # cv2.imshow('intersection', resized_frame_R)
    # cv2.waitKey(0)
    return plane, chessboard_middle_points


def drawPoints(img, points, color=(0, 0, 255)):
    for point in points:
        cv2.circle(img, tuple(point.ravel().astype(int)), 10, color, -1)
    return img


def find_chessboard(image, chessboard_size):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        return ret, corners2
    return ret, None


def estimate_pose(image, objp, chessboard_size, mtx, dist):
    ret, corners = find_chessboard(image, chessboard_size)
    flags = cv2.SOLVEPNP_IPPE
    success, rvecs, tvecs = cv2.solvePnP(objp, corners, mtx, dist, flags=flags)

    return rvecs, tvecs


def compute_plane_equation(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    normal = R[:, 2]  # The third column of R
    point_on_plane = tvec.reshape(-1)

    # Ax + By + Cz + D = 0
    A, B, C = normal
    D = -np.dot(normal, point_on_plane)

    plane = np.array([A, B, C, D])
    return plane


def calculate_z(plane, x, y):
    A, B, C, D = plane
    z = -(A * x + B * y + D) / C
    return np.array([x, y, z])


def find_intersection(plane, point, direction):
    A, B, C, D = plane
    x0, y0, z0 = point
    dx, dy, dz = direction

    denominator = A * dx + B * dy + C * dz
    if denominator == 0:
        return None

    t = -(A * x0 + B * y0 + C * z0 + D) / denominator

    # Calculate the intersection point
    x = x0 + t * dx
    y = y0 + t * dy
    z = z0 + t * dz
    intersection_point = np.array([x, y, z])
    return intersection_point


def calculate_chessboard_middle_points(chessboard_size, chessboard_dim=35.0, reversed=False):
    col, row = chessboard_size[0] + 1, chessboard_size[1] + 1
    x, y = np.meshgrid(np.arange(col), np.arange(row), indexing='ij')

    # Chessboard pattern filter
    mask = (y % 2 != x % 2)
    x_filtered = x[mask]
    y_filtered = y[mask]

    x_filtered = (x_filtered * chessboard_dim) - (chessboard_dim / 2)
    y_filtered = (y_filtered * chessboard_dim) - (chessboard_dim / 2)

    points = np.stack((x_filtered, y_filtered, np.zeros_like(x_filtered)), axis=-1)

    indices = np.lexsort((points[:, 0], points[:, 1]))
    points = points[indices]

    if (reversed):
        index_array = np.arange(1, len(points) + 1, 1).reshape(-1, 1)
    else:
        index_array = np.arange(len(points), 0, -1).reshape(-1, 1)
    points = np.hstack((points, index_array))
    return points


def project_points_to_plane(points, tvec, rvec):
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    coordinates = points[:, :3]

    # Apply the transformation to each point
    transformed_points = np.dot(coordinates, rotation_matrix.T) + tvec.T
    points[:, :3] = transformed_points
    return points


if __name__ == '__main__':
    calib_imgs_dir = 'C:/Users/Matej/Desktop/bakalarkaGit/bakalarka/BachelorThesis/nico_images/calibration'
    out_dir = 'C:/Users/Matej/Desktop/bakalarkaGit/bakalarka/BachelorThesis/nico_images/out'
    debug = 2
    eyeTrackerDir = "C:/Users/Matej/Desktop/bakalarkaGit/bakalarka/BachelorThesis/nico_images/eyetracker/with_chessboard/"
    calib_dir = load_calib_data(out_dir + "/calib_data.npy")
    img = cv2.imread(eyeTrackerDir + "0_l.png")
    detect_plane(img, calib_dir)
