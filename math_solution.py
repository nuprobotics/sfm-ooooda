import numpy as np


def get_projection(K, R, t):
    return K @ np.hstack((R.T, -R.T @ t))


def triangulation(
        camera_matrix: np.ndarray,
        camera_position1: np.ndarray,
        camera_rotation1: np.ndarray,
        camera_position2: np.ndarray,
        camera_rotation2: np.ndarray,
        image_points1: np.ndarray,
        image_points2: np.ndarray
):
    """
    :param camera_matrix: first and second camera matrix, np.ndarray 3x3
    :param camera_position1: first camera position in world coordinate system, np.ndarray 3x1
    :param camera_rotation1: first camera rotation matrix in world coordinate system, np.ndarray 3x3
    :param camera_position2: second camera position in world coordinate system, np.ndarray 3x1
    :param camera_rotation2: second camera rotation matrix in world coordinate system, np.ndarray 3x3
    :param image_points1: points in the first image, np.ndarray Nx2
    :param image_points2: points in the second image, np.ndarray Nx2
    :return: triangulated points, np.ndarray Nx3
    """

    points_3d = []
    P1 = get_projection(camera_matrix, camera_rotation1, camera_position1)
    P2 = get_projection(camera_matrix, camera_rotation2, camera_position2)
    N = image_points1.shape[0]
    for i in range(N):
        u1, v1 = image_points1[i]
        u2, v2 = image_points2[i]
        A = np.zeros((4, 4))
        A[0] = -u1 * P1[2] + P1[0]
        A[1] = v1 * P1[2] - P1[1]
        A[2] = -u2 * P2[2] + P2[0]
        A[3] = v2 * P2[2] - P2[1]
        _, _, Vt = np.linalg.svd(A)
        points_3d.append(Vt[-1][:3] / Vt[-1][3])
    return np.array(points_3d)
