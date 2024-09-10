#pylint: disable=missing-module-docstring
import numpy as np
import numpy.typing as npt
from typing import List

def find_points_in_frustum(
        lidar_frustum_corners: npt.NDArray[np.float64],
        global_points: npt.NDArray[np.float64]) -> npt.NDArray[np.int_]:
    """Finds indices of points inside frustum from global points.

    :param lidar_frustum_corners: Corners of created frustum
    :param global_points: Global points.
    :return: Points inside frustum.
    """
    #      6-------7
    #      |       |
    #    5------8  |
    #    | |    |  |
    #    | 2----|--3
    #    |      |
    #    1------4
    # pylint: disable=invalid-name
    P1 = np.array(lidar_frustum_corners[3])
    P2 = np.array(lidar_frustum_corners[7])
    P3 = np.array(lidar_frustum_corners[6])
    P4 = np.array(lidar_frustum_corners[2])
    P5 = np.array(lidar_frustum_corners[0])
    P6 = np.array(lidar_frustum_corners[4])
    P7 = np.array(lidar_frustum_corners[5])
    P8 = np.array(lidar_frustum_corners[1])
    # pylint: enable=invalid-name

    u_front = np.cross((P5 - P1), (P4 - P1))  # front face normal
    v_front = np.cross((P2 - P1), (P5 - P1))  # left face normal
    w_front = np.cross((P4 - P1), (P2 - P1))  # bottom face normal

    # Convert the normal vector to unit vectors
    u_front_unit = u_front / np.linalg.norm(u_front)
    v_front_unit = v_front / np.linalg.norm(v_front)
    w_front_unit = w_front / np.linalg.norm(w_front)

    n = global_points.shape[0]
    inlier_indices = np.array(range(0, n), dtype=np.int_)
    vec_front = global_points - np.tile(P1, (n, 1))

    u_front_dot = np.sum(vec_front * np.tile(u_front_unit,
                                             (n, 1)), axis=1, keepdims=False)
    v_front_dot = np.sum(vec_front * np.tile(v_front_unit,
                                             (n, 1)), axis=1, keepdims=False)
    w_front_dot = np.sum(vec_front * np.tile(w_front_unit,
                                             (n, 1)), axis=1, keepdims=False)

    u_back = np.cross((P6 - P7), (P3 - P7))  # back face normal
    v_back = np.cross((P8 - P7), (P6 - P7))  # top face normal
    w_back = np.cross((P3 - P7), (P8 - P7))  # right face normal

    # Convert the normal vector to unit vectors
    u_back_unit = u_back / np.linalg.norm(u_back)
    v_back_unit = v_back / np.linalg.norm(v_back)
    w_back_unit = w_back / np.linalg.norm(w_back)

    vec_back = global_points - np.tile(P7, (n, 1))

    u_back_dot = np.sum(vec_back * np.tile(u_back_unit,
                                           (n, 1)), axis=1, keepdims=False)
    v_back_dot = np.sum(vec_back * np.tile(v_back_unit,
                                           (n, 1)), axis=1, keepdims=False)
    w_back_dot = np.sum(vec_back * np.tile(w_back_unit,
                                           (n, 1)), axis=1, keepdims=False)
    masks = [u_back_dot <= 0, v_back_dot <= 0, w_back_dot <= 0,
             u_front_dot <= 0, v_front_dot <= 0, w_front_dot <= 0]
    total_mask = masks[0] & masks[1] & masks[2] & masks[3] & masks[4] & masks[5]
    total_mask = masks[0] & masks[1] & masks[2] & masks[3] & masks[4] & masks[5]
    return inlier_indices[total_mask]

def _get_corners_from_bbox(bbox: List[float]) -> npt.NDArray[np.float64]:
    """Returns the four corners from the rectangle defined as
    [x, y, l, w]. The corners are arranged in the following fashion:
           (0)---------------(1)
            |                 |
            |                 |
            |                 |
            |                 |
           (3)---------------(2)
    """
    corners_camera = np.zeros((4, 2))
    corners_camera[0, :] = bbox[0:2]
    corners_camera[1, :] = bbox[0:2] + np.array([bbox[2], 0])
    corners_camera[2, :] = bbox[0:2] + np.array(bbox[2:4])
    corners_camera[3, :] = bbox[0:2] + np.array([0, bbox[3]])
    return corners_camera

def _transform_cam_to_lidar(cam_corner: npt.NDArray[np.float64],
                            camera_intrinsic: npt.NDArray[np.float64],
                            camera_to_world: npt.NDArray[np.float64],
                            max_detection_range: List[float]) -> List[List[float]]:
    fx = camera_intrinsic[0, 0]
    cx = camera_intrinsic[0, 2]
    fy = camera_intrinsic[1, 1]
    cy = camera_intrinsic[1, 2]

    zlim = [max_detection_range[0], max_detection_range[1]]
    xlim = [0., 0.]
    ylim = [0., 0.]
    xlim[0] = float(((cam_corner[0] - cx) * zlim[0]) / fx)
    ylim[0] = float(((cam_corner[1] - cy) * zlim[0]) / fy)
    xlim[1] = float(((cam_corner[0] - cx) * zlim[1]) / fx)
    ylim[1] = float(((cam_corner[1] - cy) * zlim[1]) / fy)
    world_points_in_cam = [
        [xlim[0], ylim[0], zlim[0]], [xlim[1], ylim[1], zlim[1]]]
    new_world_points_in_cam = []
    for point in world_points_in_cam:
        new_point: List[float] = np.matmul(
            camera_to_world[:3, :3], point) + camera_to_world[:3, 3]
        new_world_points_in_cam.append(new_point)
    return new_world_points_in_cam

def find_frustum_corners(proper_box: List[float],
                         camera_intrinsic: npt.NDArray[np.float64],
                         camera_to_world: npt.NDArray[np.float64],
                         max_detection_range: List[float]) -> npt.NDArray[np.float64]:
    """Finds the corners for frustum from rectangular corners in image.

    :param proper_box: Proper corners of 2d box
    :param camera_intrinsic: camera intrinsicpip install autopep8
    :param camera_to_world: Transformation matrix
    :param max_detection_range: Max detection range inside frustum
    :return: Corner points of frustum.
    """
    corners_camera = _get_corners_from_bbox(proper_box)

    world_points1 = _transform_cam_to_lidar(
        corners_camera[0, :], camera_intrinsic, camera_to_world, max_detection_range)
    world_points2 = _transform_cam_to_lidar(
        corners_camera[1, :], camera_intrinsic, camera_to_world, max_detection_range)
    world_points3 = _transform_cam_to_lidar(
        corners_camera[2, :], camera_intrinsic, camera_to_world, max_detection_range)
    world_points4 = _transform_cam_to_lidar(
        corners_camera[3, :], camera_intrinsic, camera_to_world, max_detection_range)
    back = np.array([world_points1[1], world_points2[1],
                     world_points3[1], world_points4[1]], dtype=np.float64)
    front = np.array([world_points1[0], world_points2[0],
                      world_points3[0], world_points4[0]], dtype=np.float64)
    return np.concatenate((front, back), axis=0)
