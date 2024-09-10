#pylint: disable=missing-module-docstring
import numpy as np
import numpy.typing as npt

from typing import Dict, Tuple, List, Union
from pyquaternion import Quaternion  # type: ignore
from nuscenes.nuscenes import NuScenes # type: ignore
from nuscenes.utils.geometry_utils import transform_matrix  # type: ignore
from nuscenes.utils.data_classes import Box  # type: ignore

def points_to_global(nusc: NuScenes,
                     lidar_data: Dict[str, Union[str, int]],
                     lidar_points: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    '''Moves lidar points from lidar to global coordinates.

    :param nusc: NuScenes class
    :param lidar_data: Lidar data dict from NuScenes
    :param lidar_points: Points in Lidar coordinates
    :return: Lidar points in global coordinates
    '''
    lidar_to_world_matrix = get_sensor_to_world_transformation(nusc, lidar_data)
    new_points = np.matmul(lidar_to_world_matrix, np.vstack((lidar_points.T, np.ones(lidar_points.shape[0]))))
    return np.asarray(new_points.T[:, :3], dtype=np.float32)

def get_lidar_points(lidar_data: Dict[str, Union[str, int]], dataroot: str) -> npt.NDArray[np.float32]:
    '''Moves lidar points from lidar to global coordinates.

    :param lidar_data: Lidar data dict from NuScenes
    :param dataroot: Path to NuScenes
    :return: Lidar points from file
    '''
    try:
        scan = np.fromfile(
            dataroot + '/' + str(lidar_data['filename']), dtype=np.float32)
    except FileNotFoundError:
        return np.asarray([], dtype=np.float32)

    points = scan.reshape((-1, 5))[:, :3]
    return np.asarray(points, dtype=np.float32)

def get_sensor_to_world_transformation(nusc: NuScenes,
                                       sensor_data: Dict[str, Union[str, int]]) -> npt.NDArray[np.float64]:
    '''Gets LiDAR to world transformation as a NumPy array for Nuscenes

    :param nusc: NuScenes object with sequence to convert
    :param sensor_data: sensor data for a specific frame
    :return: LiDAR to world transformation as a NumPy array
    '''
    sensor_calibration: dict = nusc.get(
        'calibrated_sensor', sensor_data['calibrated_sensor_token'])
    sensor_to_ego: np.ndarray = transform_matrix(
        sensor_calibration['translation'], Quaternion(sensor_calibration['rotation']))

    sensor_poserecord: dict = nusc.get(
        'ego_pose', sensor_data['ego_pose_token'])
    ego_to_world: np.ndarray = transform_matrix(
        sensor_poserecord['translation'], Quaternion(sensor_poserecord['rotation']))

    return ego_to_world @ sensor_to_ego

def box_to_list(box: Box) -> Dict[str, Tuple[str, float, List[float]]]:
    """Gets parameters of box, for later serialization

    :param box: Box
    :return: Dict
    """
    new_box: Dict = {}
    if box is None:
        return new_box
    new_box['translation'] = box.center.tolist()
    new_box['size'] = box.wlh.tolist()
    new_box['rotation'] = [box.orientation[0], box.orientation[1],
                              box.orientation[2], box.orientation[3]]
    new_box['detection_name'] = box.name
    new_box['detection_score'] = box.score
    return new_box

def view_points(points: npt.NDArray[np.float64],
                view: npt.NDArray[np.float64],
                normalize: bool) -> npt.NDArray[np.float64]:
    """Changes view of points to image

    :param points: points
    :param view: camera intrinsic
    :param normalize: normalize
    :return: points in image space
    """
    viewpad = np.eye(4)
    viewpad[:view.shape[0], :view.shape[1]] = view

    nbr_points = points.shape[1]

    # Do operation in homogenous coordinates.
    points = np.concatenate((points, np.ones((1, nbr_points))))
    points = np.dot(viewpad, points)
    points = points[:3, :]

    if normalize:
        points = points / points[2:3, :].repeat(3, 0).reshape(3, nbr_points)

    return points

def rotate(points: npt.NDArray[np.float64], rot_matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Applies rotation to points

    :param points: points
    :param rot_matrix: matrix
    :return: points
    """
    points = points.T
    points[:3, :] = np.dot(rot_matrix, points[:3, :])
    return points.T

def translate(points: npt.NDArray[np.float64], x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Applies translation it points

    :param points: points
    :param x: point
    :return: points
    """
    points = points.T
    for i in range(3):
        points[i, :] = points[i, :] + x[i]
    return points.T
