#pylint: disable=missing-module-docstring
import os
import logging
import numpy as np
import numpy.typing as npt

from easydict import EasyDict # type: ignore
from typing import Dict, List, Tuple, Optional, Union
from nuscenes.nuscenes import NuScenes # type: ignore
from pyquaternion import Quaternion  # type: ignore
from helper_methods.nuscenes_helper import rotate, translate, view_points
from PIL import Image # type: ignore

def lidar_to_cam_frame(nusc: NuScenes, lidar_data: Dict[str, Union[str, int]], camera_data: Dict[str, Union[str, int]],
                       lidar_points: npt.NDArray[np.float32]) -> npt.NDArray[np.float64]:
    """Moves points from lidar to camera space

    :param nusc: NuScenes
    :param lidar_data: Dictionary of pointsensor from NuScenes
    :param camera_data: Dictionary of camera from NuScenes
    :param lidar_points: Lidar points in lidar view
    :return: Lidar points in camera frame view
    """
    cs_record = nusc.get(
        'calibrated_sensor',
        lidar_data['calibrated_sensor_token'])
    points = rotate(np.array(lidar_points, dtype=np.float64), Quaternion(
        cs_record['rotation']).rotation_matrix)
    points = translate(points, np.array(cs_record['translation']))

    # Second step: transform from ego to the global frame.
    poserecord = nusc.get('ego_pose', lidar_data['ego_pose_token'])
    points = rotate(points, Quaternion(poserecord['rotation']).rotation_matrix)
    points = translate(points, np.array(poserecord['translation']))

    # Third step: transform from global into the ego vehicle frame for the
    # timestamp of the image.
    poserecord = nusc.get('ego_pose', camera_data['ego_pose_token'])
    points = translate(points, -np.array(poserecord['translation']))
    points = rotate(
        points, Quaternion(
            poserecord['rotation']).rotation_matrix.T)

    # Fourth step: transform from ego into the camera.
    cs_record = nusc.get('calibrated_sensor', camera_data['calibrated_sensor_token'])
    points = translate(points, -np.array(cs_record['translation']))
    points = rotate(
        points, Quaternion(
            cs_record['rotation']).rotation_matrix.T)
    return points

def _points_in_2dbox(points_im: npt.NDArray[np.float64],
                     box: List[float]) -> npt.NDArray[np.bool_]:
    point_filter = (points_im[:, 0] > box[0]) & \
                   (points_im[:, 0] < box[2]) & \
                   (points_im[:, 1] > box[1]) & \
                   (points_im[:, 1] < box[3])
    return point_filter

def get_point_cloud_my_version(dataroot: str,
                               nusc: NuScenes,
                               points: npt.NDArray[np.float64],
                               camera_data: Dict[str, Union[str, int]],
                               box2d: Optional[List[float]] = None) -> Tuple[npt.NDArray[np.float64],
                                                                             npt.NDArray[np.bool_]]:
    """Gets points that are projected on camera, and if box2d is present that are inside this box

    :param dataroot: root of NuScenes database
    :param nusc: nuscenes object
    :param points, lidar points of sample in camera space
    :param camera_data: Dictionary with camera sensor data from NuScenes
    :param box2d: 2dbox on image, defaults to None
    :return: Found points, and its mask
    """

    # Fifth step: actually take a "picture" of the point cloud.
    # Grab the depths (camera frame z axis points away from the camera).
    depths = points.T[2, :]
    cs_record = nusc.get(
        'calibrated_sensor',
        camera_data['calibrated_sensor_token'])
    points_in_im = view_points(
        points.T,
        np.array(
            cs_record['camera_intrinsic']),
        normalize=True).T
    im = Image.open(os.path.join(dataroot, str(camera_data['filename'])))
    image_filter = np.ones(depths.shape[0], dtype=np.bool_)
    image_filter = np.logical_and(image_filter, depths > 1)
    image_filter = np.logical_and(image_filter, points_in_im[:, 0] > 1)
    image_filter = np.logical_and(
        image_filter, points_in_im[:, 0] < im.size[0] - 1)
    image_filter = np.logical_and(image_filter, points_in_im[:, 1] > 1)
    image_filter = np.logical_and(
        image_filter, points_in_im[:, 1] < im.size[1] - 1)

    object_filter = np.zeros(points_in_im.shape[0], dtype=np.bool_)
    if box2d is not None:
        object_filter = np.logical_or(
            _points_in_2dbox(
                points_in_im,
                box2d),
            object_filter)
        object_filter = np.logical_and(image_filter, object_filter)
    else:
        object_filter = image_filter

    return points.T, object_filter

def _check_parallel(points: npt.NDArray[np.float64]) -> bool:
    a = np.linalg.norm(points[0] - points[1])
    b = np.linalg.norm(points[1] - points[2])
    c = np.linalg.norm(points[2] - points[0])
    p = (a + b + c) / 2

    if p * (p - a) * (p - b) * (p - c) < 0:
        return True
    area = np.sqrt(p * (p - a) * (p - b) * (p - c))
    return bool(area < 1e-2)

def fit_plane(points: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Fits plane from points

    :param points: points
    :return: plane
    """
    if points.shape[0] == points.shape[1]:
        return np.linalg.solve(points, np.ones(points.shape[0]))
    else:
        return np.linalg.lstsq(points, np.ones(points.shape[0]), rcond=None)[0]

def calculate_ground(points: npt.NDArray[np.float64],
                     thresh_ransac: float,
                     back_cut: Optional[bool] = False,
                     back_cut_z: float = -5.0) -> Tuple[npt.NDArray[np.bool_],
                                                        Optional[npt.NDArray[np.float64]]]:
    """Calculates points that are ground plane

    :param points: lidar points of sample in camera space
    :param thresh_ransac: Threshold
    :param back_cut: If cut points behind camera, defaults to False
    :param back_cut_z: How far behind to cut, defaults to -5.0
    :return: Mask of points, and that points
    """

    if back_cut:
        points = points[points[:, 2] > back_cut_z]   # camera frame 3 x N
    temp = np.sort(points[:, 1])[int(points.shape[0] * 0.75)]
    cloud = points[points[:, 1] > temp]
    points_np = points
    mask_all = np.ones(points_np.shape[0], dtype=np.bool_)
    final_sample_points = None
    for _ in range(5):
        best_len = 0
        for _ in range(min(cloud.shape[0], 100)):
            sampled_points = cloud[np.random.choice(
                np.arange(cloud.shape[0]), size=(3), replace=False)]

            while _check_parallel(sampled_points):
                sampled_points = cloud[np.random.choice(
                    np.arange(cloud.shape[0]), size=(3), replace=False)]
                continue

            plane = fit_plane(sampled_points)
            diff = np.abs(np.matmul(points_np, plane) -
                          np.ones(points_np.shape[0])) / np.linalg.norm(plane)
            inlier_mask = diff < thresh_ransac
            num_inliers = inlier_mask.sum()
            if num_inliers > best_len and np.abs(
                    np.dot(plane / np.linalg.norm(plane), np.array([0, 1, 0]))) > 0.9:
                mask_ground = inlier_mask
                best_len = num_inliers
                final_sample_points = sampled_points
        mask_all = np.logical_and(mask_all, ~mask_ground, dtype=np.bool_)
    return mask_all, final_sample_points

def _region_grow_my_version(pc: npt.NDArray[np.float64],
                            mask_search: npt.NDArray[np.bool_],
                            mask_origin: npt.NDArray[np.bool_],
                            thresh: float,
                            ratio: Optional[float] = 0.8) -> npt.NDArray[np.bool_]:
    pc_search = pc[mask_search == 1]
    mask = mask_origin.copy()
    best_len = 0
    mask_best = np.zeros((pc.shape[0]), dtype=np.bool_)
    while mask.sum() > 0:
        seed = pc[mask == 1][0]
        seed_mask = np.zeros((pc_search.shape[0]))
        seed_mask_all = np.zeros((pc.shape[0]), np.bool_)
        seed_list = [seed]
        flag = 1
        while len(seed_list) > 0:
            temp = seed_list.pop(0)
            dis = np.linalg.norm(pc_search - temp, axis=-1)
            index = np.argmin(dis)
            seed_mask[index] = 1
            valid_mask = (dis < thresh) * (1 - seed_mask)
            seed_list += list(pc_search[valid_mask == 1])
            seed_mask[valid_mask == 1] = 1
            seed_mask_all[mask_search == 1] = seed_mask
            if ratio is not None and (
                    seed_mask_all * mask_origin).sum() / seed_mask.sum().astype(np.float32) < ratio:
                flag = 0
                break
        if flag == 1:
            if seed_mask.sum() > best_len:
                best_len = seed_mask.sum()
                mask_best = seed_mask_all
        mask = np.logical_and(mask, ~seed_mask_all, dtype=np.bool_)

    if ratio is not None:
        return np.logical_and(mask_best, mask_origin)
    else:
        return mask_best

def _check_truncate(img_shape: List[float], bbox_corners: List[float]) -> bool:
    return bool(min(bbox_corners[0], bbox_corners[1]) < 1 or \
                bbox_corners[2] > img_shape[0] - 2 or \
                bbox_corners[3] > img_shape[1] - 2)

def get_segmented_points(region_growth_config: EasyDict,
                         mask_ground_all: npt.NDArray[np.bool_],
                         object_filter_all: npt.NDArray[np.bool_],
                         object_filter: npt.NDArray[np.bool_],
                         pc_all: npt.NDArray[np.float64],
                         camera_data: Dict[str, Union[str, int]],
                         bbox_corners: List[float],
                         global_points: npt.NDArray[np.float32]) -> Union[Tuple[npt.NDArray[np.float32], bool],
                                                                          Tuple[None, None]]:
    """get points of segmented object

    :param region_growth_config: Config
    :param mask_ground_all: Mask of ground points
    :param object_filter_all: Mask of object
    :param object_filter: Mask of object
    :param pc_all: Points from point cloud
    :param cam: Dictionary of camera from NuScenes
    :param bbox_corners: 2d box corners in image space
    :param global_points: Points in global space
    :return: Segmented points, if 2d box is truncated
    """
    count = 0
    mask_seg_list: List[npt.NDArray[np.bool_]] = []
    thresh_seg_max = region_growth_config.THRESH_SEG_MAX
    ratio = region_growth_config.REGION_GROWTH_RATIO
    result = np.zeros((7, 2))
    for j in range(thresh_seg_max):
        thresh = (j + 1) * 0.1
        mask_search = mask_ground_all * object_filter_all
        mask_origin = mask_ground_all * object_filter
        mask_seg = _region_grow_my_version(
            pc_all.copy().T, mask_search, mask_origin, thresh, ratio)
        if mask_seg.sum() == 0:
            continue

        if j >= 1:
            mask_seg_old = mask_seg_list[-1]
            if mask_seg_old.sum() != (mask_seg * mask_seg_old).sum():
                count += 1

        result[count, 0] = j
        result[count, 1] = mask_seg.sum()
        mask_seg_list.append(mask_seg)

    best_j = result[np.argmax(result[:, 1]), 0]
    try:
        mask_seg_best = mask_seg_list[int(best_j)]
    except IndexError:
        logging.info('bad region grow result! deprecated')
        return None, None

    truncate = False
    if _check_truncate(
            [float(camera_data['width']), float(camera_data['height'])], bbox_corners):
        mask_origin_new = mask_seg_best
        mask_search_new = mask_ground_all
        thresh_new = float((best_j + 1) * 0.1)
        mask_seg_for_truncate = _region_grow_my_version(
            pc_all.copy().T, mask_search_new, mask_origin_new, thresh_new, ratio=None)
        mask_seg_best = mask_seg_for_truncate
        truncate = True

    segmented_points = global_points[mask_seg_best == 1].copy()
    return segmented_points, truncate
