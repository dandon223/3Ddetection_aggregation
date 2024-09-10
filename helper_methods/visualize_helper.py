import os
import json
import numpy as np
from typing import List
from math import sqrt
from pyquaternion import Quaternion #type: ignore
from nuscenes.utils.geometry_utils import transform_matrix #type: ignore
from nuscenes.nuscenes import NuScenes #type: ignore
def get_sequence(sequence_file: str):
    """Get input sequence in json frim file

    :param sequence_file: File name
    :type sequence_file: str
    :return: Input for segmentation program
    :rtype: SequenceDataType
    """

    if not os.path.isfile(sequence_file):
        print(f'File {sequence_file} does not exist')
        return None
    with open(sequence_file, 'r', encoding='UTF-8') as file_object:
        json_content = file_object.read()
        return json.loads(json_content)

def resolve_conflict(ious, ground_truth_max_iou_index, detection_index, frame, false_positive):

    old_pair = (ground_truth_max_iou_index, ious[ground_truth_max_iou_index])

    ious[ground_truth_max_iou_index] = (detection_index, max(frame[detection_index]))
    frame[detection_index][ground_truth_max_iou_index] = -1 # This iou is now used to pair boxes

    # Now check new detection results for detection box from old_pair
    detection_index = old_pair[1][0]
    ground_truth_max_iou_index = frame[detection_index].index(max(frame[detection_index]))

    if ground_truth_max_iou_index not in ious:
        if max(frame[detection_index]) <= 0:

            false_positive.append(detection_index)
            return ious, ground_truth_max_iou_index, detection_index, false_positive

        ious[ground_truth_max_iou_index] = (detection_index, max(frame[detection_index]))
        frame[detection_index][ground_truth_max_iou_index] = -1

        #return ious, ground_truth_max_iou_index, detection_index, false_positive
    elif ious[ground_truth_max_iou_index][1] >= max(frame[detection_index]):
            false_positive.append(detection_index)
    else:
        print('!!!konflikt w srodku !!!')
        #input("Press Enter to continue...")
        ious, ground_truth_max_iou_index, detection_index, false_positive = resolve_conflict(ious, ground_truth_max_iou_index, detection_index, frame, false_positive)
    return ious, ground_truth_max_iou_index, detection_index, false_positive


def process_frame(frame):
    if len(frame) == 0:
        return {}, [], -1, 0
    number_of_ground_truth = len(frame[0])
    ious = {}
    false_positive = []
    is_conflict = 0
    if number_of_ground_truth == 0:
        return {}, [detection_index for detection_index, _ in enumerate(frame)], 0, False
    for detection_index, detection in enumerate(frame):
        ground_truth_max_iou_index = detection.index(max(detection))
        if max(detection) == 0:
            #false_positive = false_positive + 1
            false_positive.append(detection_index)
            continue
        # Ground truth box of id max_index was not paired yet
        if ground_truth_max_iou_index not in ious:
            ious[ground_truth_max_iou_index] = (detection_index, max(detection))
            detection[ground_truth_max_iou_index] = -1 # This iou is now used to pair boxes
        # Ground truth box of id max_index was already paired with detectec box with better iou
        elif ious[ground_truth_max_iou_index][1] >= max(detection):
            #false_positive = false_positive + 1
            false_positive.append(detection_index)
        else:
           ious, _, _, false_positive = resolve_conflict(ious, ground_truth_max_iou_index, detection_index, frame, false_positive)
           is_conflict = is_conflict + 1
            
    false_negative = number_of_ground_truth - len(ious)

    return ious, false_positive, false_negative, is_conflict

def get_lidar_to_world_transformation(nusc: NuScenes, lidar_data: dict) -> np.ndarray:
        '''Gets LiDAR to world transformation as a NumPy array for Nuscenes

        :param nusc: NuScenes object with sequence to convert
        :type nusc: NuScenes
        :param lidar_data: LiDAR data for a specific frame
        :type lidar_data: dict
        :return: LiDAR to world transformation as a NumPy array
        :rtype: np.ndarray
        '''
        lidar_calibration: dict = nusc.get(
            'calibrated_sensor', lidar_data['calibrated_sensor_token'])
        lidar_to_ego: np.ndarray = transform_matrix(
            lidar_calibration['translation'], Quaternion(lidar_calibration['rotation']))

        lidar_poserecord: dict = nusc.get(
            'ego_pose', lidar_data['ego_pose_token'])
        ego_to_world: np.ndarray = transform_matrix(
            lidar_poserecord['translation'], Quaternion(lidar_poserecord['rotation']))

        return ego_to_world @ lidar_to_ego

def get_global_points(nusc: NuScenes, scene: dict, dataroot: str) -> np.ndarray:
        '''Moves lidar points from lidar to global coordinates.

        :param nusc: NuScenes class
        :type nusc: NuScenes
        :param scene: Scene dict fron NuScenes
        :type scene: dict
        :param dataroot: Path to NuScenes
        :type dataroot: str
        :return: Lidar points in global coordinates
        :rtype: np.ndarray
        '''
        lidar_data = nusc.get('sample_data', scene['data']['LIDAR_TOP'])
        lidar_to_world_matrix = get_lidar_to_world_transformation(
            nusc, lidar_data)
        try:
            scan = np.fromfile(
                dataroot + '/' + lidar_data['filename'], dtype=np.float32)
        except FileNotFoundError:
            return np.asarray([], dtype=np.float32)

        points = scan.reshape((-1, 5))[:, :3]
        new_points = []
        for point in points:
            new_point = np.matmul(
                lidar_to_world_matrix[:3, :3], point) + lidar_to_world_matrix[:3, 3]
            new_points.append(new_point)
        return np.asarray(new_points, dtype=np.float32)

def get_distance(point_one: List[float], point_two: List[float]) -> float:
        """Returns distance for 2 2D points from each other

        :param point_one: First point
        :type point_one: List[float]
        :param point_two: Second point
        :type point_two: List[float]
        :return: Distance
        :rtype: float
        """
        return sqrt((point_one[0] - point_two[0]) ** 2 + (point_one[1] - point_two[1]) ** 2)