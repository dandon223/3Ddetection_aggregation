import json
import os
import open3d as o3d  # type: ignore
import argparse
import numpy as np
import yaml
import logging

from easydict import EasyDict
from pathlib import Path
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion
from functools import reduce

from tqdm import tqdm
from typing import List
from fgr.bbox_camera_to_lidar import find_frustum_corners
from fgr.segmentation import get_point_cloud_my_version, calculate_ground, get_segmented_points, lidar_to_cam_frame
from fgr.get_box_from_points import find_box_fgr
from fgr.merge_boxes import merge_duplicates
from helper_methods.nuscenes_helper import points_to_global, get_lidar_points, get_sensor_to_world_transformation

def show_points_and_boxes(points):

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.get_render_option().point_size = 5.0

    pcl = o3d.geometry.PointCloud()
    pcl.points = o3d.utility.Vector3dVector(points[:, :3])
    colours = np.array([[0.0, 0.0, 0.0] for _ in points])
    pcl.colors = o3d.utility.Vector3dVector(np.array(colours, dtype='float64'))
    vis.add_geometry(pcl)

    vis.run()
    vis.destroy_window()

def get_sweep_points(sweep, data_path, main_frame_from_car, main_car_from_global):
    def remove_ego_points(points, center_radius=1.0):
        mask = ~((np.abs(points[:, 0]) < center_radius) & (np.abs(points[:, 1]) < center_radius))
        return points[mask]
    
    lidar_path = data_path +'/'+ sweep['lidar_path']
    points_sweep = np.fromfile(str(lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])[:, :4]
    points_sweep = remove_ego_points(points_sweep).T

    car_from_ref = sweep['car_from_ref']
    global_from_car = sweep['global_from_car']
    
    trans_matrix = reduce(np.dot, [main_frame_from_car, main_car_from_global, global_from_car, car_from_ref])
    
    num_points = points_sweep.shape[1]
    points_sweep[:3, :] = trans_matrix.dot(np.vstack((points_sweep[:3, :], np.ones(num_points))))[:3, :]
    
    return points_sweep.T

def get_sweep(curr_sd_rec, data_path, nusc):

    lidar_path = nusc.get_sample_data_path(curr_sd_rec['token'])

    curr_pose_rec = nusc.get('ego_pose', curr_sd_rec['ego_pose_token'])
    curr_cs_rec = nusc.get('calibrated_sensor', curr_sd_rec['calibrated_sensor_token'])

    car_from_ref = transform_matrix(curr_cs_rec['translation'], Quaternion(curr_cs_rec['rotation']), inverse=False)
    global_from_car = transform_matrix(curr_pose_rec['translation'], Quaternion(curr_pose_rec['rotation']), inverse=False)

    sweep = {
        'lidar_path': Path(lidar_path).relative_to(data_path).__str__(),
        'sample_token': curr_sd_rec['sample_token'],
        'sample_data_token': curr_sd_rec['token'],
        'car_from_ref': car_from_ref,
        'global_from_car': global_from_car
        }
    return sweep

def get_lidar_points_with_sweeps(sample, nusc, data_path, main_frame_from_car, main_car_from_global):
    keyframe_sd_token = sample['data']['LIDAR_TOP']
    keyframe_sd_rec = nusc.get('sample_data', keyframe_sd_token)
    sample_token = keyframe_sd_rec['sample_token']
    sweeps = []
    temp_sd_rec = keyframe_sd_rec
    while temp_sd_rec['sample_token'] == sample_token:
        sweep = get_sweep(temp_sd_rec, data_path, nusc)
        sweeps.append(sweep)

        if temp_sd_rec['next'] != '':
            temp_sd_rec = nusc.get('sample_data', temp_sd_rec['next'])
        else:
            break

    temp_sd_rec = keyframe_sd_rec
    if temp_sd_rec['prev'] != '':
        temp_sd_rec = nusc.get('sample_data', temp_sd_rec['prev'])
        while temp_sd_rec['sample_token'] == sample_token:
            sweep = get_sweep(temp_sd_rec, data_path, nusc)
            sweeps.append(sweep)

            if temp_sd_rec['prev'] != '':
                temp_sd_rec = nusc.get('sample_data', temp_sd_rec['prev'])
            else:
                break
    
    sweep_points_list = []
    for sweep in sweeps:
        points_sweep = get_sweep_points(sweep, data_path, main_frame_from_car, main_car_from_global)
        sweep_points_list.append(points_sweep)
    
    points = np.concatenate(sweep_points_list, axis=0)
    return points[:, :3]

def box_area(corners: List[float]) -> float:
    '''
    Calculates the area of a rectangular box, given its two corners 

    :param corners: coordinates of two corners: x, y of bottom left and
        x, y of top right
    :type corners: List[float]

    :return: area of the box
    :rtype: float
    '''
    return (corners[2] - corners[0]) * (corners[3] - corners[1])

def main(args, region_growth_config):
    logging.basicConfig(filename='fgr_demo.log', level=logging.INFO)
    input_file = json.load(open(args.input_file, 'r'))
    nusc = NuScenes(dataroot=input_file['dataroot'],
                    version=input_file['version'],
                    verbose=True)

    output_sequences = []
    len_sequences = len(input_file['sequences'])
    logging.info(f'sequences {len_sequences}')
    for sequence in tqdm(input_file['sequences']):
        output_sequence = []
        for frame in tqdm(sequence):
            logging.info('------------------new_frame------------------------')

            boxes = frame['boxes']
            sample = nusc.get('sample', frame['frame_token'])
            lidar_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
            output_boxes = []
            #lidar_points = get_lidar_points(lidar_data, input_file['dataroot'])
            
            main_cs_record = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
            main_pose_record = nusc.get('ego_pose', lidar_data['ego_pose_token'])
            # Homogeneous transform from ego car frame to main frame
            main_frame_from_car = transform_matrix(main_cs_record['translation'], Quaternion(main_cs_record['rotation']), inverse=True)
            # Homogeneous transformation matrix from global to main ego car frame
            main_car_from_global = transform_matrix(main_pose_record['translation'], Quaternion(main_pose_record['rotation']), inverse=True)
            
            lidar_points = get_lidar_points_with_sweeps(sample, nusc, input_file['dataroot'], main_frame_from_car, main_car_from_global)
            global_points = points_to_global(nusc, lidar_data, lidar_points)
            #print('global_points', global_points.shape)
            #show_points_and_boxes(global_points)

            for box in boxes:
                logging.info(f'new_box: {box}')
                if box_area(box['bbox_corners']) <= region_growth_config.MIN_BOX_AREA:
                    logging.info(f'area {box_area(box["bbox_corners"])} to small, min: {region_growth_config.MIN_BOX_AREA}')
                    continue
                # FGR
                camera_data = nusc.get('sample_data', box['sample_data_token'])

                points = lidar_to_cam_frame(nusc, lidar_data, camera_data, lidar_points)

                pc_all, object_filter_all = get_point_cloud_my_version(
                    input_file['dataroot'], nusc, points, camera_data)

                mask_ground_all, ground_sample_points = calculate_ground(
                    points, 0.2, back_cut=False)
                _, object_filter = get_point_cloud_my_version(
                    input_file['dataroot'], nusc, points, camera_data, box['bbox_corners'])

                segmented_points, truncate = get_segmented_points(
                    region_growth_config, mask_ground_all, object_filter_all, object_filter,
                    pc_all, camera_data, box['bbox_corners'], global_points)
                if truncate is None or segmented_points is None:
                    continue

                camera_to_world = get_sensor_to_world_transformation(nusc, camera_data)
                cs_record = nusc.get(
                    'calibrated_sensor',
                    camera_data['calibrated_sensor_token'])
                camera_intrinsic = np.array(cs_record['camera_intrinsic'])
                corners = box['bbox_corners']
                proper_box = [corners[0], corners[3], corners[2] - corners[0], corners[1] - corners[3]]
                frustum_corners = find_frustum_corners(
                    proper_box, camera_intrinsic, camera_to_world, [0.25, 100])
                found_box = find_box_fgr(
                    np.array(segmented_points, dtype=np.float64),
                    region_growth_config,
                    np.array(ground_sample_points, dtype=np.float64),
                    truncate,
                    frustum_corners)
                if found_box is not None:
                    found_box.name = box['category_name']
                    found_box.score = box['detection_score']
                    output_boxes.append(found_box)
            #print('output_boxes', len(output_boxes))
            output_boxes = merge_duplicates(output_boxes, 0.7)
            output_frame = {'frame_token': frame['frame_token'], 'boxes': output_boxes}

            output_sequence.append(output_frame)
        output_sequences.append(output_sequence)
    output_file = {'dataroot': input_file['dataroot'],
                   'version': input_file['version'], 'sequences': output_sequences}
    with open(args.output_file, 'w') as fh:
        json.dump(output_file, fh, indent=4)
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input_file', type=str,
                        default='created_files/image_annotations.json', help='Input filename.')
    parser.add_argument('--output_file', type=str,
                        default='created_files/fgr/image_annotations.json', help='Output filename.')
    parser.add_argument('--region_growth_config_path', type=str,
                        default='fgr/config.yml', help='region_growth_config_path filename.')
    args = parser.parse_args()
    if (not os.path.isdir(os.path.dirname(args.output_file)) and os.path.dirname(args.output_file) != ''):
        raise Exception('directory name in output_file parameter does not exist')
    if not os.path.basename(args.output_file).endswith('.json'):
        raise Exception('file name in output_file parameter does not end in \'.json\'')
    with open(args.region_growth_config_path, 'r') as fp:
        region_growth_config = yaml.load(fp, Loader=yaml.FullLoader)
        region_growth_config = EasyDict(region_growth_config)
    main(args, region_growth_config)
