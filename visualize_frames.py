import argparse
import numpy as np
import open3d as o3d #type: ignore
from pathlib import Path
from matplotlib import pyplot as plt
from pyquaternion import Quaternion #type: ignore
from nuscenes.utils.data_classes import Box  #type: ignore
from nuscenes.nuscenes import NuScenes #type: ignore
from objectron.iou import IoU
from objectron.box import Box as ObjectronBox

from helper_methods.visualize_helper import get_sequence, process_frame, get_distance, get_lidar_to_world_transformation, get_global_points
from helper_methods.open3d_draw import draw_box

def main():

    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument(
        "--input_file",
        type=str,
        default="outputs/ted/700scenerios/nuscenes_output.json",
        help="specify the input json from TED",
    )
    parser.add_argument(
        "--ground_truth",
        type=str,
        default="outputs/ted/nuscenes_ground_truth_output.json",
        help="specify the input json of ground truth",
    )
    parser.add_argument(
        "--input_comparisonss",
        type=str,
        default="outputs/ted/700scenerios/nuscenes_output_ious.json",
        help="specify the input json of comparisonss of iou",
    )

    args = parser.parse_args()

    input_sequence = get_sequence(args.input_file)
    ground_truth = get_sequence(args.ground_truth)
    input_comparisonss = get_sequence(args.input_comparisonss)

    nusc = NuScenes(version='v1.0-trainval', dataroot=Path(input_sequence['dataroot']), verbose=True)

    for sequence_index, sequence in enumerate(input_comparisonss):
        for frame_index, frame in enumerate(sequence):

            scene = nusc.get('sample', input_sequence['sequences'][sequence_index][frame_index]['frame_token'])
            
            lidar_data = nusc.get('sample_data', scene['data']['LIDAR_TOP'])
            lidar_to_world = get_lidar_to_world_transformation(nusc, lidar_data)
            world_to_lidar = np.linalg.inv(lidar_to_world)

            #s_rec = nusc.get('sample',input_sequence['sequences'][sequence_index][frame_index]['frame_token'])
            #global_points = get_global_points(nusc, s_rec, input_sequence['dataroot'])
            scan = np.fromfile(input_sequence['dataroot'] + '/' + lidar_data['filename'], dtype=np.float32)
            lidar_points = scan.reshape((-1, 5))[:, :3]
            lidar_points = lidar_points[lidar_points[:,0] < 25]
            lidar_points = lidar_points[lidar_points[:,0] > -25]
            lidar_points = lidar_points[lidar_points[:,1] < 25]
            lidar_points = lidar_points[lidar_points[:,1] > -25]
            new_points = []
            for point in lidar_points:
                new_point = np.matmul(lidar_to_world[:3, :3], point) + lidar_to_world[:3, 3]
                new_points.append(new_point)
            global_points = np.asarray(new_points, dtype=np.float32)
            ious, _, _, _ = process_frame(frame)

            vis = o3d.visualization.Visualizer()
            vis.create_window()
            vis.get_render_option().point_size = 5.0
            found_box = False

            for ground_truth_index, (detetection_index, iou) in ious.items():

                box = input_sequence['sequences'][sequence_index][frame_index]['boxes'][detetection_index]
                detected_box = Box(box['translation'], box['size'], Quaternion(box['rotation']))

                box = ground_truth['sequences'][sequence_index][frame_index]['boxes'][ground_truth_index]
                ground_truth_box = Box(box['translation'], box['size'], Quaternion(box['rotation']))

                ground_truth_box_center_in_lidar = np.matmul(world_to_lidar[:3, :3], ground_truth_box.center) + world_to_lidar[:3, 3]

                #if get_distance(ground_truth_box_center_in_lidar, [0,0]) < 25 and iou < 0.3 and iou > 0.2:
                print('iou = ', iou)
                b1 = ObjectronBox.from_transformation(detected_box.orientation.rotation_matrix, detected_box.center, detected_box.wlh)
                b2 = ObjectronBox.from_transformation(ground_truth_box.orientation.rotation_matrix, ground_truth_box.center, ground_truth_box.wlh)
                loss = IoU(b1, b2)
                print('iou2=', loss.iou())
                vis = draw_box(vis, [ground_truth_box, detected_box], [[1, 0, 0], [0,1,0]])
                found_box = True
            
            if ground_truth['sequences'][sequence_index][frame_index] is not None:
                for box in ground_truth['sequences'][sequence_index][frame_index]['boxes']:
                    box = Box(box['translation'], box['size'], Quaternion(box['rotation']))
                    vis = draw_box(vis, [box], [[1, 0, 0]])
                    found_box = True
            if input_sequence['sequences'][sequence_index][frame_index] is not None:
                for box in input_sequence['sequences'][sequence_index][frame_index]['boxes']:
                    box = Box(box['translation'], box['size'], Quaternion(box['rotation']))
                    vis = draw_box(vis, [box], [[0, 1, 0]])
                    found_box = True

            if found_box:
                pcl = o3d.geometry.PointCloud()
                pcl.points = o3d.utility.Vector3dVector(global_points)
                vis.add_geometry(pcl)
                vis.run()
                vis.destroy_window()

if __name__ == '__main__':
    main()