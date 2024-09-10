"""This file creates histograms from TED detectors of IOUs depending on haw far detections are placed from lidar sensor"""
import argparse
from pathlib import Path

from matplotlib import pyplot as plt
from pyquaternion import Quaternion #type: ignore
from nuscenes.utils.data_classes import Box  #type: ignore
from nuscenes.nuscenes import NuScenes #type: ignore
import numpy as np
from helper_methods.visualize_helper import get_sequence, process_frame, get_distance, get_lidar_to_world_transformation

def main():
    prefix = "outputs/merge_annotations/merge_with_strategy2/700scenerios"
    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument(
        "--input_file",
        type=str,
        default=prefix + "/nuscenes_output.json",
        help="specify the input json from TED",
    )
    parser.add_argument(
        "--ground_truth",
        type=str,
        default= "outputs/nuscenes_ground_truth_output.json",
        help="specify the input json of ground truth",
    )
    parser.add_argument(
        "--input_comparisonss",
        type=str,
        default=prefix +"/nuscenes_output_ious.json",
        help="specify the input json of comparisonss of iou",
    )
    parser.add_argument(
        "--histograms_save",
        type=str,
        default=prefix +"/histograms",
        help="specify the output files for histograms",
    )
    parser.add_argument(
        "--detector_name",
        type=str,
        default="FGR",
        help="specify the name of a detector",
    )

    args = parser.parse_args()

    input_sequence = get_sequence(args.input_file)
    ground_truth = get_sequence(args.ground_truth)
    input_comparisonss = get_sequence(args.input_comparisonss)

    false_positive_sum = 0
    false_negative_sum = 0
    conflict_sum = 0
    nusc = NuScenes(version='v1.0-trainval', dataroot="/data/sets/nuscenes/v1.0-trainval/v1.0-trainval", verbose=True)

    keys = [25, 50, 100]
    histograms = {}
    for key in keys:
        histograms[key] = []

    false_positives = {}
    for key in keys:
        false_positives[key] = 0

    false_negatives = {}
    for key in keys:
        false_negatives[key] = 0

    for sequence_index, sequence in enumerate(input_comparisonss):
        for frame_index, frame in enumerate(sequence):

            scene = nusc.get('sample', input_sequence['sequences'][sequence_index][frame_index]['frame_token'])
            
            lidar_data = nusc.get('sample_data', scene['data']['LIDAR_TOP'])
            lidar_to_world = get_lidar_to_world_transformation(nusc, lidar_data)
            world_to_lidar = np.linalg.inv(lidar_to_world)

            ious, false_positive, false_negative, conflict = process_frame(frame)

            false_positive_sum = false_positive_sum + len(false_positive)
            if false_negative == -1:
                false_negative_sum = false_negative_sum + len(ground_truth['sequences'][sequence_index][frame_index]['boxes'])
            else:
                false_negative_sum = false_negative_sum + false_negative
            conflict_sum = conflict_sum + conflict

            number_of_ground_truth = len(ground_truth['sequences'][sequence_index][frame_index]['boxes'])
            for ground_truth_index in range(number_of_ground_truth):
                if ground_truth_index not in ious:
                    box = ground_truth['sequences'][sequence_index][frame_index]['boxes'][ground_truth_index]
                    ground_truth_box = Box(box['translation'], box['size'], Quaternion(box['rotation']))
                    ground_truth_box_center_in_lidar = np.matmul(world_to_lidar[:3, :3], ground_truth_box.center) + world_to_lidar[:3, 3]

                    for key in keys:
                        if get_distance(ground_truth_box_center_in_lidar, [0,0]) < key:
                            false_negatives[key] = false_negatives[key] + 1
                            break


            for false_positive_index in false_positive:
                box = input_sequence['sequences'][sequence_index][frame_index]['boxes'][false_positive_index]
                detected_box = Box(box['translation'], box['size'], Quaternion(box['rotation']))
                detected_box_box_center_in_lidar = np.matmul(world_to_lidar[:3, :3], detected_box.center) + world_to_lidar[:3, 3]

                for key in keys:
                    if get_distance(detected_box_box_center_in_lidar, [0,0]) < key:
                        false_positives[key] = false_positives[key] + 1
                        break

            for ground_truth_index, (_, iou) in ious.items():
                box = ground_truth['sequences'][sequence_index][frame_index]['boxes'][ground_truth_index]
                ground_truth_box = Box(box['translation'], box['size'], Quaternion(box['rotation']))

                ground_truth_box_center_in_lidar = np.matmul(world_to_lidar[:3, :3], ground_truth_box.center) + world_to_lidar[:3, 3]

                for key in keys:
                    if get_distance(ground_truth_box_center_in_lidar, [0,0]) < key:
                        histograms[key].append(iou)
                        break
    
    print('conflict_sum = ', conflict_sum)
    detector_name = args.detector_name
    true_positive = 0
    with open(Path(args.histograms_save) / f"{detector_name}.txt", "w") as file:
        value_all = []
        for key, value  in histograms.items():
            for v in value:
                value_all.append(v)
            true_positive = true_positive + len(value)
            plt.clf()
            plt.hist(value, (len(value) // 100) + 10)
            plt.xlabel("IoU")
            plt.ylabel("Liczba adnotacji")
            plt.title(f"Histogram IoU - {detector_name}")
            plt.savefig(f"{args.histograms_save}/{detector_name}_<{key}.png")

            value_more_than = list(filter(lambda score: score >= 0.5, value))
            plt.clf()
            plt.hist(value_more_than, (len(value_more_than) // 100) + 10)
            plt.xlabel("IoU")
            plt.ylabel("Liczba adnotacji")
            plt.title(f"Histogram IoU - {detector_name}")
            plt.savefig(f"{args.histograms_save}/{detector_name}_<{key}>0.5iou.png")

            file.write(f'<{key}m, TP {len(value)}, FP {false_positives[key]}, FN {false_negatives[key]}, mean {np.mean(value)}, mean >=0.5 {np.mean(value_more_than)}\n')

        
        plt.clf()
        plt.hist(value_all, (len(value_all) // 100) + 10)
        plt.xlabel("IoU")
        plt.ylabel("Liczba adnotacji")
        plt.title(f"Histogram IoU - {detector_name}")
        plt.savefig(f"{args.histograms_save}/{detector_name}_all.png")

        value_more_than = list(filter(lambda score: score >= 0.5, value_all))
        plt.clf()
        plt.hist(value_more_than, (len(value_more_than) // 100) + 10)
        plt.xlabel("IoU")
        plt.ylabel("Liczba adnotacji")
        plt.title(f"Histogram IoU - {detector_name}")
        plt.savefig(f"{args.histograms_save}/{detector_name}_all>0.5iou.png")

        file.write(f'false_positive_sum {false_positive_sum}\n')
        file.write(f'false_negative_sum {false_negative_sum}\n')
        file.write(f'true_positive_sum {true_positive}, mean {np.mean(value_all)}, true_positive_sum>=0.5 {len(value_more_than)}, mean >=0.5 {np.mean(value_more_than)}\n')

if __name__ == '__main__':
    main()
