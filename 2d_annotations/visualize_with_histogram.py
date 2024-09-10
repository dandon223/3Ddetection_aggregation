import os
import argparse
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
from prepare_to_metric import get_sequence
from matplotlib import pyplot as plt

def resolve_conflict(ious: Dict, ground_truth_max_iou_index: int, detection_index: int, frame: List, false_positive: List) -> Tuple[Dict, int, int, List]:
    """Resloves conflict when new detected box has better iou with already taken ground truth box.

    :param ious: List of ious matches with this frame
    :type ious: Dict
    :param ground_truth_max_iou_index: index of best ground truth box for box
    :type ground_truth_max_iou_index: int
    :param detection_index: box index
    :type detection_index: int
    :param frame: processed frame
    :type frame: List
    :param false_positive: list of false positive indices
    :type false_positive: List
    :return: fixed ious dictionary, new best ground truth index, new detection index, list of false positive indices
    :rtype: Tuple[Dict, int, int, List]
    """
    old_pair = (ground_truth_max_iou_index, ious[ground_truth_max_iou_index])

    ious[ground_truth_max_iou_index] = (
        detection_index, max(frame[detection_index]))
    frame[detection_index][ground_truth_max_iou_index] = - \
        1  # This iou is now used to pair boxes

    # Now check new detection results for detection box from old_pair
    detection_index = old_pair[1][0]
    ground_truth_max_iou_index = frame[detection_index].index(
        max(frame[detection_index]))

    if ground_truth_max_iou_index not in ious:
        if max(frame[detection_index]) <= 0:

            false_positive.append(detection_index)
            return ious, ground_truth_max_iou_index, detection_index, false_positive

        ious[ground_truth_max_iou_index] = (
            detection_index, max(frame[detection_index]))
        frame[detection_index][ground_truth_max_iou_index] = -1

        # return ious, ground_truth_max_iou_index, detection_index, false_positive
    elif ious[ground_truth_max_iou_index][1] >= max(frame[detection_index]):
        false_positive.append(detection_index)
    else:
        ious, ground_truth_max_iou_index, detection_index, false_positive = resolve_conflict(
            ious, ground_truth_max_iou_index, detection_index, frame, false_positive)
    return ious, ground_truth_max_iou_index, detection_index, false_positive


def process_camera(iou_list: List) -> Tuple[Dict, List, int, int]:
    """Processes iou comparisons file and returns best ious.

    :param frame: iou comparisons file
    :type frame: List
    :return: ious, false_positive, false_negative, is_conflict
    :rtype: Tuple[Dict, List, int, int]
    """
    number_of_ground_truth = len(iou_list[0])
    ious = {}
    false_positive = []
    is_conflict = 0
    if number_of_ground_truth == 0:
        return {}, [detection_index for detection_index, _ in enumerate(iou_list)], 0, False
    for detection_index, detection in enumerate(iou_list):
        ground_truth_max_iou_index = detection.index(max(detection))
        if max(detection) == 0:
            # false_positive = false_positive + 1
            false_positive.append(detection_index)
            continue
        # Ground truth box of id max_index was not paired yet
        if ground_truth_max_iou_index not in ious:
            ious[ground_truth_max_iou_index] = (
                detection_index, max(detection))
            detection[ground_truth_max_iou_index] = - \
                1  # This iou is now used to pair boxes
        # Ground truth box of id max_index was already paired with detectec box with better iou
        elif ious[ground_truth_max_iou_index][1] >= max(detection):
            # false_positive = false_positive + 1
            false_positive.append(detection_index)
        else:
            ious, _, _, false_positive = resolve_conflict(
                ious, ground_truth_max_iou_index, detection_index, iou_list, false_positive)
            is_conflict = is_conflict + 1

    false_negative = number_of_ground_truth - len(ious)

    return ious, false_positive, false_negative, is_conflict

def main():

    parser = argparse.ArgumentParser(description='Visualize results', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--input_file',
        type=str,
        default='created_files/yolo2/image_annotations_prepered.json',
        help='specify the input json from 2D detector',
    )
    parser.add_argument(
        '--ground_truth',
        type=str,
        default='created_files/yolo2/ground_truth_image_annotations.json',
        help='specify the input json of ground truth',
    )
    parser.add_argument(
        '--input_comparisonss',
        type=str,
        default='created_files/yolo2/image_annotations_ious.json',
        help='specify the input json of comparisonss of iou',
    )
    parser.add_argument(
        '--histograms_save',
        type=str,
        default='created_files/yolo2/histograms',
        help='specify the output files for histograms',
    )
    parser.add_argument(
        '--ths',
        type=float,
        default='0',
        help='specify the threshold of true positive detection',
    )

    args = parser.parse_args()

    if not os.path.isdir(os.path.dirname(args.histograms_save)):
        raise Exception('bad output_file parameter')

    input_sequence = get_sequence(args.input_file)
    ground_truth = get_sequence(args.ground_truth)
    input_comparisonss = get_sequence(args.input_comparisonss)
    if input_sequence == None or ground_truth == None or input_comparisonss == None:
        raise Exception('at least one of sequences not found')
    
    best_ious = []
    false_positive_sum = 0
    false_negative_sum = 0
    number_of_detections = 0
    for sequence_index, sequence in enumerate(input_comparisonss):
        for frame_index, frame in enumerate(sequence):
            detected_camera_tokens = []

            for camera_token, iou_list in frame.items():
                detected_camera_tokens.append(camera_token)
                # If no ground truth for this camera_token
                if len(iou_list) == 0:
                    false_positive_sum += len(input_sequence['sequences'][sequence_index][frame_index]['boxes'][camera_token])
                else:
                    ious, false_positive, false_negative, _ = process_camera(iou_list)
                    false_positive_sum += len(false_positive)
                    false_negative_sum += false_negative
                    for _, (_, iou) in ious.items():
                        number_of_detections += 1
                        best_ious.append(iou)
            
            # Some cameras have ground truth boxes but detector did not create on it any detection
            for camera_token, ground_truth_boxes in ground_truth['sequences'][sequence_index][frame_index]['boxes'].items():
                if camera_token not in detected_camera_tokens:
                    false_negative_sum += len(ground_truth_boxes)

    detector_name = 'YOLO_v8'
    with open(Path(args.histograms_save) / f'{detector_name}.txt', 'w') as file:
        plt.clf()
        plt.hist(best_ious, (len(best_ious) // 100) + 10)
        plt.xlabel('IoU')
        plt.ylabel('Liczba adnotacji')
        plt.title(f'Histogram IoU - {detector_name}')
        plt.savefig(f'{args.histograms_save}/{detector_name}.png')

        best_ious_ths = []
        for best_iou in best_ious:
            if best_iou > 0.5:
               best_ious_ths.append(best_iou) 

        file.write(
            f'TP {number_of_detections}, FP {false_positive_sum}, FN {false_negative_sum}, mean {np.mean(best_ious)}, mean >=0.5 {np.mean(best_ious_ths)}\n')

if __name__ == '__main__':
    main()