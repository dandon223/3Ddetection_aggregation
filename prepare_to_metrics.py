
import os
import argparse
import json
import numpy as np
from pathlib import Path
from nuscenes.nuscenes import NuScenes #type: ignore
from typing_extensions import TypedDict
from typing import Dict, List
from pyquaternion import Quaternion #type: ignore
from objectron.box import Box
from objectron.iou import IoU

Frame = TypedDict('Frame', {'sample_token': str, 'boxes': List[Box]})
SequenceDataType = TypedDict(
    'SequenceDataType', {'dataroot': str, 'sequence': List[Frame]})

def get_sequence(sequence_file: str) -> SequenceDataType:
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

def get_ground_truth(input_sequence: SequenceDataType, output_ground_truth_file) -> SequenceDataType:

    if  os.path.exists(output_ground_truth_file):
        return get_sequence(output_ground_truth_file)

    nusc = NuScenes(version=input_sequence['version'], dataroot=Path(input_sequence['dataroot']), verbose=True)

    new_sequences = []
    index = 1
    total_sequences = 0
    total_samples = 0
    # number_of_lidar_points = {}
    for sequence in input_sequence['sequences']:
        print(f'sequence {index}')
        index = index + 1
        new_sequence = []
        total_sequences = total_sequences + 1
        for frame in sequence:
            total_samples = total_samples + 1
            sample = nusc.get('sample', frame['frame_token'])
            annotations = sample['anns']
            boxes = []
            for ground_truth_token in annotations:
                ground_truth_metadata = nusc.get('sample_annotation', ground_truth_token)
                instance_metadata = nusc.get('instance', ground_truth_metadata['instance_token'])
                category_metadata = nusc.get('category', instance_metadata['category_token'])
                if 'car' not in category_metadata['name']:
                    continue
                # if ground_truth_metadata['num_lidar_pts'] >= 50:
                #     number_of_lidar_points['>50'] = number_of_lidar_points.get('>50', 0) + 1
                # else:
                #     number_of_lidar_points[ground_truth_metadata['num_lidar_pts']] = number_of_lidar_points.get(ground_truth_metadata['num_lidar_pts'], 0) + 1
                box = {}
                box['translation'] = ground_truth_metadata['translation']
                box['size'] = ground_truth_metadata['size']
                box['rotation'] = ground_truth_metadata['rotation']
                box['id'] = ground_truth_metadata['token']
                box['name'] = category_metadata['name']
                #box['index'] = category_metadata['index']
                boxes.append(box)
            new_frame: Frame = {'frame_token': frame['frame_token'], 'boxes': boxes}
            new_sequence.append(new_frame)
        new_sequences.append(new_sequence)
    ground_truth = {'dataroot': str(Path(input_sequence['dataroot'])), 'sequences': new_sequences}
    print(f'get_ground_truth: seq {total_sequences}, frames {total_samples}')
    # print('number_of_lidar_points', number_of_lidar_points)
    return ground_truth

def get_box_comparisonss(sequence: SequenceDataType, ground_truth: SequenceDataType) -> List[List[List[float]]]:
    sequences_box_comparisonss = []
    print('get_box_comparisonss')
    for sequence_index, scene in enumerate(sequence['sequences']):
        print(sequence_index)
        sequence_box_comparisonss = []
        for frame_index, frame in enumerate(scene):

            ground_truth_frame = ground_truth['sequences'][sequence_index][frame_index]
            assert ground_truth_frame['frame_token'] == frame['frame_token']
            frame_boxes_comparisons = []
            for box in frame['boxes']:

                frame_box_comparisons = []
                for ground_truth_box in ground_truth_frame['boxes']:

                    frame_box_comparisons.append(get_iou(box, ground_truth_box))
                frame_boxes_comparisons.append(frame_box_comparisons)
            sequence_box_comparisonss.append(frame_boxes_comparisons)
        sequences_box_comparisonss.append(sequence_box_comparisonss)
    return sequences_box_comparisonss

def get_iou(box_one: Dict, box_two: Dict)-> float:
    b1 = Box.from_transformation(Quaternion(box_one['rotation']).rotation_matrix, np.asarray(box_one['translation']), np.asarray(box_one['size']))
    b2 = Box.from_transformation(Quaternion(box_two['rotation']).rotation_matrix, np.asarray(box_two['translation']), np.asarray(box_two['size']))
    loss = IoU(b1, b2)
    return loss.iou()

def main():

    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument(
        "--input_file",
        type=str,
        default="outputs/fgr_3/image_annotations.json",
        help="specify the input json",
    )
    parser.add_argument(
        "--output_file_ious",
        type=str,
        default="outputs/fgr_3/image_annotations_ious.json",
        help="specify the output json",
    )
    parser.add_argument(
        "--output_ground_truth_file",
        type=str,
        default="outputs/fgr_3/image_annotations_ground_truth_output.json",
        help="specify the output ground truth json",
    )

    args = parser.parse_args()

    input_sequence = get_sequence(args.input_file)

    ground_truth = get_ground_truth(input_sequence, args.output_ground_truth_file)
    
    box_comparisonss = get_box_comparisonss(input_sequence, ground_truth)

    json_object = json.dumps(ground_truth, indent=4)
    # Writing to sample.json
    with open(args.output_ground_truth_file, "w") as outfile:
        outfile.write(json_object)

    json_object = json.dumps(box_comparisonss, indent=4)
    # Writing to sample.json
    with open(args.output_file_ious, "w") as outfile:
        outfile.write(json_object)
    

if __name__ == '__main__':
    main()