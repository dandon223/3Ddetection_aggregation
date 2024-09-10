import os
import argparse
import json
import numpy as np
from pathlib import Path
from typing_extensions import TypedDict
from typing import Dict, List, Tuple, Union
from collections import OrderedDict
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes  # type: ignore
from nuscenes.utils.geometry_utils import view_points # type: ignore
from pyquaternion.quaternion import Quaternion
from shapely.geometry import MultiPoint, box

Frame = TypedDict('Frame', {'sample_token': str, 'boxes': List[Dict]})
SequenceDataType = TypedDict(
    'SequenceDataType', {'dataroot': str, 'sequence': List[Frame]})

NewFrame = TypedDict('Frame', {'sample_token': str, 'boxes': Dict[str, List]})
NewSequenceDataType = TypedDict(
    'SequenceDataType', {'dataroot': str, 'sequence': List[NewFrame]})

def generate_record(ann_rec: dict,
                    x1: float,
                    y1: float,
                    x2: float,
                    y2: float,
                    sample_data_token: str) -> OrderedDict:
    """
    Generate one 2D annotation record given various informations on top of the 2D bounding box coordinates.
    :param ann_rec: Original 3d annotation record.
    :param x1: Minimum value of the x coordinate.
    :param y1: Minimum value of the y coordinate.
    :param x2: Maximum value of the x coordinate.
    :param y2: Maximum value of the y coordinate.
    :param sample_data_token: Sample data token.
    :return: A sample 2D annotation record.
    """
    repro_rec = OrderedDict()
    relevant_keys = [
        'category_name',
        'sample_annotation_token',
        'visibility_token',
    ]

    for key, value in ann_rec.items():
        if key in relevant_keys:
            repro_rec[key] = value

    repro_rec['bbox_corners'] = [x1, y1, x2, y2]

    return repro_rec

def post_process_coords(corner_coords: List,
                        imsize: Tuple[int, int] = (1600, 900)) -> Union[Tuple[float, float, float, float], None]:
    """
    Get the intersection of the convex hull of the reprojected bbox corners and the image canvas, return None if no
    intersection.
    :param corner_coords: Corner coordinates of reprojected bounding box.
    :param imsize: Size of the image canvas.
    :return: Intersection of the convex hull of the 2D box corners and the image canvas.
    """
    polygon_from_2d_box = MultiPoint(corner_coords).convex_hull
    img_canvas = box(0, 0, imsize[0], imsize[1])

    if polygon_from_2d_box.intersects(img_canvas):
        img_intersection = polygon_from_2d_box.intersection(img_canvas)
        intersection_coords = np.array(
            [coord for coord in img_intersection.exterior.coords])

        min_x = min(intersection_coords[:, 0])
        min_y = min(intersection_coords[:, 1])
        max_x = max(intersection_coords[:, 0])
        max_y = max(intersection_coords[:, 1])

        return min_x, min_y, max_x, max_y
    else:
        return None

def get_2d_boxes(nusc: NuScenes, sample_data_token: str, visibilities: List[str]) -> List[OrderedDict]:
    """
    Get the 2D annotation records for a given `sample_data_token`.
    :param nusc: NuScenes
    :param sample_data_token: Sample data token belonging to a camera keyframe.
    :param visibilities: Visibility filter.
    :return: List of 2D annotation record that belongs to the input `sample_data_token`
    """

    # Get the sample data and the sample corresponding to that sample data.
    sd_rec = nusc.get('sample_data', sample_data_token)

    assert sd_rec['sensor_modality'] == 'camera', 'Error: get_2d_boxes only works for camera sample_data!'
    if not sd_rec['is_key_frame']:
        raise ValueError(
            'The 2D re-projections are available only for keyframes.')

    s_rec = nusc.get('sample', sd_rec['sample_token'])
    # Get the calibrated sensor and ego pose record to get the transformation matrices.
    cs_rec = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
    pose_rec = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    camera_intrinsic = np.array(cs_rec['camera_intrinsic'])

    # Get all the annotation with the specified visibilties.
    ann_recs = [nusc.get('sample_annotation', token)
                for token in s_rec['anns']]
    ann_recs = [ann_rec for ann_rec in ann_recs if (
        ann_rec['visibility_token'] in visibilities)]

    repro_recs = []
    for ann_rec in ann_recs:
        if 'car' not in ann_rec['category_name']:
            continue
        # Augment sample_annotation with token information.
        ann_rec['sample_annotation_token'] = ann_rec['token']
        ann_rec['sample_data_token'] = sample_data_token

        # Get the box in global coordinates.
        box = nusc.get_box(ann_rec['token'])
        # Move them to the ego-pose frame.
        box.translate(-np.array(pose_rec['translation']))
        box.rotate(Quaternion(pose_rec['rotation']).inverse)

        # Move them to the calibrated sensor frame.
        box.translate(-np.array(cs_rec['translation']))
        box.rotate(Quaternion(cs_rec['rotation']).inverse)

        # Filter out the corners that are not in front of the calibrated sensor.
        corners_3d = box.corners()
        in_front = np.argwhere(corners_3d[2, :] > 0).flatten()
        corners_3d = corners_3d[:, in_front]

        # Project 3d box to 2d.
        corner_coords = view_points(
            corners_3d, camera_intrinsic, True).T[:, :2].tolist()

        # Keep only corners that fall within the image.
        final_coords = post_process_coords(corner_coords)

        # Skip if the convex hull of the re-projected corners does not intersect the image canvas.
        if final_coords is None:
            continue
        else:
            min_x, min_y, max_x, max_y = final_coords

        # Generate dictionary record to be included in the .json file.
        repro_rec = generate_record(
            ann_rec, min_x, min_y, max_x, max_y, sample_data_token)
        repro_recs.append(repro_rec)

    return repro_recs

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
    
def prepare_sequence(input_file_prepered: str, input_sequence: SequenceDataType):
    if os.path.exists(input_file_prepered):
        return get_sequence(input_file_prepered)
    print('prepare input sequence')
    new_sequences = []
    for sequence in tqdm(input_sequence['sequences']):
        new_sequence = []
        for frame in sequence:
            new_boxes = {}
            for box in frame['boxes']:
                sample_data_token = box['sample_data_token']
                box.pop('sample_data_token')
                if sample_data_token not in new_boxes:
                    new_boxes[sample_data_token] = [box]
                else:
                    new_boxes[sample_data_token].append(box)
            new_frame: Frame = {'frame_token': frame['frame_token'], 'boxes': new_boxes}
            new_sequence.append(new_frame)
        new_sequences.append(new_sequence)
    prepared_sequence = {'dataroot': str(
        Path(input_sequence['dataroot'])), 'version':str(input_sequence['version']), 'sequences': new_sequences}
    return prepared_sequence

def get_ground_truth(input_sequence: SequenceDataType, output_ground_truth_file: str, visibilities: List) -> SequenceDataType:
    """Creates ground truth boxes file

    :param input_sequence: input sequence, for example from detector
    :type input_sequence: SequenceDataType
    :param output_ground_truth_file: name of ground truth file
    :type output_ground_truth_file: str
    :return: ground truth list
    :rtype: SequenceDataType
    """

    if os.path.exists(output_ground_truth_file):
        return get_sequence(output_ground_truth_file)

    nusc = NuScenes(version=Path(input_sequence['version']), dataroot=Path(
        input_sequence['dataroot']), verbose=True)

    new_sequences = []
    total_sequences = 0
    total_samples = 0
    print('create ground truth file')
    for sequence in tqdm(input_sequence['sequences']):
        new_sequence = []
        total_sequences = total_sequences + 1
        for frame in sequence:
            total_samples = total_samples + 1
            boxes = {}
            sample_data_dict = nusc.get('sample', frame['frame_token'])['data']
            cameras_tokens = [sample_data_dict['CAM_FRONT'], sample_data_dict['CAM_FRONT_RIGHT'], sample_data_dict['CAM_BACK_RIGHT'],
                              sample_data_dict['CAM_BACK'], sample_data_dict['CAM_BACK_LEFT'], sample_data_dict['CAM_FRONT_LEFT']]
            for camera_token in cameras_tokens:
                boxes_temp = get_2d_boxes(nusc, camera_token, visibilities)
                if len(boxes_temp) > 0:
                    boxes[camera_token] = boxes_temp
            new_frame: Frame = {
                'frame_token': frame['frame_token'], 'boxes': boxes}
            new_sequence.append(new_frame)
        new_sequences.append(new_sequence)
    ground_truth = {'dataroot': str(
        Path(input_sequence['dataroot'])), 'sequences': new_sequences}
    print(f'get_ground_truth: seq {total_sequences}, frames {total_samples}')
    return ground_truth

def get_box_comparisonss(output_ground_truth_file: str, sequence: SequenceDataType, ground_truth: SequenceDataType) -> List[List[List[float]]]:
    """Creates iou comparisons list between detected and ground truth boxes

    :param sequence: input sequence
    :type sequence: SequenceDataType
    :param ground_truth: ground truth sequence
    :type ground_truth: SequenceDataType
    :return: iou comparisons list between detected and ground truth boxes
    :rtype: List[List[List[float]]]
    """
    if os.path.exists(output_ground_truth_file):
        return get_sequence(output_ground_truth_file)
    sequences_box_comparisonss = []
    print('compare boxes')
    for sequence_index, scene in enumerate(tqdm(sequence['sequences'])):
        sequence_box_comparisonss = []
        for frame_index, frame in enumerate(scene):

            ground_truth_frame = ground_truth['sequences'][sequence_index][frame_index]
            if ground_truth_frame['frame_token'] != frame['frame_token']:
                print('Not same frames.')
                exit
            
            frame_boxes_comparisons = {}
            for camera_token, detected_boxes in frame['boxes'].items():
                frame_boxes_comparisons[camera_token] = []
                if camera_token not in ground_truth_frame['boxes']:
                    continue
                for box in detected_boxes:
                    frame_box_comparisons = []
                    for ground_truth_box in ground_truth_frame['boxes'][camera_token]:
                        frame_box_comparisons.append(get_iou(box, ground_truth_box))
                    frame_boxes_comparisons[camera_token].append(frame_box_comparisons)
            sequence_box_comparisonss.append(frame_boxes_comparisons)
        sequences_box_comparisonss.append(sequence_box_comparisonss)
    return sequences_box_comparisonss

def get_iou(box_one: Dict, box_two: Dict) -> float:
    """returns iou between two boxes

    :param box_one: box one
    :type box_one: Dict
    :param box_two: box two
    :type box_two: Dict
    :return: iou metric
    :rtype: float
    """
    b1 = box(minx=box_one['bbox_corners'][0], miny=box_one['bbox_corners'][1], maxx=box_one['bbox_corners'][2], maxy=box_one['bbox_corners'][3])
    b2 = box(minx=box_two['bbox_corners'][0], miny=box_two['bbox_corners'][1], maxx=box_two['bbox_corners'][2], maxy=box_two['bbox_corners'][3])
    return b1.intersection(b2).area / b1.union(b2).area

def main():

    parser = argparse.ArgumentParser(description="Prepare for metrics", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--input_file",
        type=str,
        default="created_files/yolo2/image_annotations.json",
        help="specify the input json from 2D detector",
    )
    parser.add_argument(
        "--input_file_prepered",
        type=str,
        default="created_files/yolo2/image_annotations_prepered.json",
        help="specify json prepered file from 2D detector (this file is created by script)",
    )
    parser.add_argument(
        "--output_file_ious",
        type=str,
        default="created_files/yolo2/image_annotations_ious.json",
        help="specify the output json",
    )
    parser.add_argument(
        "--output_ground_truth_file",
        type=str,
        default="created_files/yolo2/ground_truth_image_annotations.json",
        help="specify the output ground truth json",
    )
    parser.add_argument('--visibilities', type=str, default=['', '1', '2', '3', '4'],
                        help='Visibility bins, the higher the number the higher the visibility.', nargs='+')

    args = parser.parse_args()

    input_sequence = get_sequence(args.input_file)
    input_sequence = prepare_sequence(args.input_file_prepered, input_sequence)

    json_object = json.dumps(input_sequence, indent=4)
    # Writing to sample.json
    with open(args.input_file_prepered, "w") as outfile:
        outfile.write(json_object)

    ground_truth = get_ground_truth(
        input_sequence, args.output_ground_truth_file, args.visibilities)

    json_object = json.dumps(ground_truth, indent=4)
    # Writing to sample.json
    with open(args.output_ground_truth_file, "w") as outfile:
        outfile.write(json_object)
    
    box_comparisonss = get_box_comparisonss(args.output_file_ious, input_sequence, ground_truth)
    
    json_object = json.dumps(box_comparisonss, indent=4)
    # Writing to sample.json
    with open(args.output_file_ious, "w") as outfile:
        outfile.write(json_object)

if __name__ == '__main__':
    main()