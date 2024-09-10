import argparse
import json
import os
import logging
from ultralytics import YOLO
from nuscenes.nuscenes import NuScenes
from tqdm import tqdm
from collections import OrderedDict
from typing import List
from nuscenes.utils import splits

def at_least_one_box(sequence: List) -> bool:
    """Checks if there is atleast one box in scene

    :param sequence: scene
    :type sequence: List
    :return: true fi there is atleast one box
    :rtype: bool
    """
    for frame in sequence:
        if len(frame['boxes']) > 0:
            return True
    return False

def get_nusc_scenes(nusc: NuScenes) -> List:
    """Returns list of scenes with sample tokens

    :param nusc: NuScenes
    :type nusc: NuScenes
    :return: list of scenes
    :rtype: List
    """
    scenes = []
    for scene in nusc.scene:
        samples = []
        my_sample = nusc.get('sample', scene['first_sample_token'])
        while True:
            samples.append(my_sample['token'])
            if my_sample['next'] == '':
                break
            else:
                my_sample = nusc.get('sample', my_sample['next'])
        scenes.append(samples)
    return scenes

def generate_record(box, sample_data_token):

    box_record = OrderedDict()
    box_record['sample_data_token'] = sample_data_token
    if box.cls == 2:
        box_record['category_name'] = 'car' 
    else:
        box_record['category_name'] = int(box.cls)
    xyxy = box.xyxy[0]
    box_record['bbox_corners'] = [float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])]
    box_record['detection_score'] = float(box.conf)
    return box_record


def get_2d_boxes(sample_data_token: str, nusc: NuScenes, model: YOLO, dataroot: str) -> List[OrderedDict]:
    """
    Get the 2D annotations records for a given picture in `sample_data_token`.
    :param sample_data_token: Sample data token belonging to a camera keyframe.
    :param nusc: Nuscenes database
    :param model: YOLO model
    :param dataroot: dataroot of a nuscenes database
    :return: List of 2D annotation record that belongs to the input `sample_data_token`
    """
    sd_rec = nusc.get('sample_data', sample_data_token)
    assert sd_rec['sensor_modality'] == 'camera', 'Error: get_2d_boxes only works for camera sample_data!'
    file_name = sd_rec['filename']
    results = model(dataroot + "/" + file_name, verbose=False)
    detections = []
    for result in results:
        result = result.cpu().numpy()
        for box in result.boxes:
            detections.append(generate_record(box, sample_data_token))
    return detections

def main(args):
    """Generates 2D adnotations of nuscenes dataset."""
    logging.basicConfig(filename='yolo.log', level=logging.INFO)
    logging.info('start')
    model = YOLO(args.model)
    nusc = NuScenes(dataroot=args.dataroot, version=args.version)

    # Get tokens for all camera images.
    sample_data_camera_tokens = [s['token'] for s in nusc.sample_data if (s['sensor_modality'] == 'camera') and
                                 s['is_key_frame']]

    if args.only_valid:
        sample_data_camera_tokens_temp = []
        for sample_data_camera_token in sample_data_camera_tokens:
            sample_data = nusc.get("sample_data", sample_data_camera_token)
            scene_token = nusc.get("sample", sample_data['sample_token'])['scene_token']
            scene_name =  nusc.get("scene", scene_token)['name']
            if scene_name in splits.val:
                sample_data_camera_tokens_temp.append(sample_data_camera_token)
        sample_data_camera_tokens = sample_data_camera_tokens_temp
    
    # For debugging purposes: Only produce the first n images.
    if args.image_limit != -1:
        sample_data_camera_tokens = sample_data_camera_tokens[:args.image_limit]

    # Loop through the records and apply the yolo algorithm.
    detections = []
    for token in tqdm(sample_data_camera_tokens):
        detections_records = get_2d_boxes(token, nusc, model, args.dataroot)
        detections.extend(detections_records)

    sample_tokens = {}
    for detection in detections:
        sd_rec = nusc.get('sample_data', detection['sample_data_token'])
        s_rec = nusc.get('sample', sd_rec['sample_token'])
        if detection['category_name'] == 'car':
            if s_rec['token'] in sample_tokens:
                sample_tokens[s_rec['token']].append(detection)
            else:
                sample_tokens[s_rec['token']] = [detection]

    scenes = get_nusc_scenes(nusc)
    sequences = []
    for scene in scenes:
        sequence = []
        for sample_token in scene:
            if sample_token in sample_tokens:
                boxes = sample_tokens[sample_token]
            else:
                boxes = []
            frame = {'frame_token': sample_token, 'boxes': boxes}
            sequence.append(frame)
        if at_least_one_box(sequence):
            sequences.append(sequence)

    detections = {'dataroot': args.dataroot,
                     'version': args.version, 'sequences': sequences}
    logging.info("Saving the 2D re-projections under {}".format(args.output_file))
    # Save to a .json file.
    dest_path = os.path.join(args.dataroot, args.version)
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    with open(args.output_file, 'w') as fh:
        json.dump(detections, fh, indent=4)

    logging.info("Saved the 2D re-projections under {}".format(args.output_file))
    logging.info('end')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='Yolo model to download.')
    parser.add_argument('--output_file', type=str, default='created_files/image_annotations.json', help='Output filename.')
    parser.add_argument('--dataroot', type=str, default='/data/sets/nuscenes/v1.0-trainval/v1.0-trainval', help="Path where nuScenes is saved.")
    parser.add_argument('--version', type=str, default='v1.0-trainval', help='Dataset version.')
    parser.add_argument('--image_limit', type=int, default=-1, help='Number of images to process or -1 to process all.')
    parser.add_argument('--only_valid', type=bool, default=True, help='If only validation set.')
    args = parser.parse_args()
    if (not os.path.isdir(os.path.dirname(args.output_file)) and os.path.dirname(args.output_file) != ''):
        raise Exception('directory name in output_file parameter does not exist')
    if not os.path.basename(args.output_file).endswith('.json'):
        raise Exception('file name in output_file parameter does not end in \'.json\'')
    main(args)