"""Generate ground truth from detection file in DARTS format"""
import logging
from typing import Dict, List
from nuscenes.nuscenes import NuScenes  # type: ignore
MATCH_CLASSES = {'vehicle.car': 'car'}

def _match_classes(class_str: str) -> str:
    if class_str in MATCH_CLASSES:
        return MATCH_CLASSES[class_str]
    return class_str

def create_ground_truth(detections: Dict, nusc: NuScenes, classes: List[str]) -> Dict:
    new_sequences = []
    total_sequences = 0
    total_samples = 0
    attribute_map = {a['token']: a['name'] for a in nusc.attribute}
    for sequence in detections['sequences']:
        new_sequence = []
        total_sequences = total_sequences + 1
        for frame in sequence:
            total_samples = total_samples + 1
            sample = nusc.get('sample', frame['sample_token'])
            annotations = sample['anns']
            boxes = []
            for ground_truth_token in annotations:
                ground_truth_metadata = nusc.get('sample_annotation', ground_truth_token)
                instance_metadata = nusc.get('instance', ground_truth_metadata['instance_token'])
                category_metadata = nusc.get('category', instance_metadata['category_token'])
                if _match_classes(category_metadata['name']) not in classes:
                    continue
                # Get attribute_name.
                attr_tokens = ground_truth_metadata['attribute_tokens']
                attr_count = len(attr_tokens)
                if attr_count == 0:
                    attribute_name = ''
                elif attr_count == 1:
                    attribute_name = attribute_map[attr_tokens[0]]
                else:
                    raise Exception('Error: GT annotations must not have more than one attribute!') # pylint: disable=broad-exception-raised
                box = {}
                box['center'] = ground_truth_metadata['translation']
                box['size'] = ground_truth_metadata['size']
                box['orientation'] = ground_truth_metadata['rotation']
                box['name'] = _match_classes(category_metadata['name'])
                box['score'] = -1
                box['attribute_name'] = attribute_name
                box['velocity'] = list(nusc.box_velocity(ground_truth_metadata['token'])[:2])
                box['num_pts'] = ground_truth_metadata['num_lidar_pts'] + ground_truth_metadata['num_radar_pts']
                boxes.append(box)
            new_frame = {'sample_token': frame['sample_token'], 'boxes': boxes}
            new_sequence.append(new_frame)
        new_sequences.append(new_sequence)
    ground_truth = {'sequences': new_sequences}
    logging.info('get_ground_truth: seq %s, frames %s', total_sequences, total_samples)
    return ground_truth
