"""main"""
import argparse
import os
import json
import logging
import jsonschema # type: ignore
from typing import Dict

from map_metric import evaluate_map_metric
from read_annotations import read_annotations_nuscenes
from file_schemas import ground_truth_schema, generated_annotations_schema
from create_ground_truth import create_ground_truth

from nuscenes import NuScenes # type: ignore
from nuscenes.eval.detection.data_classes import DetectionConfig # type: ignore

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.DEBUG,
    datefmt='%Y-%m-%d %H:%M:%S'
)

MATCH_CLASSES = {'vehicle.car': 'car'}

def _match_classes(class_str: str) -> str:
    if class_str in MATCH_CLASSES:
        return MATCH_CLASSES[class_str]
    return class_str

def _read_file(file_path: str) -> Dict[str, Dict]:
    if not os.path.isfile(file_path):
        logging.error('File %s does not exist', file_path)
        raise IOError(f'File {file_path} does not exist')
    with open(file_path, 'r', encoding='UTF-8') as file_object:
        json_content = file_object.read()
        return json.loads(json_content)

def _check_if_same_frame_order(gt_json, dt_json):
    assert len(dt_json['sequences']) == len(gt_json['sequences'])
    for sequence_id, dt_sequence in enumerate(dt_json['sequences']):
        gt_sequence = gt_json['sequences'][sequence_id]
        assert len(dt_sequence) == len(gt_sequence), f'Different length of sequence {sequence_id}.'
        for frame_index, dt_frame in enumerate(dt_sequence):
            assert dt_frame['sample_token'] == gt_sequence[frame_index]['sample_token'],\
                f'Different frame_token for detection frame {dt_frame["sample_token"]}.'

def map_detections(dt_json: Dict):
    new_sequences = []
    for sequence in dt_json['sequences']:
        new_sequence = []
        for frame in sequence:
            new_boxes = []
            for box in frame['boxes']:
                new_box = {}
                new_box['center'] = box['translation']
                new_box['size'] = box['size']
                new_box['orientation'] = box['rotation']
                new_box['name'] = _match_classes(box['detection_name'])
                new_box['score'] = box['detection_score']
                new_boxes.append(new_box)
            new_frame = {'sample_token': frame['frame_token'], 'boxes': new_boxes}
            new_sequence.append(new_frame)
        new_sequences.append(new_sequence)
    new_dt_json = {'sequences': new_sequences}
    return new_dt_json

def main() -> Dict[str, Dict]:
    """Runs the evaluation for our custom correction metric and NuScenes mAP.
    The results are printed to standard output.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='configs/sample_config.json',
                        help='Path to config for the evaluation')
    args = parser.parse_args()
    config = _read_file(args.config_path)
    map_params = config['map_params']
    nusc = NuScenes(version=map_params['version'], dataroot=map_params['dataroot'], verbose=False)
    dt_json = _read_file(config['annotation_reading_params']['dt_path'])
    dt_json = map_detections(dt_json)
    gt_json = create_ground_truth(dt_json, nusc, map_params['classes'])
    try:
        jsonschema.validate(gt_json, ground_truth_schema, format_checker=jsonschema.FormatChecker())
        jsonschema.validate(dt_json, generated_annotations_schema, format_checker=jsonschema.FormatChecker())
        _check_if_same_frame_order(gt_json, dt_json)
    except Exception as e:
        logging.error(e)
        raise e

    result_metric: Dict[str, Dict] = {'all': {}}
    for cls in config['map_params']['classes']:
        result_metric[cls] = {}

    # mAP metrics
    with open(config['map_params']['detection_config_file'], 'r', encoding='UTF-8') as file:
        cfg = DetectionConfig.deserialize(json.load(file))
    gt_annos_map, dt_annos_map, meta = read_annotations_nuscenes(gt_json, dt_json, cfg, map_params)

    map_metrics = evaluate_map_metric(nusc, cfg, dt_annos_map, gt_annos_map, meta)
    ap_sum = 0
    for cls in config['map_params']['classes']:
        ap_sum += map_metrics[cls]['AP']
        result_metric[cls]['AP'] = map_metrics[cls]['AP']
    result_metric['all']['mAP'] = ap_sum / len(config['map_params']['classes'])

    logging.info(json.dumps(result_metric, indent=4))
    return result_metric

if __name__ == '__main__':
    main()
