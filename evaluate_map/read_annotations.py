"""Module to read annotations to CAR and NuScenes formats"""
from typing import Dict, Tuple
from nuscenes.eval.common.data_classes import EvalBoxes # type: ignore
from nuscenes.eval.detection.data_classes import DetectionConfig, DetectionBox # type: ignore

def _format_annotations_nuscenes(detection: Dict, frame_token: str) -> Dict:
    new_detection = {}
    new_detection['sample_token'] = frame_token
    new_detection['translation'] = detection['center']
    new_detection['size'] = detection['size']
    new_detection['rotation'] = detection['orientation']
    new_detection['detection_name'] = detection['name']
    new_detection['detection_score'] = detection['score']
    new_detection['attribute_name'] = detection.get('attribute_name', '')
    new_detection['velocity'] = detection.get('velocity', [0, 0])
    new_detection['num_pts'] = detection.get('num_pts', -1)
    return new_detection

def _load_prediction(data: Dict, max_boxes_per_sample: int, box_cls) -> Tuple[EvalBoxes, Dict]:
    assert 'results' in data, 'Error: No field `results` in result file. Please note that the result format changed.' \
                              'See https://www.nuscenes.org/object-detection for more information.'

    # Deserialize results and get meta data.
    all_results = EvalBoxes.deserialize(data['results'], box_cls)
    meta = data['meta']

    # Check that each sample has no more than x predicted boxes.
    for sample_token in all_results.sample_tokens:
        assert len(all_results.boxes[sample_token]) <= max_boxes_per_sample, \
            f'Error: Only <= {max_boxes_per_sample} boxes per sample allowed!'

    return all_results, meta

def read_annotations_nuscenes(gt_json: Dict[str, Dict],
                                dt_json: Dict[str, Dict],
                                cfg: DetectionConfig,
                                map_params: Dict) -> Tuple[EvalBoxes, EvalBoxes, Dict]:
    """Creates new file with detections from DARTS to NuScenes format

    :param gt_json: ground truth json from file
    :param dt_json: detections json from file
    :param map_params: detector parameters needed for NuScenes evaluation
    :return: detections and ground truth in NuScenes format
    """
    meta = {}
    meta['use_camera'] = map_params['use_camera']
    meta['use_lidar'] = map_params['use_lidar']
    meta['use_radar'] = map_params['use_radar']
    meta['use_map'] = map_params['use_map']
    meta['use_external'] = map_params['use_external']
    gt_annos: Dict[str, Dict] = {'results':{}, 'meta':meta}
    dt_annos: Dict[str, Dict] = {'results':{}, 'meta':meta}

    for sequence in gt_json['sequences']:
        for frame in sequence:
            frame_detections_in_nuscenes_format = []
            frame_token = frame['sample_token']
            for detection in frame['boxes']:
                frame_detections_in_nuscenes_format.append(_format_annotations_nuscenes(detection, frame_token))
            gt_annos['results'][frame_token] = frame_detections_in_nuscenes_format

    for sequence in dt_json['sequences']:
        for frame in sequence:
            frame_detections_in_nuscenes_format = []
            frame_token = frame['sample_token']
            for detection in frame['boxes']:
                frame_detections_in_nuscenes_format.append(_format_annotations_nuscenes(detection, frame_token))
            dt_annos['results'][frame_token] = frame_detections_in_nuscenes_format

    dt_annos, meta = _load_prediction(dt_annos, cfg.max_boxes_per_sample, DetectionBox)
    gt_annos, _ = _load_prediction(gt_annos, cfg.max_boxes_per_sample, DetectionBox)

    return gt_annos, dt_annos, meta
