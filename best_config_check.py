import argparse
import os
import logging
import json
from typing import Dict, List, Union
from tqdm import tqdm
from nuscenes import NuScenes # type: ignore
from nuscenes.eval.detection.data_classes import DetectionConfig # type: ignore

from merge_annotations.merge_annotations import read_annotation_files, get_ious_pairs_to_calculate,\
get_sequences_by_model, get_frames_by_model, merge_with_strategy, get_ious_by_model

from evaluate_map.main import map_detections, _check_if_same_frame_order
from evaluate_map.create_ground_truth import create_ground_truth
from evaluate_map.read_annotations import read_annotations_nuscenes
from evaluate_map.map_metric import evaluate_map_metric

from best_config_merge import get_all_strategies


logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.DEBUG,
    datefmt='%Y-%m-%d %H:%M:%S'
)

map_params = {
        "dataroot": "/data/sets/nuscenes/v1.0-trainval/v1.0-trainval",
        "version": "v1.0-trainval",
        "use_camera": "False",
        "use_lidar": "True",
        "use_radar": "True",
        "use_map": "False",
        "use_external": "False",
        "detection_config_file": "evaluate_map/configs/detection_cvpr_2019.json",
        "classes": ["car"]
    }

merge_config = {
    "annotation_files":{
        "TED": "outputs/ted/7scenerios/nuscenes_output.json",
        "SPHEREFORMER": "outputs/sphereformer/7scenerios/nuscenes_output.json",
        "FGR": "outputs/fgr_yolo3/image_annotations.json"
    },
    "strategy":[]
    }

def get_all_strategies_best(strategies_checked, how_many):
    best_strategies: List[List[Union[str, float]]] = []
    aps = []
    strategies = []
    for strategy, ap in strategies_checked.items():
        strategies.append(strategy)
        aps.append(ap)
    
    for _ in range(how_many):
        max_index = aps.index(max(aps))
        best_strategies.append(strategies[max_index])
        strategies.pop(max_index)
        aps.pop(max_index)

    all_strategies = get_all_strategies()
    all_strategies_best = []
    for strategy in all_strategies:
        if str(strategy) in best_strategies:
            all_strategies_best.append(strategy)
    return all_strategies_best

def get_output_file(annotations, all_ious_by_model, config, args) -> Dict:
    output_file = {"dataroot": "/data/sets/nuscenes/v1.0-trainval/v1.0-trainval", "version": "v1.0-trainval",'sequences': []}
    sequences_old = list(annotations.values())[0]
    for sequence_index, sequence in enumerate(sequences_old):
        new_sequence = []
        sequence_by_model = get_sequences_by_model(annotations, sequence_index)
        for frame_index, frame in enumerate(sequence):
            frame_by_model = get_frames_by_model(sequence_by_model, frame_index)

            ious_by_model = all_ious_by_model[sequence_index][frame_index]
            new_boxes = merge_with_strategy(config['strategy'], ious_by_model, frame_by_model, args.debug, args.vis)

            new_frame = {'frame_token': frame['frame_token'], 'boxes': new_boxes}
            new_sequence.append(new_frame)
        output_file['sequences'].append(new_sequence)
    return output_file

def main():

    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument('--merge_all', action='store_true')
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--number_of_scenes', type=int, default=-1)
    parser.add_argument('--input_file', type=str)
    parser.add_argument('--how_many', type=int, default=10)
    args = parser.parse_args()

    with open(args.input_file, 'r', encoding='UTF-8') as file_object:
        json_content = file_object.read()
        strategies_checked = json.loads(json_content)
    
    strategy_list = get_all_strategies_best(strategies_checked, args.how_many)

    print('read NuScenes database')
    nusc = NuScenes(version='v1.0-trainval', dataroot='/data/sets/nuscenes/v1.0-trainval/v1.0-trainval', verbose=False)
    print('create_ground_truth')
    with open(merge_config['annotation_files']['TED'], 'r', encoding='UTF-8') as file_object:
        json_content = file_object.read()
        detections = json.loads(json_content)
        if args.number_of_scenes != -1:
            detections['sequences'] = detections['sequences'][:args.number_of_scenes]
        dt_json = map_detections(detections)
        gt_json = create_ground_truth(dt_json, nusc, ['car'])

    print('get ious pairs')
    annotations = read_annotation_files(merge_config['annotation_files'], args.number_of_scenes)
    sequences_old = list(annotations.values())[0]
    ious_pairs_to_calculate = get_ious_pairs_to_calculate([["TED", "FGR", "SPHEREFORMER", 1], ["SPHEREFORMER", "FGR", 0.2], ["TED"], ["SPHEREFORMER"], ["FGR"]])
    all_ious_by_model = []
    for sequence_index, sequence in enumerate(tqdm(sequences_old)):
        all_ious_by_model.append([])
        sequence_by_model = get_sequences_by_model(annotations, sequence_index)
        for frame_index, frame in enumerate(sequence):
            frame_by_model = get_frames_by_model(sequence_by_model, frame_index)

            ious_by_model = get_ious_by_model(frame_by_model, ious_pairs_to_calculate)
            all_ious_by_model[-1].append(ious_by_model)
    print('get ious pairs finished')
    new_strategy_results = {}
    for strategy in tqdm(strategy_list):

        merge_config['strategy'] = strategy
        output_file = get_output_file(annotations, all_ious_by_model, merge_config, args)
        dt_json = map_detections(output_file)
        _check_if_same_frame_order(gt_json, dt_json)
        with open(map_params['detection_config_file'], 'r', encoding='UTF-8') as file:
            cfg = DetectionConfig.deserialize(json.load(file))
        gt_annos_map, dt_annos_map, meta = read_annotations_nuscenes(gt_json, dt_json, cfg, map_params)
        map_metrics = evaluate_map_metric(nusc, cfg, dt_annos_map, gt_annos_map, meta)
        car_ap = map_metrics['car']['AP']
        new_strategy_results[str(strategy)] = car_ap
        print(f'{car_ap}, {str(strategy)}')
    
    best_strategy = None
    best_ap = 0
    for strategy, ap in new_strategy_results.items():
        if ap > best_ap:
            best_strategy = strategy
            best_ap = ap
    print('best strategy = ', best_strategy)
    print('ap of strategy= ', best_ap)
    return

if __name__ == '__main__':
    main()