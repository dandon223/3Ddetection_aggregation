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
        "TED": "outputs/ted/700scenerios/nuscenes_output.json",
        "SPHEREFORMER": "outputs/sphereformer/700scenerios/nuscenes_output.json",
        "FGR": "outputs/fgr_yolo3/image_annotations.json"
    },
    "strategy":[]
    }
def get_all_strategies():
    all_strategies: List[List[Union[str, float]]] = []

    new_strategies = get_strategies(0, 1, 2)
    for new_strategy in new_strategies:
        if new_strategy not in all_strategies:
            all_strategies.append(new_strategy)
    
    new_strategies = get_strategies(0, 2, 1)
    for new_strategy in new_strategies:
        if new_strategy not in all_strategies:
            all_strategies.append(new_strategy)
    
    new_strategies = get_strategies(1, 0, 2)
    for new_strategy in new_strategies:
        if new_strategy not in all_strategies:
            all_strategies.append(new_strategy)
    
    new_strategies = get_strategies(1, 2, 0)
    for new_strategy in new_strategies:
        if new_strategy not in all_strategies:
            all_strategies.append(new_strategy)
    
    new_strategies = get_strategies(2, 1, 0)
    for new_strategy in new_strategies:
        if new_strategy not in all_strategies:
            all_strategies.append(new_strategy)
    
    new_strategies = get_strategies(2, 0, 1)
    for new_strategy in new_strategies:
        if new_strategy not in all_strategies:
            all_strategies.append(new_strategy)
    
    
    return all_strategies

def get_strategies(index_1, index_2, index_3) -> List[List[Union[str, float]]]:
    all_strategies: List[List[Union[str, float]]] = []
    all_ted_sphereformer_fgr = [["TED", "FGR", "SPHEREFORMER", 0.2], ["TED", "FGR", "SPHEREFORMER", 0.5], ["TED", "FGR", "SPHEREFORMER", 0.7], ["TED", "FGR", "SPHEREFORMER", 1.0]]
    all_ted_sphereformer = [["TED", "SPHEREFORMER", 0.2], ["TED", "SPHEREFORMER", 0.5], ["TED", "SPHEREFORMER", 0.7], ["TED", "SPHEREFORMER", 1]]
    all_ted_fgr = [["TED", "FGR", 0.2], ["TED", "FGR", 0.5], ["TED", "FGR", 0.7], ["TED", "FGR", 1]]
    all_sphereformer_fgr = [["SPHEREFORMER", "FGR", 0.2], ["SPHEREFORMER", "FGR", 0.5], ["SPHEREFORMER", "FGR", 0.7], ["SPHEREFORMER", "FGR", 1]]
    all_ted = [["TED"], [1]]
    all_sphereformer = [["SPHEREFORMER"], [1]]
    all_fgr = [["FGR"], [1]]

    for ted_sphereformer_fgr in all_ted_sphereformer_fgr:
        if ted_sphereformer_fgr[-1] == 1:
            all_strategies.append([[]])
        else:
            all_strategies.append([ted_sphereformer_fgr])
    
    rest_strategies = [all_ted_sphereformer, all_ted_fgr, all_sphereformer_fgr, all_ted, all_sphereformer, all_fgr]
    assert len(set([index_1, index_2, index_3])) == 3
    assert index_1 < 3
    assert index_2 < 3
    assert index_3 < 3
    rest_strategies = [rest_strategies[index_1], rest_strategies[index_2], rest_strategies[index_3], rest_strategies[3], rest_strategies[4], rest_strategies[5]]
    for strategy_type in rest_strategies:
        new_strategies = []
        for strategy in strategy_type:
            for strategies in all_strategies:
                if strategy[-1] == 1:
                    new_strategies.append(strategies)
                else:
                    temp = strategies.copy()
                    temp.append(strategy)
                    new_strategies.append(temp)
        all_strategies = new_strategies

    new_strategies = []
    for strategy in all_strategies:
        new_strategy = []
        for strategy_part in strategy:
            if len(strategy_part) > 0:
                new_strategy.append(strategy_part)
        if len(new_strategy) > 0:
            new_strategies.append(new_strategy)
    all_strategies = new_strategies
    
    new_strategies = []
    for strategy in all_strategies:
        if strategy == [['TED']] or strategy == [['SPHEREFORMER']] or strategy == [['FGR']] or strategy == [['TED'], ['SPHEREFORMER']] or strategy == [['SPHEREFORMER'], ['FGR']] or strategy == [['TED'], ['FGR']] or strategy == [['TED'], ['SPHEREFORMER'], ['FGR']]:
            continue
        else:
            new_strategies.append(strategy)
    all_strategies = new_strategies

    test_if_every_strategy = {}
    for strategy in all_strategies:
        assert str(strategy) not in test_if_every_strategy
        test_if_every_strategy[str(strategy)] = 0
    
    return all_strategies

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
    args = parser.parse_args()

    strategy_list = get_all_strategies()
    print('strategy_list', len(strategy_list))
    file_to_save_results = "best_config_search.json"

    if not os.path.isfile(file_to_save_results):
        with open(file_to_save_results, "w") as outfile:
            json_object = json.dumps({}, indent=4)
            outfile.write(json_object)
        strategies_checked = {}
    else:
        with open(file_to_save_results, 'r', encoding='UTF-8') as file_object:
            json_content = file_object.read()
            strategies_checked = json.loads(json_content)
    
    if len(strategy_list) == len(strategies_checked.keys()):
        best_strategy = None
        best_ap = 0
        for strategy, ap in strategies_checked.items():
            if ap > best_ap:
                best_strategy = strategy
                best_ap = ap
        print('best strategy = ', best_strategy)
        print('ap of strategy= ', best_ap)
        return
    
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

    for strategy in tqdm(strategy_list):
        if str(strategy) in strategies_checked:
            continue
        merge_config['strategy'] = strategy
        output_file = get_output_file(annotations, all_ious_by_model, merge_config, args)
        dt_json = map_detections(output_file)
        _check_if_same_frame_order(gt_json, dt_json)
        with open(map_params['detection_config_file'], 'r', encoding='UTF-8') as file:
            cfg = DetectionConfig.deserialize(json.load(file))
        gt_annos_map, dt_annos_map, meta = read_annotations_nuscenes(gt_json, dt_json, cfg, map_params)
        map_metrics = evaluate_map_metric(nusc, cfg, dt_annos_map, gt_annos_map, meta)
        car_ap = map_metrics['car']['AP']
        strategies_checked[str(strategy)] = car_ap
        with open(file_to_save_results, "w") as outfile:
            json_object = json.dumps(strategies_checked, indent=4)
            outfile.write(json_object)

if __name__ == '__main__':
    main()