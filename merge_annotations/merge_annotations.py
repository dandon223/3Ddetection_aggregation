import ast
import argparse
import logging
import os
import open3d as o3d # type: ignore
import json
import numpy as np
from tqdm import tqdm #type: ignore
from typing import Dict, List, Tuple, Union, TypedDict
from pyquaternion import Quaternion #type: ignore
from helper_methods.get_iou import get_iou
from helper_methods.open3d_draw import draw_box 

class DictBox(TypedDict):
    translation: List[float]
    size: List[float]
    rotation: List[float]
    detection_name: str
class Frame(TypedDict):
    frame_token: str
    boxes: List[DictBox]

def read_annotation_files(annotation_files: Dict[str, str], number_of_scenes: int) -> Dict[str, List[List[Frame]]]:
    annotations = {}
    for model_name, file_path in annotation_files.items():
        model_annotations = json.load(open(file_path, 'r'))
        if number_of_scenes == -1:
            annotations[model_name] = model_annotations['sequences']
        else:
            annotations[model_name] = model_annotations['sequences'][:number_of_scenes]
    return annotations

def get_sequences_by_model(annotations: Dict[str, List[List[Frame]]], sequence_index: int) -> Dict[str, List[Frame]]:
    sequence_by_model = {}
    for model_name, sequences in annotations.items():
        sequence_by_model[model_name] = sequences[sequence_index]
    return sequence_by_model

def get_frames_by_model(sequences: Dict[str, List[Frame]], frame_index: int) -> Dict[str, Frame]:
    frame_by_model = {}
    for model_name, frames in sequences.items():
        frame_by_model[model_name] = frames[frame_index]
    return frame_by_model

def get_ious_pairs_to_calculate(strategy_list: List[Union[str, float]]) -> List[str]:
    ious_pairs_to_calculate = []
    for strategy in strategy_list:
        if len(strategy) == 1:
            ious_pairs_to_calculate.append(strategy[0] + '+' + strategy[0])
        else:
            for index in range(len(strategy) - 2):
                ious_pairs_to_calculate.append(strategy[0] + '+' + strategy[index + 1])
    return np.unique(ious_pairs_to_calculate).tolist()

def get_ious(frame_model_one: Frame, frame_model_two: Frame) -> List[List[float]]:
    assert frame_model_one['frame_token'] == frame_model_two['frame_token']
    model_one_boxes_ious = []
    for model_one_box in frame_model_one['boxes']:

        model_one_box_ious = []
        for model_two_box in frame_model_two['boxes']:

            model_one_box_ious.append(get_iou(model_one_box, model_two_box))
        model_one_boxes_ious.append(model_one_box_ious)
    return model_one_boxes_ious

def get_ious_by_model(frame_by_model: Dict[str, Frame], ious_pairs_to_calculate: List[str]) -> Dict[str, List[List[float]]]:
    ious_by_model = {}
    for ious_pair in ious_pairs_to_calculate:
        models = ious_pair.split('+')
        if models[0] == models[1]:
            boxes_number = len(frame_by_model[models[0]]['boxes'])
            boxes_ious = []
            for index in range(boxes_number):
                inside_list = [0 for _ in range(boxes_number)]
                inside_list[index] = 1
                boxes_ious.append(inside_list)
            ious_by_model[ious_pair] = boxes_ious
        else:
            ious_by_model[ious_pair] = get_ious(frame_by_model[models[0]], frame_by_model[models[1]])
    return ious_by_model

def merge_all(frame_by_model: Dict[str, Frame]) -> List[DictBox]:
    new_boxes = []
    frame_token = list(frame_by_model.values())[0]['frame_token']
    for model_frame in frame_by_model.values():
        assert model_frame['frame_token'] == frame_token 
        for box in model_frame['boxes']:
            new_boxes.append(box)
    return new_boxes

def get_passed_indices(models_iou_list: List[List[float]], iou_threshold: float) -> Dict[int, List[Tuple[int, float]]]:
    results = {}
    for left_model_index, box_ious in enumerate(models_iou_list):
        temp_list = []
        for right_model_index, iou in enumerate(box_ious):
            if iou >= iou_threshold:
                temp_list.append((right_model_index, iou))
        if len(temp_list) > 0:
            results[left_model_index] = temp_list
    return results

def filter_out_passed_indices(passed_indices: Dict[int, List[Tuple[int, float]]], indices_of_boxes_passed_left: List[int], indices_of_boxes_passed_right: List[int]) -> Dict[int, List[Tuple[int, float]]]:
    passed_indices_filtered = {}
    for left_model_index, right_model_indices in passed_indices.items():
        if left_model_index in indices_of_boxes_passed_left:
            continue
        temp_list = []
        for right_model_index, iou in right_model_indices:
            if right_model_index not in indices_of_boxes_passed_right:
                temp_list.append((right_model_index, iou))
        if len(temp_list) > 0:
            passed_indices_filtered[left_model_index] = temp_list
    return passed_indices_filtered

def flatten(lis: List[Union[int, List[int]]]) -> List[int]:
    lis_temp = []
    for element in lis:
        if type(element) == int:
            lis_temp.append(element)
        elif type(element) == list:
            for elem in element:
                lis_temp.append(elem)
    return lis_temp

def set_intersection_left(indices_of_boxes_strategy_passed_left: Dict[int, List[Tuple[int, float]]], indices_of_boxes_strategy_passed_right: Dict[int, List[Tuple[int, float]]]) -> Tuple[Dict[int, List[Tuple[int, float]]], Dict[int, List[Tuple[int, float]]]]:
    indices_left = [index for index in indices_of_boxes_strategy_passed_left.keys()]
    indices_right = [index for index in indices_of_boxes_strategy_passed_right.keys()]
    indices_intersects = list(set(indices_left).intersection(indices_right))
    indices_of_boxes_strategy_passed_left_new = {}
    indices_of_boxes_strategy_passed_right_new = {}
    for index in indices_intersects:
            indices_of_boxes_strategy_passed_left_new[index] = indices_of_boxes_strategy_passed_left[index]
            indices_of_boxes_strategy_passed_right_new[index] = indices_of_boxes_strategy_passed_right[index]
    return indices_of_boxes_strategy_passed_left_new, indices_of_boxes_strategy_passed_right_new

def filtered_indices_right(indices_intersects: List[int], indices_of_boxes_strategy_passed: Dict[int, List[Tuple[int, float]]]) -> Dict[int, List[Tuple[int, float]]]:
    indices_of_boxes_strategy_passed_new = {}
    for index, values in indices_of_boxes_strategy_passed.items():
        values_temp = []
        for (index_right, iou) in values:
            if index_right in indices_intersects:
                values_temp.append((index_right, iou))
        if len(values_temp) > 0:
            indices_of_boxes_strategy_passed_new[index] = values_temp
    return indices_of_boxes_strategy_passed_new

def set_intersection_right(indices_of_boxes_strategy_passed_left: Dict[int, List[Tuple[int, float]]], indices_of_boxes_strategy_passed_right: Dict[int, List[Tuple[int, float]]]) -> Tuple[Dict[int, List[Tuple[int, float]]], Dict[int, List[Tuple[int, float]]]]:
    indices_left = flatten([[index for (index, _) in values] for values in indices_of_boxes_strategy_passed_left.values()])
    indices_right = flatten([[index for (index, _) in values] for values in indices_of_boxes_strategy_passed_right.values()])
    indices_intersects = list(set(indices_left).intersection(indices_right))
    indices_of_boxes_strategy_passed_left_new = filtered_indices_right(indices_intersects, indices_of_boxes_strategy_passed_left)
    indices_of_boxes_strategy_passed_right_new = filtered_indices_right(indices_intersects, indices_of_boxes_strategy_passed_right)
    return indices_of_boxes_strategy_passed_left_new, indices_of_boxes_strategy_passed_right_new

def set_intersection_left_right(indices_of_boxes_strategy_passed_left: Dict[int, List[Tuple[int, float]]], indices_of_boxes_strategy_passed_right: Dict[int, List[Tuple[int, float]]]) -> Tuple[Dict[int, List[Tuple[int, float]]], Dict[int, List[Tuple[int, float]]]]:
    indices_left = [index for index in indices_of_boxes_strategy_passed_left.keys()]
    indices_right = flatten([[index for (index, _) in values] for values in indices_of_boxes_strategy_passed_right.values()])
    indices_intersects = list(set(indices_left).intersection(indices_right))
    indices_of_boxes_strategy_passed_left_new = {}
    for index in indices_intersects:
            indices_of_boxes_strategy_passed_left_new[index] = indices_of_boxes_strategy_passed_left[index]
    indices_of_boxes_strategy_passed_right_new = filtered_indices_right(indices_intersects, indices_of_boxes_strategy_passed_right)
    return indices_of_boxes_strategy_passed_left_new, indices_of_boxes_strategy_passed_right_new

def merge_boxes_strategy(indices_of_boxes_strategy_passed: Dict[str, Dict[int, List[Tuple[int, float]]]]) -> Dict[str, Dict[int, List[Tuple[int, float]]]]:
    ious_pairs_done = []
    for ious_pair in indices_of_boxes_strategy_passed.keys():
        for ious_pair_two in indices_of_boxes_strategy_passed.keys():
            if ious_pair == ious_pair_two or (ious_pair, ious_pair_two) in ious_pairs_done:
                continue
            ious_pairs_done.append((ious_pair_two, ious_pair))
            if ious_pair.split('+')[0] == ious_pair_two.split('+')[0]:
                indices_of_boxes_strategy_passed[ious_pair], indices_of_boxes_strategy_passed[ious_pair_two] = set_intersection_left(indices_of_boxes_strategy_passed[ious_pair], indices_of_boxes_strategy_passed[ious_pair_two])
            elif ious_pair.split('+')[0] == ious_pair_two.split('+')[1]:
                indices_of_boxes_strategy_passed[ious_pair], indices_of_boxes_strategy_passed[ious_pair_two] = set_intersection_left_right(indices_of_boxes_strategy_passed[ious_pair], indices_of_boxes_strategy_passed[ious_pair_two])
            elif ious_pair.split('+')[1] == ious_pair_two.split('+')[0]:
                indices_of_boxes_strategy_passed[ious_pair_two], indices_of_boxes_strategy_passed[ious_pair] = set_intersection_left_right(indices_of_boxes_strategy_passed[ious_pair_two], indices_of_boxes_strategy_passed[ious_pair])
            elif ious_pair.split('+')[1] == ious_pair_two.split('+')[1]:
                indices_of_boxes_strategy_passed[ious_pair], indices_of_boxes_strategy_passed[ious_pair_two] = set_intersection_right(indices_of_boxes_strategy_passed[ious_pair], indices_of_boxes_strategy_passed[ious_pair_two])
    return indices_of_boxes_strategy_passed

def choose_one_indices_of_boxes(duplicate_indices_of_boxes: List[Tuple[int, ...]], ious_by_model: Dict[str, List[List[float]]], order_of_models: List[str]) -> Tuple[int, ...]:
    ious_list: Dict[str, List[float]] = {}
    for indices_of_boxes in duplicate_indices_of_boxes:
        ious_list[str(indices_of_boxes)] = []
    for models_pair, ious in ious_by_model.items():
        models = models_pair.split('+')
        if models[0] not in order_of_models or models[1] not in order_of_models:
            continue
        for indices_of_boxes in duplicate_indices_of_boxes:
            model_left_index = order_of_models.index(models[0])
            iou_left_index = indices_of_boxes[model_left_index]
            model_right_index = order_of_models.index(models[1])
            iou_right_index = indices_of_boxes[model_right_index]
            iou = ious[iou_left_index][iou_right_index]
            ious_list[str(indices_of_boxes)].append(iou)
    best_iou_average = 0.
    best_indices: Tuple[int] = tuple(ast.literal_eval(list(ious_list.keys())[0]))
    for indices, list_ious in ious_list.items():
        iou_sum = 0.
        for iou in list_ious:
            iou_sum += iou
        iou_average = iou_sum/len(list_ious)
        if  iou_average > best_iou_average:
            best_iou_average = iou_average
            best_indices = tuple(ast.literal_eval(indices))
        
    return best_indices

def delete_duplicates(order_of_models: List[str], duplicates: Dict[str, List[int]], indices_of_boxes_to_merge: List[Tuple[int, ...]], ious_by_model: Dict[str, List[List[float]]], debug) -> List[Tuple[int, ...]]:
    order_to_delete = []
    for model_name in order_of_models:
        for duplicate in duplicates[model_name]:
            order_to_delete.append((model_name,) + duplicate)
    order_to_delete = sorted(order_to_delete, key = lambda x: x[2], reverse=True)

    for model_name, duplicate_index, _ in order_to_delete:
        model_order_index = order_of_models.index(model_name)
        duplicate_indices_of_boxes = []
        for indices_of_boxes in indices_of_boxes_to_merge:
            if indices_of_boxes[model_order_index] == duplicate_index:
                duplicate_indices_of_boxes.append(indices_of_boxes)
        if debug:
            logging.debug('duplicate_indices_of_boxes %s', duplicate_indices_of_boxes)
        if len(duplicate_indices_of_boxes) <=1 :
            continue
        for indices_of_boxes in duplicate_indices_of_boxes:
            indices_of_boxes_to_merge.pop(indices_of_boxes_to_merge.index(indices_of_boxes))
        indices_of_boxes = choose_one_indices_of_boxes(duplicate_indices_of_boxes, ious_by_model, order_of_models)
        indices_of_boxes_to_merge.append(indices_of_boxes)

    return indices_of_boxes_to_merge

def get_boxes_to_merge(indices_of_boxes_strategy_passed: Dict[str, Dict[int, List[Tuple[int, float]]]], ious_by_model: Dict[str, List[List[float]]], debug) -> Tuple[List[str], List[Tuple[int, ...]]]:
    order_of_models: List[str] = []
    indices_of_boxes_to_merge: List[Tuple[int, ...]] = []
    models_pair_to_go_back_to = [models_pair for models_pair in indices_of_boxes_strategy_passed.keys()]
    while len(models_pair_to_go_back_to) > 0:
        for models_pair, models_pair_indices in indices_of_boxes_strategy_passed.items():
            models = models_pair.split('+')

            if len(order_of_models) == 0:
                order_of_models.append(models[0])
                order_of_models.append(models[1])
                for left_model_index_box, values in models_pair_indices.items():
                    for right_model_index_box, _ in values:
                        indices_of_boxes_to_merge.append((left_model_index_box, right_model_index_box))
                models_pair_to_go_back_to.pop(models_pair_to_go_back_to.index(models_pair))

            elif models[0] in order_of_models and models[1] in order_of_models:
                models_pair_to_go_back_to.pop(models_pair_to_go_back_to.index(models_pair))
                continue

            elif models[0] in order_of_models:
                order_of_models.append(models[1])
                models_pair_to_go_back_to.pop(models_pair_to_go_back_to.index(models_pair))
                index_of_order_model = order_of_models.index(models[0])
                indices_of_boxes_to_merge_temp = []
                for left_model_index_box, values in models_pair_indices.items():
                    for indices_boxes_to_merge in indices_of_boxes_to_merge:
                        if indices_boxes_to_merge[index_of_order_model] == left_model_index_box:
                            for right_model_index_box, _ in values:
                                indices_of_boxes_to_merge_temp.append(indices_boxes_to_merge + (right_model_index_box,))
                indices_of_boxes_to_merge = indices_of_boxes_to_merge_temp

            elif models[1] in order_of_models:
                order_of_models.append(models[0])
                models_pair_to_go_back_to.pop(models_pair_to_go_back_to.index(models_pair))
                index_of_order_model = order_of_models.index(models[1])
                indices_of_boxes_to_merge_temp = []
                for left_model_index_box, values in models_pair_indices.items():
                    for right_model_index_box, _ in values:
                        for indices_boxes_to_merge in indices_of_boxes_to_merge:
                            if indices_boxes_to_merge[index_of_order_model] == right_model_index_box:
                                indices_of_boxes_to_merge_temp.append(indices_boxes_to_merge + (left_model_index_box,))
                indices_of_boxes_to_merge = indices_of_boxes_to_merge_temp

    duplicates: Dict[str, List[int]] = {}
    for index, model_name in enumerate(order_of_models):
        duplicates[model_name] = []
        number_of_elements: Dict[int, int] = {}
        for indices_of_boxes in indices_of_boxes_to_merge:
            number_of_elements[indices_of_boxes[index]] = number_of_elements.get(indices_of_boxes[index], 0) + 1
        for key, value in number_of_elements.items():
            if value > 1:
                duplicates[model_name].append((key, value))

    indices_of_boxes_to_merge = delete_duplicates(order_of_models, duplicates, indices_of_boxes_to_merge, ious_by_model, debug)

    return order_of_models, indices_of_boxes_to_merge

def update_boxes_passed(indices_of_boxes_passed: Dict[str, List[int]], order_of_methods: List[str], indices_of_boxes_to_merge: List[Tuple[int, ...]]) -> Dict[str, List[int]]:
    if len(order_of_methods) == 2 and order_of_methods[0] == order_of_methods[1]:
        for indices_tuple in indices_of_boxes_to_merge:
            assert indices_tuple[0] not in indices_of_boxes_passed[order_of_methods[0]]
            indices_of_boxes_passed[order_of_methods[0]].append(indices_tuple[0])
        return indices_of_boxes_passed

    for method_index, method_name in enumerate(order_of_methods):
        for indices_tuple in indices_of_boxes_to_merge:
            assert indices_tuple[method_index] not in indices_of_boxes_passed[method_name]
            indices_of_boxes_passed[method_name].append(indices_tuple[method_index])
    return indices_of_boxes_passed

def format_rotation(rotation: Quaternion) -> float:
    yaw = rotation.yaw_pitch_roll[0]
    while yaw < 0:
        yaw = yaw + np.pi
    while yaw >= np.pi:
        yaw = yaw - np.pi
    return yaw

def get_rotation_difference(i: float, rotation_element: float) -> float:
    difference1 = abs(i - rotation_element) % np.pi
    if i < rotation_element:
        temp = rotation_element - np.pi
        difference2 = abs(i - temp) % np.pi
    else:
        temp = i - np.pi
        difference2 = abs(temp - rotation_element) % np.pi
    if difference1 <= difference2:
        return difference1
    else:
        return difference2
    
def average_rotation(rotation_list_temp: List[float]) -> float:
    seq = np.arange(0, 180 * np.pi / 180, 0.5 * np.pi / 180)
    rotation_best = 0
    min_best_distance = np.pi
    for i in seq:
        distance = []
        for rotation_element in rotation_list_temp:
            distance.append(get_rotation_difference(i, rotation_element))
        average_rotation_difference = max(distance)
        if average_rotation_difference < min_best_distance:
            min_best_distance = average_rotation_difference
            rotation_best = i
    return rotation_best

def average_box(boxes_temp: List[DictBox]) -> DictBox:
    translation_temp = np.array([0., 0., 0.])
    size_temp = np.array([0., 0., 0.])
    rotation_list_temp = []
    detection_score = 0
    for box in boxes_temp:
        translation_temp += box['translation']
        size_temp += box['size']
        rotation_list_temp.append(format_rotation(Quaternion(box['rotation'])))
        detection_score += box['detection_score']
    rotation_temp = average_rotation(rotation_list_temp)
    rotation = Quaternion(axis=[0, 0, 1], radians=rotation_temp)
    rotation = [rotation[0], rotation[1], rotation[2], rotation[3]]
    return {'translation':list(translation_temp/len(boxes_temp)), 'size': list(size_temp/len(boxes_temp)), 'rotation':rotation, 'detection_name': boxes_temp[0]['detection_name'], 'detection_score': detection_score/len(boxes_temp)}

def merge_boxes(indices_of_boxes_to_merge: List[Tuple[int, ...]], order_of_methods: List[str], frame_by_model: Dict[str, Frame], visualize) -> List[DictBox]:
    new_boxes = []
    if len(order_of_methods) == 2 and order_of_methods[0] == order_of_methods[1]:
        method_name = order_of_methods[0]
        for indices_of_boxes in indices_of_boxes_to_merge:
            new_boxes.append(frame_by_model[method_name]['boxes'][indices_of_boxes[0]])
        return new_boxes

    for indices_of_boxes in indices_of_boxes_to_merge:
        if visualize:
            vis = o3d.visualization.Visualizer()
            vis.create_window()
            vis.get_render_option().point_size = 5.0
        boxes_temp = []
        for index, method_name in enumerate(order_of_methods):
            boxes_temp.append(frame_by_model[method_name]['boxes'][indices_of_boxes[index]])
        avg_box = average_box(boxes_temp)
        new_boxes.append(avg_box)
        if visualize:
            for box in boxes_temp:
                vis = draw_box(vis, [box], [[1, 0, 0]])
            vis = draw_box(vis, [avg_box], [[0, 1, 0]])
            vis.run()
            vis.destroy_window()
    return new_boxes

def merge_with_strategy(strategy_list: List[Dict[str, float]], ious_by_model: Dict[str, List[List[float]]], frame_by_model: Dict[str, Frame], debug, visualize) -> List[DictBox]:
    merged_boxes: List[DictBox] = []
    indices_of_boxes_passed: Dict[str, List[int]] = {}
    for key in ious_by_model.keys():
        models = key.split('+')
        indices_of_boxes_passed[models[0]] = []
        indices_of_boxes_passed[models[1]] = []

    for strategy in strategy_list:
        indices_of_boxes_strategy_passed = {}
        if debug:
            logging.debug('-----strategy start-----')

        new_strategy = {}
        if len(strategy) == 1:
            new_strategy[strategy[0] + '+' + strategy[0]] = 0.99
        else:
            for index in range(len(strategy) - 2):
                new_strategy[strategy[0] + '+' + strategy[index + 1]] = strategy[-1]

        for ious_pair, iou_threshold in new_strategy.items():
            passed_indices = get_passed_indices(ious_by_model[ious_pair], iou_threshold)
            indices_of_boxes_strategy_passed[ious_pair] = filter_out_passed_indices(passed_indices, indices_of_boxes_passed[ious_pair.split('+')[0]], indices_of_boxes_passed[ious_pair.split('+')[1]])
            if debug:
                logging.debug('ious_pair %s, iou_threshold %s', ious_pair, iou_threshold)
                logging.debug('indices_of_boxes_strategy_passed[ious_pair] %s',indices_of_boxes_strategy_passed[ious_pair])
                logging.debug('--------------------------')

        indices_of_boxes_strategy_passed = merge_boxes_strategy(indices_of_boxes_strategy_passed)
        if debug:
            logging.debug('xxxxxxxxxxxxxxxxxxxxxxxxxx')
        order_of_methods, indices_of_boxes_to_merge = get_boxes_to_merge(indices_of_boxes_strategy_passed, ious_by_model, debug)
        if debug:
            logging.debug('order_of_methods %s', order_of_methods)
            logging.debug('indices_of_boxes_to_merge %s', indices_of_boxes_to_merge)
        indices_of_boxes_passed = update_boxes_passed(indices_of_boxes_passed, order_of_methods, indices_of_boxes_to_merge)
        if debug:
            logging.debug('indices_of_boxes_passed %s', indices_of_boxes_passed)
        merged_boxes.extend(merge_boxes(indices_of_boxes_to_merge, order_of_methods, frame_by_model, visualize))
        if debug:
            logging.debug('-----strategy end-----')
    if debug:
        logging.debug('===============================')
    return merged_boxes

def get_output_file(annotations, config, args) -> Dict:
    output_file = {"dataroot": "/data/sets/nuscenes/v1.0-trainval/v1.0-trainval", "version": "v1.0-trainval",'sequences': []}
    sequences_old = list(annotations.values())[0]
    ious_pairs_to_calculate = get_ious_pairs_to_calculate(config['strategy'])
    for sequence_index, sequence in enumerate(tqdm(sequences_old)):
        new_sequence = []
        sequence_by_model = get_sequences_by_model(annotations, sequence_index)
        for frame_index, frame in enumerate(sequence):
            frame_by_model = get_frames_by_model(sequence_by_model, frame_index)

            if args.merge_all:
                new_boxes = merge_all(frame_by_model)
            else:
                ious_by_model = get_ious_by_model(frame_by_model, ious_pairs_to_calculate)
                new_boxes = merge_with_strategy(config['strategy'], ious_by_model, frame_by_model, args.debug, args.vis)
            new_frame = {'frame_token': frame['frame_token'], 'boxes': new_boxes}
            new_sequence.append(new_frame)
        output_file['sequences'].append(new_sequence)
    return output_file

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config_file', type=str,
                        default='merge_annotations/config.json', help='Input config file.')
    parser.add_argument('--merge_all', action='store_true')
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--number_of_scenes', type=int, default=-1)
    args = parser.parse_args()
    logging.basicConfig(filename='merge_annotations.log', level=logging.DEBUG)
    config = json.load(open(args.config_file, 'r'))
    if (not os.path.isdir(os.path.dirname(config['output_file'])) and os.path.dirname(config['output_file']) != ''):
        raise Exception('directory name in output_file parameter does not exist')
    if not os.path.basename(config['output_file']).endswith('.json'):
        raise Exception('file name in output_file parameter does not end in \'.json\'')

    annotations = read_annotation_files(config['annotation_files'], args.number_of_scenes)

    for sequences in annotations.values():
        for sequence in sequences:
            for frame in sequence:
                for box in frame['boxes']:
                    assert box['size'][1] >= box['size'][0]

    output_file = get_output_file(annotations, config, args)
    with open(config['output_file'], 'w') as fh:
        json.dump(output_file, fh, indent=4)
    
if __name__ == '__main__':
    main()
