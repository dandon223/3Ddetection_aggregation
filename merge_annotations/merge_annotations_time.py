import argparse
import logging
import os
import time
import json
import pandas as pd
from tqdm import tqdm #type: ignore
from typing import Dict, List, TypedDict
import numpy as np

from merge_annotations import read_annotation_files, get_output_file, get_ious_pairs_to_calculate, get_sequences_by_model, get_frames_by_model

class DictBox(TypedDict):
    translation: List[float]
    size: List[float]
    rotation: List[float]
    detection_name: str
class Frame(TypedDict):
    frame_token: str
    boxes: List[DictBox]

def get_ious(frame_model_one: Frame, frame_model_two: Frame) -> List[List[float]]:
    number_of_ious_counting = 0
    assert frame_model_one['frame_token'] == frame_model_two['frame_token']
    model_one_boxes_ious = []
    for model_one_box in frame_model_one['boxes']:

        model_one_box_ious = []
        for model_two_box in frame_model_two['boxes']:

            number_of_ious_counting += 1
        model_one_boxes_ious.append(model_one_box_ious)
    return number_of_ious_counting

def get_ious_by_model(frame_by_model: Dict[str, Frame], ious_pairs_to_calculate: List[str]) -> Dict[str, List[List[float]]]:
    ious_by_model = {}
    number_of_ious_counting = 0
    for ious_pair in ious_pairs_to_calculate:
        models = ious_pair.split('+')
        if models[0] == models[1]:
            raise Exception('not allowed comparing left and right same model')
        else:
            number_of_ious_counting += get_ious(frame_by_model[models[0]], frame_by_model[models[1]])
    return number_of_ious_counting

def get_number_of_ious_counting(annotations, config):
    number_of_ious_counting = 0
    ious_pairs_to_calculate = get_ious_pairs_to_calculate(config['strategy'])
    sequences_old = list(annotations.values())[0]
    for sequence_index, sequence in enumerate(tqdm(sequences_old)):
        sequence_by_model = get_sequences_by_model(annotations, sequence_index)
        for frame_index, _ in enumerate(sequence):
            frame_by_model = get_frames_by_model(sequence_by_model, frame_index)
            ious_by_model = get_ious_by_model(frame_by_model, ious_pairs_to_calculate)
            number_of_ious_counting += ious_by_model
    return number_of_ious_counting

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config_file', type=str, default='merge_annotations/config.json', help='Input config file.')
    parser.add_argument('--merge_all', action='store_true')
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--repeat', type=int, default=5)
    args = parser.parse_args()
    logging.basicConfig(filename='merge_annotations.log', level=logging.DEBUG)
    config = json.load(open(args.config_file, 'r'))
    if (not os.path.isdir(os.path.dirname(config['output_file'])) and os.path.dirname(config['output_file']) != ''):
        raise Exception('directory name in output_file parameter does not exist')
    if not os.path.basename(config['output_file']).endswith('.json'):
        raise Exception('file name in output_file parameter does not end in \'.json\'')
    

    data = {"strategia":[],
            "liczba scenariuszy treningowych":[],
            "liczba scenariuszy":[],
            "liczba porownan iou":[],
            "liczba prob":[],
            "czas [s]": [],
            "odchylenie standardowe": []}
    number_of_scenes_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 40, 80, 150]
    for number_of_scenes in number_of_scenes_list:
        annotations = read_annotation_files(config['annotation_files'], number_of_scenes)
        number_of_ious_counting = get_number_of_ious_counting(annotations, config)
        times = []
        print(f'Number of ious counting: {number_of_ious_counting}')
        for i in range(args.repeat):
            annotations = read_annotation_files(config['annotation_files'], number_of_scenes)

            start = time.time()
            get_output_file(annotations, config, args)
            end = time.time()

            print(f'Scenes: {number_of_scenes}, Number {i+1}, Time: {end-start}')
            times.append(end-start)
    
        data["strategia"].append(config["strategy"])
        data["liczba scenariuszy treningowych"].append(700)
        data["liczba scenariuszy"].append(number_of_scenes)
        data["liczba porownan iou"].append(number_of_ious_counting)
        data["liczba prob"].append(args.repeat)
        data["czas [s]"].append(np.mean(times))
        data["odchylenie standardowe"].append(np.std(times))
    
    pd.DataFrame(data).to_csv('czas_adnotacji_trzy_zasady.csv', index=False) 

if __name__ == '__main__':
    main()