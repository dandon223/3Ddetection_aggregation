import argparse
import json
from logging import Logger
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple, Union
import torch
from nuscenes.nuscenes import NuScenes #type: ignore
from easydict import EasyDict  # type: ignore
from pcdet.config import cfg, cfg_from_yaml_file  # type: ignore
from pcdet.models import build_network, load_data_to_gpu  # type: ignore
from pcdet.models.detectors import Detector3DTemplate  # type: ignore
from pcdet.utils import common_utils  # type: ignore

from dataset_predict import (
    DATASETS,
    Dataset,
    InvalidPointCloudException,
)


def parse_config() -> Tuple[argparse.Namespace, EasyDict]:
    """Parses script arguments and configuration files

    :return: script arguments, config
    :rtype: Tuple[argparse.Namespace, EasyDict]
    """
    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument(
        "--cfg_file",
        type=str,
        default="cfgs/models/nuscenes/TED-S.yaml",
        help="specify the config for prediction",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/data/sets/nuscenes/v1.0-mini/",
        help=(
            "specify the point cloud data file or directory, "
            "for Nuscenes format, this should be the path given "
            "in the NuScenes class constructor from nuscenes-devkit"
        ),
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="ckpt/nuscenes/TED-S.pth",
        help="specify the pretrained model",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="output.json",
        help="specify the file to which the result will be saved",
    )
    parser.add_argument(
        "--is_only_valid",
        type=bool,
        default=True,
        help="specify if predict only scenes from validation set",
    )
    parser.add_argument(
        "--detector",
        type=str,
        default="SPHEREFORMER",
        help="specify if predict only scenes from validation set",
    )

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def create_model(
    cfg: EasyDict, dataset: Dataset, ckpt: str, logger: Optional[Logger] = None
) -> Detector3DTemplate:
    """Builds a detector

    :param cfg: config
    :type cfg: EasyDict
    :param dataset: dataset for which pre-annotations are to be generated
    :type dataset: Dataset
    :param ckpt: checkpoint used for detection
    :type ckpt: str
    :param logger: logger, defaults to None
    :type logger: Optional[Logger]
    :return: model - detector
    :rtype: Detector3DTemplate
    """
    model = build_network(
        model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset
    )
    model.load_params_from_file(filename=ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()

    return model


def generate_annotation(
    dataset: Dataset, model: Detector3DTemplate, args, logger: Optional[Logger] = None
) -> Dict[str, List[Dict[str, Union[str, float, List[float], None]]]]:
    """Generates pre-annotations

    :param dataset: dataset for which pre-annotations are to be generated
    :type dataset: Dataset
    :param model: detector used to generate pre-annotations
    :type model: Detector3DTemplate
    :param logger: logger, defaults to None
    :type logger: Optional[Logger], optional
    :return: Dictionary containing annotations for sequences
    :rtype: Dict[str, List[Dict[str, Union[str, float, List[float], None]]]]
    """
    results: Dict[str, List[Dict[str, Union[str, float, List[float], None]]]] = {}
    with torch.no_grad():
        for idx, data_dict in enumerate(dataset):
            if args.is_only_valid and type(dataset) == DATASETS['NuScenesDataset']:
                if not dataset.is_scene_from_validation_set(data_dict['frame_id']):
                    continue

            if logger is not None:
                logger.info(f"Sample index: \t{idx + 1}")
            data_dict = dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, *_ = model(data_dict)

            annos = dataset.generate_prediction_dicts(
                data_dict, pred_dicts, dataset.class_names
            )

            for scene_id, frame_annos in annos["results"].items():
                if scene_id in results:
                    results[scene_id].extend(frame_annos)
                else:
                    results[scene_id] = frame_annos

        if type(dataset) == DATASETS['NuScenesDataset']:
            results = change_results_to_nuscenes(results, Path(args.data_path))
        return results

def change_results_to_nuscenes(results, data_path):

    nusc = NuScenes(dataroot=data_path.parent, version=data_path.name, verbose=False)

    sequences_json = {'dataroot':str(data_path.parent), 'sequences':[]}
    for scene_id, sequence in results.items():
        new_sequence = {'scene_id':scene_id, 'frames':{}}
        for detection in sequence:
            if 'car' not in detection['detection_name']:
                continue
            frame_id = detection['sample_token']
            detection.pop('sample_token')
            detection.pop('velocity')
            detection.pop('attribute_name')
            if frame_id in new_sequence['frames']:
                new_sequence['frames'][frame_id].append(detection)
            else:
                new_sequence['frames'][frame_id] = [detection]
        sequences_json['sequences'].append(new_sequence)

    new_sequences = []
    for sequence in sequences_json['sequences']:
        scene_dict = nusc.get('scene', sequence['scene_id'])
        frame_dict = nusc.get('sample', scene_dict['first_sample_token'])
        new_frames = []
        while True:
            if frame_dict['token'] in sequence['frames']:
                new_frames.append({'frame_token':frame_dict['token'],'boxes':sequence['frames'][frame_dict['token']]})
            else:
                print(frame_dict['token'], "was not in sequence['frames']")
                new_frames.append({'frame_token':frame_dict['token'],'boxes':[]})
            if frame_dict['next'] == '':
                break
            frame_dict = nusc.get('sample', frame_dict['next'])
        new_sequences.append(new_frames)
    
    sequences_json['sequences'] = new_sequences
    return sequences_json

def main() -> None:
    """
    Main function
    """
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info("RUN ANNOTATION GENERATOR")

    dataset_name = cfg["DATA_CONFIG"]["DATASET"]
    DatasetClass = DATASETS[dataset_name]

    try:
        dataset = DatasetClass(
            dataset_cfg=cfg["DATA_CONFIG"],
            class_names=cfg["CLASS_NAMES"],
            detector=args.detector,
            root_path=Path(args.data_path),
            logger=logger
        )
    except ValueError:
        logger.error("Dataset loading failed. Exiting...")
        sys.exit(1)

    logger.info("Total number of samples: \t %d}", len(dataset))

    model = create_model(cfg, dataset, args.ckpt, logger)

    try:
        results = generate_annotation(dataset, model, args, logger=logger)
    except InvalidPointCloudException as exc:
        logger.error(str(exc))
        sys.exit(2)

    output_file = (
        args.output if args.output.endswith(".json") else args.output + ".json"
    )

    with open(output_file, "w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=4)

    logger.info("Generation done.")


if __name__ == "__main__":
    main()
