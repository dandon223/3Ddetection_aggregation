# Evaluate NuScenes
Allows to evaluate results of models on NuScenes database.

## Setup

Create conda env
```
conda env create -f environment.yml
```

Activate conda env
```
conda activate evaluate_map
```

## How to run

Run ```main.py``` file. Config parameters from NuScenes mAP metrics are merged into one json file. Such default file can be found under ```configs/sample_config.json```.

Config file contains:
* 'annotation_reading_params' field: path to generated annotations
* 'map_params' field: parameters used by NuScenes mAP evaluation
  * 'dataroot' and 'version' field: root and version of NuScenes database used by detector
  * 'use_*' fields: what detector used to generate annotations
  * 'detection_config_file' field: path to config file used by official NuScenes methods that evaluate mAP for classes. File ```configs/detection_cvpr_2019.json``` is official default config file.
  * 'classes' field: list of classes to evaluate

## How to test
```
python -m pytest -s
```
