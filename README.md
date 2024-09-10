# 3D Detection aggregation

This repository is used for the research on annotation speedup of objects in lidar space using machine learning.\
It allows to train both TED and Sphereformer on nuscenes database and later to generate annotation on validation set. \
It also allows to create IoU histograms of results showing how well model taught itself.

## Overall structure of repo
* 2d_annotations folder is used for generating 2D annotations on pictures for FGR method to use later
* cfgs is a soft link to either sphereformer or tedfork configuration files
* evaluate_map for evaluating results with ap metric
* fgr is folder with modified code from https://github.com/weiyithu/FGR
* In helper_methods folder there are written methods that help with visualization of results both in 3d using open3d and by means of histograms.
* merge_annotations folder contains code for algorithm that merges annotations from multiple models/methods using heuristics given by user.
* objectron folder is fork of https://github.com/google-research-datasets/Objectron/tree/master/objectron/dataset used for IoU calculating.
* outputs folder is used for results of various computations.
* sphereformer folder is fork of https://github.com/dvlab-research/SphereFormer code with few changes.
* tedfork folder is fork of https://github.com/hailanyi/TED code with few changes.


## General flow of experiments
* Create needed env with .yml file and conda.
* Prepare database with pcdet.
* Train model with pcdet.
* Generate annotations with gen_annotations.py file.
* Prepare IoU metrics with prepare_to_metrics.py file.
* Create histogram visualizations of created boxes based on their distance from lidar sensor with visualize_with_histogram.py file.
* visualize_frames.py file is used if we want to see frame in 3d with both generated and ground truth boxes.

# TED
Code is based on https://github.com/hailanyi/TED.
## Prepare environment

To create the environment, you can use the `create_env.sh` script that is in `./tedfork` folder, while being in `base` conda environment.
```bash
./create_env.sh
```

Activate the environment:

```bash
conda activate TED
```

## Prepare NuScenes dataset
- You should download NuScenes dataset from https://www.nuscenes.org/nuscenes#download
- Both TED and Sphereformer are based on pcdet, so process of preparing nuscenes dataset is similar.
For TED we create symbolic links in tedfork/data/nuscenes folder and for Sphereformer we create them inside sphereformer/detection/data/nuscenes folder.

```
mkdir tedfork/data/nuscenes/v1.0-trainval
ln -s .../nuscenes/v1.0-trainval/maps/ tedfork/data/nuscenes/v1.0-trainval/maps
ln -s  .../nuscenes/v1.0-trainval/sweeps/ tedfork/data/nuscenes/v1.0-trainval/sweeps
ln -s  .../nuscenes/v1.0-trainval/samples/ tedfork/data/nuscenes/v1.0-trainval/samples
ln -s  .../nuscenes/v1.0-trainval/v1.0-trainval tedfork/data/nuscenes/v1.0-trainval/v1.0-trainval
cd tedfork/
python -m pcdet.datasets.nuscenes.nuscenes_dataset --func create_nuscenes_infos --cfg_file tools/cfgs/dataset_configs/nuscenes_dataset.yaml --version v1.0-trainval --train_size=700
```
- Note: `/path/to/nuscenes` should be a folder containing the `v1.0-mini` or/and `v.1.0-trainval` directory
- Note: train_size parameter is the number of scenes that will be prepared for model to learn from.

## Training
* Run following command in `TED` environment to start training:
```bash
cd tedfork/tools/ && python train.py --cfg_file path/to/model_config
```
e.g. for NuScenes
```bash
cd tedfork/tools/ && python train.py --cfg_file cfgs/models/nuscenes/TED-S.yaml
```
- Note: Checkpoints will be created in `tedfork/output/models/...`

# SPHEREFORMER

This module checks how segmentation can be used for refirenment of 3d boxes.
It uses model from SphereFormer based on https://github.com/dvlab-research/SphereFormer.

## Prepare environment

To create the environment, you can use the `environment.yml` file that is in `./sphereformer` folder, while being in `base` conda environment.
```bash
conda env create -f environment.yml
```

Activate the environment:

```bash
conda activate sphereformer
```
- Note: Remember for installing pcdet with python setup.py install per pcdet official instructions.

* How to install if conda env is not working. \
```
conda create -n sphereformer2 python=3.9 
conda activate sphereformer2 
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html 
pip install spconv-cu111==2.1.25 
pip install Pillow==10.0 
python setup.py develop 
cd tools/third_party/SparseTransformer/ 
python setup.py install 
pip install torch_scatter==2.0.9 
pip install tensorboard timm termcolor tensorboardX 
pip install torch_geometric==1.7.2 
pip install torch_sparse==0.6.12 -f https://data.pyg.org/whl/torch-1.8.1+cu111.html 
pip install torch_cluster==1.5.9 -f https://data.pyg.org/whl/torch-1.8.1+cu111.html 
```

## Prepare NuScenes dataset
- You should download NuScenes dataset from https://www.nuscenes.org/nuscenes#download
- Both TED and Sphereformer are based on pcdet, so process of preparing nuscenes dataset is similar.
For TED we create symbolic links in tedfork/data/nuscenes folder and for Sphereformer we create them inside sphereformer/detection/data/nuscenes folder.

```
mkdir sphereformer/detection/data/nuscenes/v1.0-trainval
ln -s  .../nuscenes/v1.0-trainval/maps/ sphereformer/detection/data/nuscenes/v1.0-trainval/maps
ln -s  .../v1.0-trainval/sweeps/ sphereformer/detection/data/nuscenes/v1.0-trainval/sweeps
ln -s  .../v1.0-trainval/samples/ sphereformer/detection/data/nuscenes/v1.0-trainval/samples
ln -s  .../v1.0-trainval/v1.0-trainval sphereformer/detection/data/nuscenes/v1.0-trainval/v1.0-trainval
cd sphereformer/detection/
python -m pcdet.datasets.nuscenes.nuscenes_dataset --func create_nuscenes_infos --cfg_file tools/cfgs/dataset_configs/nuscenes_dataset.yaml --version v1.0-trainval --train_size=700
```
- Note: `/path/to/nuscenes` should be a folder containing the `v1.0-mini` or/and `v.1.0-trainval` directory
- Note: train_size parameter is the number of scenes that will be prepared for model to learn from.

## Training
```
cd sphereformer/detection/tools/
CUDA_VISIBLE_DEVICES=6 python3 train.py --cfg_file cfgs/nuscenes_models/cbgs_voxel0075_res3d_centerpoint_sphereformer.yaml
```
- Note: Checkpoints will be created in `sphereformer/detection/output/models/...`

# Annotation generator for TED and SPHEREFORMER
Remember to change soft link of folder cfgs either to TED or SPHEREFORMER to folder in tools.
```
export PYTHONPATH=`pwd`
```
## Usage
To generate annotations, run the `gen_annotations.py` script with the appropriate parameters, the annotations will be saved to the file.

```
usage: gen_annotations.py [-h] [--cfg_file CFG_FILE] [--data_path DATA_PATH] [--ckpt CKPT] [--output OUTPUT]

optional arguments:
  -h, --help            show this help message and exit

  --cfg_file CFG_FILE   specify the config for prediction
  
  --data_path DATA_PATH
                        specify the point cloud data file or directory, for Nuscenes format, this should be the path given in the NuScenes class constructor from nuscenes-devkit
  
  --ckpt CKPT           specify the pretrained model
  
  --output OUTPUT       specify the file to which the    result will be saved
```

- the currently used detector config should be in `annotation_generator/cfgs/models/{dataset_name}`
- the currently used dataset config should be in `annotation_generator/cfgs/dataset_configs`
- the currently used checkpoint config should be in `annotation_generator/ckpt/{dataset_name}`, stored in git-lfs

# FGR
Implementation of code from https://arxiv.org/pdf/2105.07647.pdf in python with some differences. Original code was supports Kitti database.
In this reimplementation code now works with NuScenes database. The biggerst difference is that NuScenes have many cameras compared to one in Kitti. This code treats those cameras independently. From every camera we get detected boxes. If two cameras point at the same detected object, those detections in 3D should have iou metric > 0. If iou > 0.7 those boxes are merged together into one. 2D detections can be created in `2d_annotations` folder.

## Usage
The code works in fgr folder and ```from_2d_projection``` conda environment.
```
usage: fgr_algorithm.py [-h] [--vis] [--vis_global_points] [--input_file INPUT_FILE] [--output_file OUTPUT_FILE]
                   [--region_growth_config_path REGION_GROWTH_CONFIG_PATH]

Create 3D matlab projections.

optional arguments:
  -h, --help            show this help message and exit
  --vis                 If visualize frame (default: False)
  --vis_global_points   If visualize global frame points (default: False)
  --input_file INPUT_FILE
                        Input filename. (default: created_files/image_annotations.json)
  --output_file OUTPUT_FILE
                        Output filename. (default: created_files/fgr/image_annotations.json)
  --region_growth_config_path REGION_GROWTH_CONFIG_PATH
                        region_growth_config_path filename. (default: fgr/config.yml)
```
# Merge annotations
This algorithm is used to merges annotations from multiple models/methods using heuristics given by user.

## Usage
From root folder and from_2d_annotations conda environment.
```
export PYTHONPATH=`pwd`
python merge_annotations/merge_annotations.py
```
```
usage: merge_annotations.py [-h] [--config_file CONFIG_FILE]

options:
  -h, --help            show this help message and exit
  --config_file CONFIG_FILE
                        Input filename. (default: merge_annotations/config.json)
```
Most parameters used by this method is in config file.

# Find best merge strategy for input files.

## Usage
From root folder and from_2d_annotations conda environment.
```
cd evaluate_map/
export PYTHONPATH=`pwd`
cd ..
python best_config_merge.py
```

# Results summary

* Run prepare_to_metrics.py
```
usage: prepare_to_metrics.py [-h] [--input_file INPUT_FILE] [--output_file_ious OUTPUT_FILE_IOUS] [--output_ground_truth_file OUTPUT_GROUND_TRUTH_FILE]

optional arguments:
  -h, --help            show this help message and exit
  --input_file INPUT_FILE
                        specify the input json
  --output_file_ious OUTPUT_FILE_IOUS
                        specify the output json
  --output_ground_truth_file OUTPUT_GROUND_TRUTH_FILE
                        specify the output ground truth json
```
* Run visualize_with_histogram.py
```
usage: visualize_with_histogram.py [-h] [--input_file INPUT_FILE] [--ground_truth GROUND_TRUTH] [--input_comparisonss INPUT_COMPARISONSS]
                                   [--histograms_save HISTOGRAMS_SAVE] [--detector_name DETECTOR_NAME]

optional arguments:
  -h, --help            show this help message and exit
  --input_file INPUT_FILE
                        specify the input json from TED
  --ground_truth GROUND_TRUTH
                        specify the input json of ground truth
  --input_comparisonss INPUT_COMPARISONSS
                        specify the input json of comparisonss of iou
  --histograms_save HISTOGRAMS_SAVE
                        specify the output files for histograms
  --detector_name DETECTOR_NAME
                        specify the name of a detector
```

* Run visualize_frames.py for frames visualizations in 3D with detected and ground truth boxes
```
usage: visualize_frames.py [-h] [--input_file INPUT_FILE] [--ground_truth GROUND_TRUTH] [--input_comparisonss INPUT_COMPARISONSS]

optional arguments:
  -h, --help            show this help message and exit
  --input_file INPUT_FILE
                        specify the input json from TED
  --ground_truth GROUND_TRUTH
                        specify the input json of ground truth
  --input_comparisonss INPUT_COMPARISONSS
                        specify the input json of comparisonss of iou
```

# Licence

This library with source codes is available under MIT license.

Copyright (c) 2024 Daniel Gorniak, Robert Nowak, Warsaw University of Technology

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.