# 2D annotations
This module is used for generation of 2D labels and to study how good they are compared to ground truth.

## Models
For now tests are being done on yolo_v8 as a detector that is well integrated with Python and pip and is pretrained.

### Yolo_v8
Yolo_v8 is installed from https://github.com/ultralytics/ultralytics repo.

#### How to install in conda
```
conda create -n yolo_v8 python=3.8
conda activate yolo_v8
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install ultralytics
pip install nuscenes-devkit

```
#### Demo
Inside yolo_v8 folder run:
```
python demo.py
```
Program will download yolo checkpoint trained on coco dataset and bus.jpg image. Then it will save 2d predictions on runs/detect/predict folder. It will also create 2D annotations on image.

#### Detections
Inside yolo_v8 folder run:
```
usage: create_predictions.py [-h] [--model MODEL] [--output_file OUTPUT_FILE] [--dataroot DATAROOT] [--version VERSION]
                             [--image_limit IMAGE_LIMIT] [--only_valid ONLY_VALID]

Create 3D projections.

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         Yolo model to download. (default: yolov8n.pt)
  --output_file OUTPUT_FILE
                        Output filename. (default: ./image_annotations.json)
  --dataroot DATAROOT   Path where nuScenes is saved. (default: /data/sets/nuscenes/v1.0-trainval/v1.0-trainval)
  --version VERSION     Dataset version. (default: v1.0-trainval)
  --image_limit IMAGE_LIMIT
                        Number of images to process or -1 to process all. (default: -1)
  --only_valid ONLY_VALID
                        If only validation set. (default: True)
```
Results will be saved in OUTPUT_FILE file in json format. This file can be used as an input file for programs in `fgr` folder.

## Study results
Every python file in this section works while being in 'from_2d_projection' envirenment created from environment.yml file.

### Step by step instruction
Every file has default parameters such that with base usage user does not have to overwrite them. 

* Activate 'from_2d_projection' conda envirenment.
* Use command 'python prepare_to_metric.py' in order to create needed json files for further processing.
```
usage: prepare_to_metric.py [-h] [--input_file INPUT_FILE] [--input_file_prepered INPUT_FILE_PREPERED]
                            [--output_file_ious OUTPUT_FILE_IOUS] [--output_ground_truth_file OUTPUT_GROUND_TRUTH_FILE]
                            [--visibilities VISIBILITIES [VISIBILITIES ...]]

Prepare for metrics

optional arguments:
  -h, --help            show this help message and exit
  --input_file INPUT_FILE
                        specify the input json from 2D detector (default: created_files/image_annotations.json)
  --input_file_prepered INPUT_FILE_PREPERED
                        specify json prepered file from 2D detector (this file is created by script) (default:
                        created_files/image_annotations_prepered.json)
  --output_file_ious OUTPUT_FILE_IOUS
                        specify the output json (default: created_files/image_annotations_ious.json)
  --output_ground_truth_file OUTPUT_GROUND_TRUTH_FILE
                        specify the output ground truth json (default: created_files/ground_truth_image_annotations.json)
  --visibilities VISIBILITIES [VISIBILITIES ...]
                        Visibility bins, the higher the number the higher the visibility. (default: ['', '1', '2', '3', '4'])
```
* Use command 'python visualize_with_histogram.py' to create IOUs histogram and other statistics.
```
usage: visualize_with_histogram.py [-h] [--input_file INPUT_FILE] [--ground_truth GROUND_TRUTH]
                                   [--input_comparisonss INPUT_COMPARISONSS] [--histograms_save HISTOGRAMS_SAVE] [--ths THS]

Visualize results

optional arguments:
  -h, --help            show this help message and exit
  --input_file INPUT_FILE
                        specify the input json from 2D detector (default: created_files/image_annotations_prepered.json)
  --ground_truth GROUND_TRUTH
                        specify the input json of ground truth (default: created_files/ground_truth_image_annotations.json)
  --input_comparisonss INPUT_COMPARISONSS
                        specify the input json of comparisonss of iou (default: created_files/image_annotations_ious.json)
  --histograms_save HISTOGRAMS_SAVE
                        specify the output files for histograms (default: created_files/histograms)
  --ths THS             specify the threshold of true positive detection (default: 0)
```
* Use command 'python display_2d_annotations.py' to visualize every picture from cameras where there are either created boxes by detector or ground truth boxes together with their best IOU.
```
usage: display_2d_annotations.py [-h] [--input_file INPUT_FILE] [--ground_truth GROUND_TRUTH]
                                 [--input_comparisonss INPUT_COMPARISONSS]

Display 2D annotations from reprojections.

optional arguments:
  -h, --help            show this help message and exit
  --input_file INPUT_FILE
                        specify the input json from 2D detector (default: created_files/image_annotations_prepered.json)
  --ground_truth GROUND_TRUTH
                        specify the input json of ground truth (default: created_files/ground_truth_image_annotations.json)
  --input_comparisonss INPUT_COMPARISONSS
                        specify the input json of comparisonss of iou (default: created_files/image_annotations_ious.json)
```
* Use command 'python export_2d_annotations_as_json.py' to export 2D annotations from NuScenes database.
```
usage: export_2d_annotations_as_json.py [-h] [--dataroot DATAROOT] [--version VERSION] [--output_file OUTPUT_FILE]
                                        [--visibilities VISIBILITIES [VISIBILITIES ...]] [--image_limit IMAGE_LIMIT]
                                        [--only_valid ONLY_VALID]

options:
  -h, --help            show this help message and exit
  --dataroot DATAROOT   Path where nuScenes is saved. (default: /data/sets/nuscenes/v1.0-trainval/v1.0-trainval)
  --version VERSION     Dataset version. (default: v1.0-trainval)
  --output_file OUTPUT_FILE
                        Output filename. (default: created_files/image_annotations.json)
  --visibilities VISIBILITIES [VISIBILITIES ...]
                        Visibility bins, the higher the number the higher the visibility. (default: ['', '1', '2', '3', '4'])
  --image_limit IMAGE_LIMIT
                        Number of images to process or -1 to process all. (default: -1)
  --only_valid ONLY_VALID
                        If only validation set. (default: True)
```