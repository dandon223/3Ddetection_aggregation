import os
import argparse
from prepare_to_metric import get_sequence
from visualize_with_histogram import process_camera
from matplotlib import pyplot as plt
from nuscenes.nuscenes import NuScenes  # type: ignore
from matplotlib.patches import Rectangle


def main(args):

    input_file = get_sequence(args.input_file)
    ground_truth = get_sequence(args.ground_truth)
    input_comparisonss = get_sequence(args.input_comparisonss)
    nusc = NuScenes(dataroot=input_file['dataroot'],
                    version=input_file['version'],
                    verbose=True)


    for sequence_index, sequence in enumerate(input_comparisonss):
        for frame_index, frame in enumerate(sequence):
            sample_data_dict = nusc.get('sample', input_file['sequences'][sequence_index][frame_index]['frame_token'])['data']
            cameras_tokens = [sample_data_dict['CAM_FRONT'], sample_data_dict['CAM_FRONT_RIGHT'], sample_data_dict['CAM_BACK_RIGHT'],
                              sample_data_dict['CAM_BACK'], sample_data_dict['CAM_BACK_LEFT'], sample_data_dict['CAM_FRONT_LEFT']]
            # for camera_token, ground_truth_boxes in ground_truth['sequences'][sequence_index][frame_index]['boxes'].items():
            for camera_token in cameras_tokens:
                no_boxes = True
                sd_rec = nusc.get('sample_data', camera_token)

            # display 2d box annotation
                img = plt.imread(os.path.join(
                    input_file['dataroot'], sd_rec['filename']))
                _, ax = plt.subplots()
                ax.imshow(img)
                if camera_token in ground_truth['sequences'][sequence_index][frame_index]['boxes']:
                    if camera_token in frame:
                        ious, _, _, _ = process_camera(frame[camera_token])
                    else:
                        ious = {}
                    for ground_truth_index, box_data in enumerate(ground_truth['sequences'][sequence_index][frame_index]['boxes'][camera_token]):
                        box_dims = box_data['bbox_corners']
                        box_root = (box_dims[0], box_dims[1])
                        box_size = (box_dims[2] - box_dims[0],
                                    box_dims[3] - box_dims[1])
                        ax.add_patch(
                            Rectangle(box_root, *box_size, fc='none', color="red"))
                        if ground_truth_index in ious:
                            (_, iou) = ious[ground_truth_index]
                            ax.text(box_root[0]+box_size[0]/2.0, box_root[1]+box_size[1]/2.0, str(round(iou, 2) ), fontdict=None, color="black", size="xx-large")
                        no_boxes = False
                
                if camera_token in input_file['sequences'][sequence_index][frame_index]['boxes']:
                    for index, box_data in enumerate(input_file['sequences'][sequence_index][frame_index]['boxes'][camera_token]):
                        box_dims = box_data['bbox_corners']
                        box_root = (box_dims[0], box_dims[1])
                        box_size = (box_dims[2] - box_dims[0],
                                    box_dims[3] - box_dims[1])
                        ax.add_patch(
                            Rectangle(box_root, *box_size, fc='none', color="green"))
                        no_boxes = False
                plt.title(
                    str(os.path.join(input_file['dataroot'], sd_rec['filename'])))
                if not no_boxes:
                    plt.show()
                else:
                    plt.close()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Display 2D annotations from reprojections.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--input_file',
        type=str,
        default='created_files/image_annotations_prepered.json',
        help='specify the input json from 2D detector',
    )
    parser.add_argument(
        '--ground_truth',
        type=str,
        default='created_files/ground_truth_image_annotations.json',
        help='specify the input json of ground truth',
    )
    parser.add_argument(
        '--input_comparisonss',
        type=str,
        default='created_files/image_annotations_ious.json',
        help='specify the input json of comparisonss of iou',
    )
    args = parser.parse_args()
    main(args)