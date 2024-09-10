# Copied from https://github.com/open-mmlab/OpenPCDet/blob/master/tools/demo.py
# and slightly modified

import argparse
from pathlib import Path

try:
    import open3d
    from visual_utils import open3d_vis_utils as V

    OPEN3D_FLAG = True
except:
    import mayavi.mlab as mlab
    from visual_utils import visualize_utils as V

    OPEN3D_FLAG = False

import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models import build_network, load_data_to_gpu
from pcdet.models.model_utils import model_nms_utils
from pcdet.utils import common_utils

from dataset_predict import DATASETS


def parse_config():
    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument(
        "--cfg_file",
        type=str,
        default="cfgs/models/nuscenes/TED-S.yaml",
        help="specify the config for demo",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/data/sets/nuscenes/v1.0-mini",
        help="specify the point cloud data file or directory",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="../../ckpt/nuscenes/TED-S.pth",
        help="specify the pretrained model",
    )

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info("-----------------Quick Demo of OpenPCDet-------------------------")

    dataset_name = cfg["DATA_CONFIG"]["DATASET"]
    DatasetClass = DATASETS[dataset_name]

    dataset = DatasetClass(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        root_path=Path(args.data_path),
        logger=logger,
    )
    logger.info(f"Total number of samples: \t{len(dataset)}")

    model = build_network(
        model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset
    )
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    with torch.no_grad():
        for idx, data_dict in enumerate(dataset):
            logger.info(f"Visualized sample index: \t{idx + 1}")
            data_dict = dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            boxes_dict, _, points_dict = model.forward(data_dict)

            box_dict = boxes_dict[0]

            points = points_dict["points1"][:, 1:]
            boxes = box_dict["pred_boxes"]

            pred_scores = box_dict["pred_scores"].cpu().numpy()
            pred_boxes = box_dict["pred_boxes"].cpu().numpy()
            pred_labels = box_dict["pred_labels"].cpu().numpy()

            if "WBF" in box_dict:
                pred_labels, pred_scores, pred_boxes = model_nms_utils.compute_WBF(
                    pred_labels, pred_scores, pred_boxes
                )

            V.draw_scenes(
                points=points,
                ref_boxes=pred_boxes,
                ref_scores=pred_scores,
                ref_labels=pred_labels,
            )

            if not OPEN3D_FLAG:
                mlab.show(stop=True)

    logger.info("Demo done.")


if __name__ == "__main__":
    main()
