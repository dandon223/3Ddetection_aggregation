"""map metric"""
import time
import logging
import numpy as np
from typing import Dict

from nuscenes import NuScenes # type: ignore
from nuscenes.eval.common.data_classes import EvalBoxes # type: ignore
from nuscenes.eval.common.loaders import add_center_dist, filter_eval_boxes # type: ignore
from nuscenes.eval.detection.algo import accumulate, calc_ap, calc_tp # type: ignore
from nuscenes.eval.detection.data_classes import DetectionConfig, DetectionMetrics, DetectionMetricDataList # type: ignore
from nuscenes.eval.detection.constants import TP_METRICS # type: ignore

def evaluate_map_metric(nusc: NuScenes,
                        cfg: DetectionConfig,
                        pred_boxes: EvalBoxes,
                        gt_boxes: EvalBoxes,
                        meta: Dict) -> Dict:
    # pylint: disable=line-too-long
    """Main method for calculating map in NuScenes format,
    mostly from https://github.com/nutonomy/nuscenes-devkit/blob/9b165b1018a64623b65c17b64f3c9dd746040f36/python-sdk/nuscenes/eval/detection/evaluate.py,
    but this method can be used on every combination of frames

    :param nusc: NuScenes
    :param cfg: config from NuScenes
    :param dt_annos: detections in NuScenes format
    :param gt_annos: ground truth in NuScenes format
    :return: mAP metrics for classses
    """
    # pylint: enable=line-too-long

    #logging.info('Initializing nuScenes detection evaluation')
    assert set(pred_boxes.sample_tokens) == set(gt_boxes.sample_tokens),\
        'Samples in split does not match samples in predictions.'

    # Add center distances.
    pred_boxes = add_center_dist(nusc, pred_boxes)
    gt_boxes = add_center_dist(nusc, gt_boxes)

    # Filter boxes (distance, points per box, etc.).
    #logging.info('Filtering predictions')
    pred_boxes = filter_eval_boxes(nusc, pred_boxes, cfg.class_range, verbose=True)
    #logging.info('Filtering ground truth annotations')
    gt_boxes = filter_eval_boxes(nusc, gt_boxes, cfg.class_range, verbose=True)

    # Evaluation
    start_time = time.time()
    # -----------------------------------
    # Step 1: Accumulate metric data for all classes and distance thresholds.
    # -----------------------------------
    #logging.info('Accumulating metric data...')
    metric_data_list = DetectionMetricDataList()
    for class_name in cfg.class_names:
        for dist_th in cfg.dist_ths:
            md = accumulate(gt_boxes, pred_boxes, class_name, cfg.dist_fcn_callable, dist_th)
            metric_data_list.set(class_name, dist_th, md)

    # -----------------------------------
    # Step 2: Calculate metrics from the data.
    # -----------------------------------
    #logging.info('Calculating metrics...')
    metrics = DetectionMetrics(cfg)
    for class_name in cfg.class_names:
        # Compute APs.
        for dist_th in cfg.dist_ths:
            metric_data = metric_data_list[(class_name, dist_th)]
            ap = calc_ap(metric_data, cfg.min_recall, cfg.min_precision)
            metrics.add_label_ap(class_name, dist_th, ap)

        # Compute TP metrics.
        for metric_name in TP_METRICS:
            metric_data = metric_data_list[(class_name, cfg.dist_th_tp)]
            if class_name in ['traffic_cone'] and metric_name in ['attr_err', 'vel_err', 'orient_err']:
                tp = np.nan
            elif class_name in ['barrier'] and metric_name in ['attr_err', 'vel_err']:
                tp = np.nan
            else:
                tp = calc_tp(metric_data, cfg.min_recall, metric_name)
            metrics.add_label_tp(class_name, metric_name, tp)

    # Compute evaluation time.
    metrics.add_runtime(time.time() - start_time)

    metrics_summary = metrics.serialize()
    metrics_summary['meta'] = meta.copy()

    # Print high-level metrics.
    # logging.info('mAP: %.4f', metrics_summary['mean_ap'])
    err_name_mapping = {
        'trans_err': 'mATE',
        'scale_err': 'mASE',
        'orient_err': 'mAOE',
        'vel_err': 'mAVE',
        'attr_err': 'mAAE'
    }
    #for tp_name, tp_val in metrics_summary['tp_errors'].items():
    #     logging.info('%s: %.4f', err_name_mapping[tp_name], tp_val)
    # logging.info('NDS: %.4f', metrics_summary['nd_score'])
    # logging.info('Eval time: %.1fs', metrics_summary['eval_time'])

    # Print per-class metrics.
    # logging.info('Per-class results:')
    # logging.info('%-20s\t%-6s\t%-6s\t%-6s\t%-6s\t%-6s\t%-6s',
    #              'Object Class', 'AP', 'ATE', 'ASE', 'AOE', 'AVE', 'AAE')
    class_aps = metrics_summary['mean_dist_aps']
    class_tps = metrics_summary['label_tp_errors']
    final_results: Dict[str, Dict] = {'ALL': {}}
    for class_name in class_aps.keys():
        final_results[class_name] = {}
        # logging.info('%-20s\t%-6.3f\t%-6.3f\t%-6.3f\t%-6.3f\t%-6.3f\t%-6.3f',
        #              class_name, class_aps[class_name],
        #              class_tps[class_name]['trans_err'],
        #              class_tps[class_name]['scale_err'],
        #              class_tps[class_name]['orient_err'],
        #              class_tps[class_name]['vel_err'],
        #              class_tps[class_name]['attr_err'])
        final_results[class_name]['AP'] = class_aps[class_name]
    return final_results
