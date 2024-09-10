#pylint: disable=missing-module-docstring
import numpy as np

from typing import List, Dict, Tuple
from nuscenes.utils.data_classes import Box  # type: ignore
from pyquaternion import Quaternion  # type: ignore
from helper_methods.get_iou import get_iou
from helper_methods.nuscenes_helper import box_to_list

def _average_angle(yaw_one: float, yaw_two: float, period: float) -> float:
    difference = abs(yaw_one - yaw_two) % (period / 2)
    if yaw_one < yaw_two:
        average = yaw_one + difference/2.
    elif yaw_two < yaw_one:
        average = yaw_two + difference/2.
    else:
        average = yaw_one

    while average < 0:
        average = average + np.pi
    while average >= np.pi:
        average = average - np.pi
    return average

def _merge_boxes(box: Box, new_box: Box) -> Box:
    center = [(box.center[0] + new_box.center[0]) / 2.0,
              (box.center[1] + new_box.center[1]) / 2.0,
              (box.center[2] + new_box.center[2]) / 2.0]
    wlh = [(box.wlh[0] + new_box.wlh[0]) / 2.0,
           (box.wlh[1] + new_box.wlh[1]) / 2.0,
           (box.wlh[2] + new_box.wlh[2]) / 2.0]
    new_angle = _average_angle(
        box.orientation.yaw_pitch_roll[0],
        new_box.orientation.yaw_pitch_roll[0],
        2*np.pi)
    orientation = Quaternion(axis=[0, 0, 1], radians=new_angle)
    return Box(
        center,
        wlh,
        orientation,
        name=box.name,
        score=(box.score + new_box.score)/2.0,
        label=box.label)

def merge_duplicates(output_boxes: List[Box], iou_ths: float) -> List[Dict[str, Tuple[str, float, List[float]]]]:
    """Merges boxes with iou > 0.7

    :param output_boxes: input boxes
    :return: merged boxes
    """
    new_output_boxes = []
    merged: List = []
    for i in range(len(output_boxes)):
        if i in merged:
            continue
        best_iou = 0.
        index_j = 0
        for j in range(len(output_boxes)):
            if j <= i or j in merged:
                continue
            iou = get_iou(box_to_list(output_boxes[i]),
                          box_to_list(output_boxes[j]))
            if iou > best_iou:
                best_iou = iou
                index_j = j

        if best_iou > iou_ths:
            new_output_boxes.append(box_to_list(_merge_boxes(output_boxes[i],
                                                             output_boxes[index_j])))
            merged.append(i)
            merged.append(index_j)
        else:
            new_output_boxes.append(box_to_list(output_boxes[i]))
            merged.append(i)

    return new_output_boxes
