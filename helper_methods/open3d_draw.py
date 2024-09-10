import open3d as o3d #type: ignore
import numpy as np
from typing import List
from pyquaternion import Quaternion #type: ignore
from nuscenes.utils.data_classes import Box  #type: ignore

def translate_boxes_to_open3d_instance(box: Box) -> o3d.geometry.LineSet:
    """Translates boxed to open3d instance
          4-------- 6
         /|         /|
        5 -------- 3 .
        | |        | |
        . 7 -------- 1
        |/         |/
        2 -------- 0

    :param box: _description_
    :type box: Box
    :return: _description_
    :rtype: o3d.geometry.LineSet
    """
    center = box.center.copy()
    wlh = box.wlh.copy()
    lwh = np.asarray([wlh[1], wlh[0], wlh[2]])
    rot = box.orientation.rotation_matrix
    box3d = o3d.geometry.OrientedBoundingBox(center, rot, lwh)

    line_set = o3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

    lines = np.asarray(line_set.lines)
    lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

    line_set.lines = o3d.utility.Vector2iVector(lines)

    return line_set

def to_nuscenes_box(box):
    return Box(box['translation'], box['size'], Quaternion(box['rotation']))

def draw_box(vis: o3d.visualization.Visualizer, boxes: List[Box], colours, ref_labels=None) -> o3d.visualization.Visualizer:
    """Draw box with open3d Visualizer

    :param vis: Visualizer
    :type vis: o3d.visualization.Visualizer
    :param boxes: List of boxes to draw
    :type boxes: List[Box]
    :param color: Color to set box, defaults to (0, 1, 0)
    :type color: tuple, optional
    :param ref_labels: Ref. labels, defaults to None
    :type ref_labels: _type_, optional
    :param score: IOU score, defaults to None
    :type score: _type_, optional
    :return: Visualizer object
    :rtype: o3d.visualization.Visualizer
    """
    for index, box in enumerate(boxes):
        if box is None:
            continue
        if type(box) == o3d.cuda.pybind.geometry.OrientedBoundingBox:
            line_set = o3d.geometry.LineSet.create_from_oriented_bounding_box(box)

            lines = np.asarray(line_set.lines)
            lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

            line_set.lines = o3d.utility.Vector2iVector(lines)
        elif type(box) == Box:
            line_set = translate_boxes_to_open3d_instance(box)
        else:
            box = to_nuscenes_box(box)
            line_set = translate_boxes_to_open3d_instance(box)
        if ref_labels is None:
            line_set.paint_uniform_color(colours[index])

        vis.add_geometry(line_set)

    return vis
