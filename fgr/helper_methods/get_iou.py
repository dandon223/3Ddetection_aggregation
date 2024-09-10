#pylint: disable=missing-module-docstring
import numpy as np
from typing import Dict, List, Tuple
from pyquaternion import Quaternion  # type: ignore
from objectron.box import Box # type: ignore
from objectron.iou import IoU # type: ignore

def get_iou(box_one: Dict[str, Tuple[str, float, List[float]]],
            box_two: Dict[str, Tuple[str, float, List[float]]]) -> float:
    """returns iou between two boxes

    :param box_one: box one
    :param box_two: box two
    :return: iou metric
    """
    b1 = Box.from_transformation(Quaternion(box_one['rotation']).rotation_matrix, np.asarray(
        box_one['translation']), np.asarray(box_one['size']))
    b2 = Box.from_transformation(Quaternion(box_two['rotation']).rotation_matrix, np.asarray(
        box_two['translation']), np.asarray(box_two['size']))
    loss = IoU(b1, b2)
    return loss.iou()
