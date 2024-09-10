#pylint: disable=missing-module-docstring, broad-exception-caught, superfluous-parens, unsubscriptable-object
import math
import logging
import numpy as np
import numpy.typing as npt

from typing import List, Tuple, Optional
from easydict import EasyDict # type: ignore
from nuscenes.utils.data_classes import Box  # type: ignore
from pyquaternion import Quaternion  # type: ignore

from fgr.bbox_camera_to_lidar import find_points_in_frustum
from fgr.segmentation import fit_plane

def _check_number_of_points_inside(
        box_not_rotated: npt.NDArray[np.float64],
        points: npt.NDArray[np.float64]) -> int:
    number_of_points_inside = 0
    x_max = np.max(box_not_rotated[:, 0])
    x_min = np.min(box_not_rotated[:, 0])
    y_max = np.max(box_not_rotated[:, 1])
    y_min = np.min(box_not_rotated[:, 1])
    for point in points:
        if point[0] < x_min or point[0] > x_max or point[1] < y_min or point[1] > y_max:
            continue
        number_of_points_inside += 1
    return number_of_points_inside

def _check_if_other_direction(
        final_point: npt.NDArray[np.float64],
        loc1: npt.NDArray[np.float64],
        loc2: npt.NDArray[np.float64],
        angle: float,
        new_points: npt.NDArray[np.float64],
        number_of_points_inside: int,
        loc1_point: npt.NDArray[np.float64],
        loc1_delta: npt.NDArray[np.float64],
        distance: float) -> npt.NDArray[np.float64]:
    box_not_rotated = _rotate(
        np.array([final_point, loc1, loc2]), angle, [0, 0])
    number_of_points_inside_after = _check_number_of_points_inside(
        box_not_rotated, new_points)
    if number_of_points_inside_after <= number_of_points_inside:
        loc1_after = _point_in_distance_from_point_on_line(
            loc1_point, -loc1_delta, distance)
        box_not_rotated = _rotate(
            np.array([final_point, loc1_after, loc2]), angle, [0, 0])
        number_of_points_inside_after_2 = _check_number_of_points_inside(
            box_not_rotated, new_points)
        if number_of_points_inside_after_2 > number_of_points_inside_after:
            loc1 = loc1_after
        if number_of_points_inside_after_2 < number_of_points_inside \
            and number_of_points_inside_after < number_of_points_inside:
            logging.info('both number of points less, %s and %s < %s',\
                         number_of_points_inside_after_2, number_of_points_inside_after, number_of_points_inside)
    return loc1

def _rotate(points_2d: npt.NDArray[np.float64],
            angle: float,
            center: List[float]) -> npt.NDArray[np.float64]:
    if angle <= 0:
        rotate_matrix = np.array(
            [[np.cos(-angle), np.sin(-angle)], [-1 * np.sin(-angle), np.cos(-angle)]])
    else:
        rotate_matrix = np.array(
            [[np.cos(angle), -1 * np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    temp = points_2d - center
    temp = np.dot(temp, rotate_matrix)
    temp = temp + center
    return temp

def _find_key_vertex_by_pc_number(points_2d: npt.NDArray[np.float64],
                                  box: npt.NDArray[np.float64]) -> Tuple[int,
                                                                         int,
                                                                         npt.NDArray[np.float64],
                                                                         npt.NDArray[np.float64],
                                                                         np.int64,
                                                                         np.int64]:
    # first diagonal: (box[1], box[3]), corresponding points: box[0] / box[2]
    # (... > 0) is the constraint for key vertex's side towards the diagnoal
    if box[0][0] * (box[1][1] - box[3][1]) - box[0][1] * (box[1][0] - box[3][0]) + (
            box[1][0] * box[3][1] - box[1][1] * box[3][0]) > 0:
        index_1 = 0
    else:
        index_1 = 2

    # first diagonal: (box[1], box[3]), and calculate the point number on one side of this diagonal,
    # (... > 0) to constraint the current side, which is equal
    # to the side of key vertex (box[index_1])
    filter_1 = (points_2d[:, 0] * (box[1][1] - box[3][1])
                - points_2d[:, 1] * (box[1][0] - box[3][0])
                + (box[1][0] * box[3][1] - box[1][1] * box[3][0]) > 0)
    number_1 = np.sum(filter_1)

    # find which side contains more points, record this side and corresponding point number,
    # and key vertex, towards current diagonal (box[1], box[3])

    # number_1: most point number
    # index_1:  corresponding key vertex's index of bbox points
    # point_1:  corresponding key vertex

    if number_1 < points_2d.shape[0] / 2:
        number_1 = points_2d.shape[0] - number_1
        index_1 = (index_1 + 2) % 4

    point_1 = box[index_1]

    # second diagonal: (box[0], box[2]), corresponding points: box[1] / box[3]
    # (... > 0) to constraint the current side, which is equal to the side of key vertex (box[index_2])
    if box[1][0] * (box[0][1] - box[2][1]) - box[2][1] * (box[0][0] - box[2][0]) + (
            box[0][0] * box[2][1] - box[0][1] * box[2][0]) > 0:
        index_2 = 1
    else:
        index_2 = 3

    # find which side contains more points, record this side and corresponding point number,
    # and key vertex, towards current diagonal (box[0], box[2])

    # number_2: most point number
    # index_2:  corresponding key vertex's index of bbox points
    # point_2:  corresponding key vertex

    filter_2 = (points_2d[:, 0] *
                (box[0][1] -
                 box[2][1]) -
                points_2d[:, 1] *
                (box[0][0] -
                 box[2][0]) +
                (box[0][0] *
                 box[2][1] -
                 box[0][1] *
                 box[2][0]) > 0)
    number_2 = np.sum(filter_2)

    if number_2 < points_2d.shape[0] / 2:
        number_2 = points_2d.shape[0] - number_2
        index_2 = (index_2 + 2) % 4

    point_2 = box[index_2]
    return index_1, index_2, point_1, point_2, number_1, number_2

def _get_nuscene_box_from_box(box: npt.NDArray[np.float64],
                              angle: float,
                              z_max: np.float64,
                              z_min: np.float64,
                              center: npt.NDArray[np.float64]) -> Box:
    # Rotate box to original position
    box2d_rotated = _rotate(box, angle, [0, 0])
    center_rotated: List[float] = _rotate(center, angle, [0, 0])[0].tolist()
    seq = np.arange(0, 90.5 * np.pi / 180, 0.5 * np.pi / 180)
    angle = 0
    # Find angle of rotation of the box around its center
    for i in seq:
        temp = _rotate(box2d_rotated, i, center_rotated)
        if round(float(temp[0][0]), 3) == round(float(temp[1][0]), 3):
            angle = i
            break
    axis_aligned = _rotate(box2d_rotated, angle, center_rotated)
    center_box = [
        (axis_aligned[0][0] + axis_aligned[2][0]) / 2.0,
        (axis_aligned[0][1] + axis_aligned[2][1]) / 2.0,
        (z_max + z_min) / 2.0]
    scale = [float(abs(axis_aligned[0][0] - axis_aligned[2][0])),
             float(abs(axis_aligned[0][1] - axis_aligned[2][1])),
             float(abs((z_max - z_min)))]

    if scale[0] > scale[1]:
        scale = [scale[1], scale[0], scale[2]]
    else:
        if angle < np.pi / 2.0:
            angle = angle + np.pi / 2.0
        else:
            angle = angle - np.pi / 2.0

    return Box(center_box, scale, Quaternion(axis=[0, 0, 1], radians=angle))

def _delete_noisy_point_cloud(final: npt.NDArray[np.float64],
                              current_index: int,
                              points_2d: npt.NDArray[np.float64],
                              delete_times_every_epoch: int = 2) -> Tuple[npt.NDArray[np.float64],
                                                                          npt.NDArray[np.float64]]:
    # re-calculate key-vertex's location
    # points_2d: original point cloud
    # final: [rotated] point cloud
    # deleting method: from points_2d, calculate the point with maximum/minimum x and y,
    # extract their indexes, and delete them from numpy.array
    # one basic assumption on box's location order is: 0 to 3 => left-bottom
    # to left_top (counter-clockwise)
    if current_index in [2, 3]:
        for _ in range(delete_times_every_epoch):
            index = np.where(final == np.max(final[:, 0], axis=0))
            points_2d = np.delete(points_2d, index[0][0], axis=0)
            final = np.delete(final, index[0][0], axis=0)
    else:
        for _ in range(delete_times_every_epoch):
            index = np.where(final == np.min(final[:, 0], axis=0))
            points_2d = np.delete(points_2d, index[0][0], axis=0)
            final = np.delete(final, index[0][0], axis=0)

    if current_index in [1, 2]:
        for _ in range(delete_times_every_epoch):
            index = np.where(final == np.max(final[:, 1], axis=0))
            points_2d = np.delete(points_2d, index[0][0], axis=0)
            final = np.delete(final, index[0][0], axis=0)

    else:
        for _ in range(delete_times_every_epoch):
            index = np.where(final == np.min(final[:, 1], axis=0))
            points_2d = np.delete(points_2d, index[0][0], axis=0)
            final = np.delete(final, index[0][0], axis=0)
    return points_2d, final

def _calculate_plane(p1: npt.NDArray[np.float64],
                     p2: npt.NDArray[np.float64],
                     p3: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    v1 = p3 - p1
    v2 = p2 - p1

    # the cross product is a vector normal to the plane
    cp = np.cross(v1, v2)
    a, b, c = cp

    # This evaluates a * x3 + b * y3 + c * z3 which equals d
    d = np.dot(cp, p3)
    return np.array([a, b, c, d], dtype=np.float64)

def _calculate_line(
        p1: npt.NDArray[np.float64],
        p2: npt.NDArray[np.float64]) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    p_delta = p2 - p1
    return p1, p_delta  # x = x0 + t*x_delta, ...

def _point_plane_intesection(plane: npt.NDArray[np.float64],
                             point: npt.NDArray[np.float64],
                             delta: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    a, b, c, d = plane
    x_0, y_0, z_0 = point
    x_d, y_d, z_d = delta

    t = (d - a * x_0 - b * y_0 - c * z_0) / (a * x_d + b * y_d + c * z_d)

    return point + t * delta

def _calculate_height(frustum_corners: npt.NDArray[np.float64],
                      final_point: npt.NDArray[np.float64]) -> Tuple[np.float64, np.float64]:
    #      6-------7
    #      |       |
    #    5------8  |
    #    | |    |  |
    #    | 2----|--3
    #    |      |
    #    1------4
     # pylint: disable=invalid-name
    P1 = np.array(frustum_corners[3])
    P2 = np.array(frustum_corners[7])
    P3 = np.array(frustum_corners[6])
    P5 = np.array(frustum_corners[0])
    P6 = np.array(frustum_corners[4])
    P7 = np.array(frustum_corners[5])
     # pylint: enable=invalid-name
    final_point = np.array([final_point[0], final_point[1], 0], dtype=np.float64)
    plane = _calculate_plane(P1, P2, P3)
    point, delta = _calculate_line(final_point, final_point + [0, 0, 1])
    z_min = _point_plane_intesection(plane, point, delta)[2]

    plane = _calculate_plane(P5, P6, P7)
    point, delta = _calculate_line(final_point, final_point + [0, 0, 1])
    z_max = _point_plane_intesection(plane, point, delta)[2]

    return np.float64(z_max), np.float64(z_min)

def _point_in_distance_from_point_on_line(
        point: npt.NDArray[np.float64], delta: npt.NDArray[np.float64], distance: float) -> npt.NDArray[np.float64]:
    t = distance / math.sqrt(delta[0] * delta[0] + delta[1] * delta[1])
    return point + t * delta

def _angle_plane_line(
        plane: npt.NDArray[np.float64], delta: npt.NDArray[np.float64]) -> float:
    up = abs(plane[0] * delta[0] + plane[1] * delta[1] + plane[2] * delta[2])

    down = math.sqrt(
        plane[0] * plane[0] + plane[1] * plane[1] + plane[2] * plane[2]) * math.sqrt(
            delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2])

    sinus_angle = up / down

    return math.asin(sinus_angle)

def _find_intersection_point(box: npt.NDArray[np.float64],
                             final_index: int,
                             final_point: npt.NDArray[np.float64],
                             frustum_corners: npt.NDArray[np.float64],
                             z_min: np.float64,
                             indices_inside: npt.NDArray[np.int_]) -> Tuple[npt.NDArray[np.float64],
                                                                            npt.NDArray[np.float64],
                                                                            npt.NDArray[np.float64],
                                                                            float,
                                                                            float]:
    #      6-------7
    #      |       |
    #    5------8  |
    #    | |    |  |
    #    | 2----|--3
    #    |      |
    #    1------4
    # pylint: disable=invalid-name
    P1 = np.array(frustum_corners[3])
    P2 = np.array(frustum_corners[7])
    P3 = np.array(frustum_corners[6])
    P4 = np.array(frustum_corners[2])
    P5 = np.array(frustum_corners[0])
    P8 = np.array(frustum_corners[1])
    # pylint: enable=invalid-name
    left_point_box = box[final_index - 1]
    right_point_box = box[(final_index + 1) % 4]

    left_point, left_delta = _calculate_line(np.array([final_point[0], final_point[1], z_min]), np.array([
        left_point_box[0], left_point_box[1], z_min], dtype=np.float64))
    left_frustum_plane = _calculate_plane(P1, P2, P5)
    right_point, right_delta = _calculate_line(np.array([final_point[0], final_point[1], z_min]), np.array([
        right_point_box[0], right_point_box[1], z_min], dtype=np.float64))
    right_frustum_plane = _calculate_plane(P4, P3, P8)

    lp_lf_point = _point_plane_intesection(
        left_frustum_plane, left_point, left_delta)
    lp_lf_angle = _angle_plane_line(left_frustum_plane, left_delta)
    lp_rf_point = _point_plane_intesection(
        right_frustum_plane, left_point, left_delta)
    lp_rf_angle = _angle_plane_line(right_frustum_plane, left_delta)

    if math.dist(left_point_box, lp_lf_point[:2]) < math.dist(
            final_point, lp_lf_point[:2]) and (final_index - 1) % 4 in indices_inside:
        loc1 = lp_lf_point[:2]
        angle_1 = lp_lf_angle
    elif (final_index - 1) % 4 in indices_inside:
        loc1 = lp_rf_point[:2]
        angle_1 = lp_rf_angle
    else:
        loc1 = left_point_box
        angle_1 = np.pi / 2.0

    rp_rf_point = _point_plane_intesection(
        right_frustum_plane, right_point, right_delta)
    rp_rf_angle = _angle_plane_line(
        right_frustum_plane, right_delta)
    rp_lf_point = _point_plane_intesection(
        left_frustum_plane, right_point, right_delta)
    rp_lf_angle = _angle_plane_line(
        left_frustum_plane, right_delta)

    if math.dist(right_point_box, rp_lf_point[:2]) < math.dist(
            final_point, rp_lf_point[:2]) and (final_index + 1) % 4 in indices_inside:
        loc2 = rp_lf_point[:2]
        angle_2 = rp_lf_angle
    elif (final_index + 1) % 4 in indices_inside:
        loc2 = rp_rf_point[:2]
        angle_2 = rp_rf_angle
    else:
        loc2 = right_point_box
        angle_2 = np.pi / 2.0

    loc3 = np.array([0., 0.], dtype=np.float64)
    loc3[0] = loc1[0] - final_point[0] + loc2[0]
    loc3[1] = loc1[1] - final_point[1] + loc2[1]
    return loc1, loc2, loc3, angle_1, angle_2

def _check_anchor_fitting(number_of_points_inside: int,
                          new_points: npt.NDArray[np.float64],
                          angle: float,
                          loc1: npt.NDArray[np.float64],
                          loc2: npt.NDArray[np.float64],
                          loc3: npt.NDArray[np.float64],
                          angle_1: float,
                          angle_2: float,
                          final_point: npt.NDArray[np.float64],
                          z_max: np.float64,
                          z_min: np.float64,
                          detect_config: EasyDict) -> Tuple[npt.NDArray[np.float64],
                                                            npt.NDArray[np.float64],
                                                            npt.NDArray[np.float64]]:
    width_distance = np.abs(z_max - z_min) / detect_config.HEIGHT_WIDTH_RATE
    length_distance = np.abs(z_max - z_min) / detect_config.HEIGHT_LENGTH_RATE

    loc1_point, loc1_delta = _calculate_line(final_point, loc1)
    loc2_point, loc2_delta = _calculate_line(final_point, loc2)

    if angle_1 * 180 / np.pi < detect_config.ANCHOR_FIT_DEGREE_THRESH:
        current_distance = np.sqrt(
            (loc2[0] - final_point[0]) ** 2 + (loc2[1] - final_point[1]) ** 2)

        # if current_distance is larger than 2.2, we assume current boundary is length, otherwise width,
        # then use length-width rate to calculate another boundary
        if current_distance > detect_config.LENGTH_WIDTH_BOUNDARY:
            loc1 = _point_in_distance_from_point_on_line(
                loc1_point, loc1_delta, width_distance)
            loc1 = _check_if_other_direction(
                final_point,
                loc1,
                loc2,
                angle,
                new_points,
                number_of_points_inside,
                loc1_point,
                loc1_delta,
                width_distance)
        else:
            loc1 = _point_in_distance_from_point_on_line(
                loc1_point, loc1_delta, length_distance)
            loc1 = _check_if_other_direction(
                final_point,
                loc1,
                loc2,
                angle,
                new_points,
                number_of_points_inside,
                loc1_point,
                loc1_delta,
                length_distance)

        loc3[0] = loc1[0] - final_point[0] + loc2[0]
        loc3[1] = loc1[1] - final_point[1] + loc2[1]

    elif angle_2 * 180 / np.pi < detect_config.ANCHOR_FIT_DEGREE_THRESH:
        current_distance = np.sqrt(
            (loc1[0] - final_point[0]) ** 2 + (loc1[1] - final_point[1]) ** 2)

        if current_distance > detect_config.LENGTH_WIDTH_BOUNDARY:
            loc2 = _point_in_distance_from_point_on_line(
                loc2_point, loc2_delta, width_distance)
            loc2 = _check_if_other_direction(
                final_point,
                loc2,
                loc1,
                angle,
                new_points,
                number_of_points_inside,
                loc2_point,
                loc2_delta,
                width_distance)

        else:
            loc2 = _point_in_distance_from_point_on_line(
                loc2_point, loc2_delta, length_distance)
            loc2 = _check_if_other_direction(
                final_point,
                loc2,
                loc1,
                angle,
                new_points,
                number_of_points_inside,
                loc2_point,
                loc2_delta,
                length_distance)

        loc3[0] = loc1[0] - final_point[0] + loc2[0]
        loc3[1] = loc1[1] - final_point[1] + loc2[1]

    len_1 = np.sqrt((loc1[0] - final_point[0]) ** 2 +
                    (loc1[1] - final_point[1]) ** 2)
    len_2 = np.sqrt((loc2[0] - final_point[0]) ** 2 +
                    (loc2[1] - final_point[1]) ** 2)

    if len_1 < len_2:
        if len_1 < detect_config.MIN_WIDTH or len_1 > detect_config.MAX_WIDTH:
            loc1 = _point_in_distance_from_point_on_line(
                loc1_point, loc1_delta, width_distance)
            loc1 = _check_if_other_direction(
                final_point,
                loc1,
                loc2,
                angle,
                new_points,
                number_of_points_inside,
                loc1_point,
                loc1_delta,
                width_distance)

        if len_2 < detect_config.MIN_LENGTH or len_2 > detect_config.MAX_LENGTH:
            loc2 = _point_in_distance_from_point_on_line(
                loc2_point, loc2_delta, length_distance)
            loc2 = _check_if_other_direction(
                final_point,
                loc2,
                loc1,
                angle,
                new_points,
                number_of_points_inside,
                loc2_point,
                loc2_delta,
                length_distance)

        if not len_1 < detect_config.MIN_WIDTH \
            or len_1 > detect_config.MAX_WIDTH and not len_2 < detect_config.MIN_LENGTH \
            or len_2 > detect_config.MAX_LENGTH:
            loc1 = _check_if_other_direction(
                final_point,
                loc1,
                loc2,
                angle,
                new_points,
                number_of_points_inside,
                loc1_point,
                loc1_delta,
                width_distance)
            loc2 = _check_if_other_direction(
                final_point,
                loc2,
                loc1,
                angle,
                new_points,
                number_of_points_inside,
                loc2_point,
                loc2_delta,
                length_distance)

        loc3[0] = loc1[0] - final_point[0] + loc2[0]
        loc3[1] = loc1[1] - final_point[1] + loc2[1]

    elif len_2 < len_1:
        if len_1 < detect_config.MIN_LENGTH or len_1 > detect_config.MAX_LENGTH:
            loc1 = _point_in_distance_from_point_on_line(
                loc1_point, loc1_delta, length_distance)
            loc1 = _check_if_other_direction(
                final_point,
                loc1,
                loc2,
                angle,
                new_points,
                number_of_points_inside,
                loc1_point,
                loc1_delta,
                length_distance)
        if len_2 < detect_config.MIN_WIDTH or len_2 > detect_config.MAX_WIDTH:
            loc2 = _point_in_distance_from_point_on_line(
                loc2_point, loc2_delta, width_distance)
            loc2 = _check_if_other_direction(
                final_point,
                loc2,
                loc1,
                angle,
                new_points,
                number_of_points_inside,
                loc2_point,
                loc2_delta,
                width_distance)

        if not len_1 < detect_config.MIN_LENGTH \
            or len_1 > detect_config.MAX_LENGTH and not len_2 < detect_config.MIN_WIDTH \
            or len_2 > detect_config.MAX_WIDTH:
            loc1 = _check_if_other_direction(
                final_point,
                loc1,
                loc2,
                angle,
                new_points,
                number_of_points_inside,
                loc1_point,
                loc1_delta,
                length_distance)
            loc2 = _check_if_other_direction(
                final_point,
                loc2,
                loc1,
                angle,
                new_points,
                number_of_points_inside,
                loc2_point,
                loc2_delta,
                width_distance)

        loc3[0] = loc1[0] - final_point[0] + loc2[0]
        loc3[1] = loc1[1] - final_point[1] + loc2[1]

    return loc1, loc2, loc3

def find_box_fgr(points: npt.NDArray[np.float64],
                 detect_config: EasyDict,
                 ground_sample_points: npt.NDArray[np.float64],
                 truncate: bool,
                 frustum_corners: npt.NDArray[np.float64]) -> Optional[Box]:
    """Finds detection in 3D based on segmented points

    :param points: segmented points
    :param detect_config: Config parameters
    :param ground_sample_points: Sample of ground points
    :param truncate: If 2D detection is truncated
    :param frustum_corners: Corners of created frustum
    :return: Box if found, otherwise None
    """
    if len(points) < 10:
        logging.info('except points: %s', len(points))
        return None

    points_2d = points[:, 0:2].copy()
    current_angle = 0.0
    min_x = 0
    min_y = 0
    max_x = 100
    max_y = 100
    seq = np.arange(0, 90.5 * np.pi / 180, 0.5 * np.pi / 180)
    final_point = np.array([0., 0.], dtype=np.float64)
    cut_times = 0

    while True:
        min_value = np.float64(-1.)
        for i in seq:
            try:
                temp = _rotate(points_2d, i, [0, 0])
                current_min_x, current_min_y = np.amin(temp, axis=0)
                current_max_x, current_max_y = np.amax(temp, axis=0)

                # construct a sub-rectangle smaller than bounding box, whose
                # x_range and y_range is defined below:
                thresh_min_x = current_min_x + detect_config.RECT_SHRINK_THRESHOLD * \
                    (current_max_x - current_min_x)
                thresh_max_x = current_max_x - detect_config.RECT_SHRINK_THRESHOLD * \
                    (current_max_x - current_min_x)
                thresh_min_y = current_min_y + detect_config.RECT_SHRINK_THRESHOLD * \
                    (current_max_y - current_min_y)
                thresh_max_y = current_max_y - detect_config.RECT_SHRINK_THRESHOLD * \
                    (current_max_y - current_min_y)

                thresh_filter_1 = (temp[:, 0] >= thresh_min_x) & (
                    temp[:, 0] <= thresh_max_x)
                thresh_filter_2 = (temp[:, 1] >= thresh_min_y) & (
                    temp[:, 1] <= thresh_max_y)
                thresh_filter = (thresh_filter_1 &
                                 thresh_filter_2).astype(np.uint8)

                # calculate satisfying point number between original bbox and shrinked bbox
                # current_value = self.get_min_distance(temp_box, temp, temp_with_weights)
                current_value = np.float64(np.sum(thresh_filter) / temp.shape[0])

            except BaseException:
                logging.info('except')
                return None

            if current_value < min_value or min_value < 0:
                final = temp
                min_value = current_value
                current_angle = i
                min_x = current_min_x
                min_y = current_min_y
                max_x = current_max_x
                max_y = current_max_y

        box = np.array([[min_x, min_y],
                        [min_x, max_y],
                        [max_x, max_y],
                        [max_x, min_y]], dtype=np.float64)  # rotate clockwise
        angle = current_angle
        box_rotated = _rotate(box, -angle, [0, 0])
        index_1, index_2, point_1, point_2, number_1, number_2 = _find_key_vertex_by_pc_number(
            points_2d, box_rotated)
        # compare which side has the most points, then determine final diagonal,
        # final key vertex (current_final_point) and its index (current_index)
        # in bbox
        if number_1 < number_2:
            current_final_point = point_2
            current_index = index_2
        else:
            current_final_point = point_1
            current_index = index_1

        # quitting this loop requires:
        # 1. deleting point process has not stopped (cut_times is not positive)
        # 2. after deleting points, key vertex point's location is almost same
        # as that before deleting points
        if cut_times == 0 and (current_final_point[0] - final_point[0]) ** 2 + (
                current_final_point[1] - final_point[1]) ** 2 < detect_config.KEY_VERTEX_MOVE_DIST_THRESH:
            break
        else:
            if cut_times == 0:
                # the end of deleting point process, re-calculate new cut_times
                # with lower number of variable [points_2d]
                final_point = current_final_point

            else:
                # continue current deleting point process
                cut_times -= 1

                # avoid too fierce deleting
                if points_2d.shape[0] < detect_config.THRESH_MIN_POINTS_AFTER_DELETING:
                    logging.info('to fierce deleting')
                    return None

                points_2d, final = _delete_noisy_point_cloud(
                    final, current_index, points_2d, detect_config.DELETE_TIMES_EVERY_EPOCH)

    # while the loop is broken, the variable [box] is the final selected bbox
    # for car point clouds
    index_1, index_2, point_1, point_2, number_1, number_2 = _find_key_vertex_by_pc_number(
        points_2d, box_rotated)

    # here we get final key-vertex (final_point) and its index in box
    # (final_index)
    if number_1 < number_2:
        final_point = point_2
        final_index = index_2
    else:
        final_point = point_1
        final_index = index_1

    z_min, z_max = _calculate_height(frustum_corners, final_point)

    if np.abs(z_max - z_min) < detect_config.MIN_HEIGHT_NORMAL or \
       np.abs(z_max - z_min) > detect_config.MAX_HEIGHT_NORMAL or \
       (truncate and (z_max < detect_config.MIN_TOP_TRUNCATE or
                      z_max > detect_config.MAX_TOP_TRUNCATE or
                      z_min < detect_config.MIN_BOT_TRUNCATE or
                      z_min > detect_config.MAX_BOT_TRUNCATE)):
        # for truncate cases, calculating height from frustum may fail if
        # key-vertex is not inside frustum area
        z_min = np.min(points[:, 2])
        plane = fit_plane(ground_sample_points)
        eps = 1e-8
        sign = np.sign(np.sign(plane[2]) + 0.5)
        try:
            z_max = np.float64(-1 * (plane[0] * final_point[0] + plane[1]
                                     * final_point[1] - 1) / (plane[2] + eps * sign))
        except BaseException:
            z_max = np.max(points[:, 2])
        if np.abs(z_max - z_min) < detect_config.MIN_HEIGHT_NORMAL or \
           np.abs(z_max - z_min) > detect_config.MAX_HEIGHT_NORMAL:
            z_max = np.max(points[:, 2])
            if np.abs(z_max - z_min) < detect_config.MIN_HEIGHT_NORMAL or \
               np.abs(z_max - z_min) > detect_config.MAX_HEIGHT_NORMAL:
                z_max = np.float64(z_min + (detect_config.MIN_HEIGHT_NORMAL +
                                            detect_config.MAX_HEIGHT_NORMAL) / 2.0)

    # filter cars with very bad height
    if np.abs(z_max - z_min) < detect_config.MIN_HEIGHT_NORMAL or \
       np.abs(z_max - z_min) > detect_config.MAX_HEIGHT_NORMAL or \
       (truncate and (z_max < detect_config.MIN_TOP_TRUNCATE or
                      z_max > detect_config.MAX_TOP_TRUNCATE or
                      z_min < detect_config.MIN_BOT_TRUNCATE or
                      z_min > detect_config.MAX_BOT_TRUNCATE)):

        logging.info('top: %.4f, bottom: %.4f, car height: %.4f, deprecated', z_max, z_min, np.abs(z_max - z_min))
        return None

    new_points_temp = []
    for point in points:
        new_points_temp.append([point[0], point[1]])
    new_points = _rotate(np.array(new_points_temp), angle, [0, 0])

    number_of_points_inside = 0
    box_not_rotated = _rotate(box_rotated, angle, [0, 0])
    number_of_points_inside = _check_number_of_points_inside(
        box_not_rotated, new_points)

    # 3. calculate intersection from key-vertex to frustum [horizontally], to
    # get car's length and width
    if truncate:
        loc1 = box_rotated[final_index - 1]
        loc2 = box_rotated[(final_index + 1) % 4]
        loc3 = np.array([0., 0.], dtype=np.float64)
        loc3[0] = loc1[0] - final_point[0] + loc2[0]
        loc3[1] = loc1[1] - final_point[1] + loc2[1]

        average_z = (z_max + z_min) / 2.0
        box_rotated_copy = np.array([[box_rotated[0][0],
                                      box_rotated[0][1],
                                      average_z],
                                     [box_rotated[1][0],
                                      box_rotated[1][1],
                                      average_z],
                                     [box_rotated[2][0],
                                      box_rotated[2][1],
                                      average_z],
                                     [box_rotated[3][0],
                                      box_rotated[3][1],
                                      average_z]], dtype=np.float64)
        indices_inside = find_points_in_frustum(
            frustum_corners, box_rotated_copy)
        if final_index in indices_inside:
            loc1, loc2, loc3, angle_1, angle_2 = _find_intersection_point(
                box_rotated, final_index, final_point, frustum_corners, z_min, indices_inside)
            loc1, loc2, loc3 = _check_anchor_fitting(number_of_points_inside, new_points, angle,
                                                     loc1, loc2, loc3, angle_1, angle_2,
                                                     final_point, z_max, z_min, detect_config)

    else:
        indices_inside = np.array([0, 1, 2, 3])
        loc1, loc2, loc3, angle_1, angle_2 = _find_intersection_point(
            box_rotated, final_index, final_point, frustum_corners, z_min, indices_inside)
        loc1, loc2, loc3 = _check_anchor_fitting(number_of_points_inside, new_points, angle,
                                                 loc1, loc2, loc3, angle_1, angle_2,
                                                 final_point, z_max, z_min, detect_config)

    len_1 = np.sqrt((loc1[0] - final_point[0]) ** 2 +
                    (loc1[1] - final_point[1]) ** 2)
    len_2 = np.sqrt((loc2[0] - final_point[0]) ** 2 +
                    (loc2[1] - final_point[1]) ** 2)

    car_length = max(len_1, len_2)
    car_width = min(len_1, len_2)
    if not (detect_config.MIN_WIDTH <= car_width <= detect_config.MAX_WIDTH) or \
       not (detect_config.MIN_LENGTH <= car_length <= detect_config.MAX_LENGTH):

        logging.info('length: %.4f, width: %.4f, deprecated', car_length, car_width)
        return None

    points_rotated = np.array([final_point, loc1, loc2, loc3], dtype=np.float64)
    points = _rotate(points_rotated, angle, [0, 0])
    min_x, min_y = np.amin(points, axis=0)
    max_x, max_y = np.amax(points, axis=0)
    center = np.array([[(min_x + max_x) / 2.0, (min_y + max_y) / 2.0]], dtype=np.float64)
    box = np.array([[min_x, min_y],
                    [min_x, max_y],
                    [max_x, max_y],
                    [max_x, min_y]])
    nuscene_box = _get_nuscene_box_from_box(box, -angle, z_max, z_min, center)
    return nuscene_box
