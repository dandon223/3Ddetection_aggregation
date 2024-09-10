# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring

import pytest # type: ignore
import numpy as np
from typing import Optional, List, Dict

from car_metric.correction_metric import evaluate_correction_metric
from car_metric.correction_metric_config import CorrectionMetricConfig, AverageTimesToCorrect, Thresholds
from read_annotations import read_annotations_car
from tests.test_data import test_data

CORRECTION_METRIC_CONFIG = CorrectionMetricConfig(
    average_times_to_correct=AverageTimesToCorrect(
        false_positive=1.0,
        false_negative=1.0,
        translation=1.0,
        rotation=1.0,
        scale=1.0,
        translation_and_rotation=1.0,
        rotation_and_scale=1.0,
        translation_and_scale=1.0,
        translation_and_rotation_and_scale=1.0,
        classification=1.0),
    thresholds={
        'Car': Thresholds(
            translation=0.1,
            rotation=0.1,
            scale=0.1)
    },
    classes=['Car'],
    iou_ths=0.0,
    score_ths=0.5
)

def dict_comparer(first: Dict, second: Dict, msg: Optional[str] = None) -> None:
    if first.keys() != second.keys():
        raise Exception( # pylint: disable=broad-exception-raised
            f'First dict contains the following keys: {first.keys()} '
            f'while second dict contains: {second.keys()}' if msg is None else msg) 

    top_level_fields = [
        'total_correction_time',
        'correction_time_by_frame',
        'correction_time_by_object',
        'baseline_correction_time',
        'correction_acceleration_ratio']
    for field in top_level_fields:
        if first[field] != second[field]:
            raise Exception( # pylint: disable=broad-exception-raised
                f'Value of field {field} differs: value in first dict is {first[field]} '
                f'while value in second dict is {second[field]}' if msg is None else msg)
        first.pop(field)
        second.pop(field)

    for class_name in first.keys():

        if first[class_name].keys() != second[class_name].keys():
            raise Exception( # pylint: disable=broad-exception-raised
                f'Class {class_name} in first dict contains the following keys: {first[class_name].keys()} '
                f'while the second dict contains: {second[class_name].keys()}' if msg is None else msg)

        remaining_fields = first[class_name].keys()
        for field in remaining_fields:
            if first[class_name][field] != second[class_name][field]:
                raise Exception( # pylint: disable=broad-exception-raised
                    f'Value of field {field} in class {class_name} differs: value in first dict is '
                    f'{first[class_name][field]} while value in second dict is '
                    f'{second[class_name][field]}' if msg is None else msg)

    if first != second: # all of the above checks should cover every scenario, so this should never fail
        raise Exception( # pylint: disable=broad-exception-raised
            f'Remainders of both dicts are not equal - first one is {first} and second one is {second}'
            if msg is None else msg)

class TestEvaluate():

    def test_single_object_ideal_placement(self):
        gt_annos: List[Dict[str, np.ndarray]] = [
            {
                'name': np.array(['Car']),
                'dimensions': np.array([[1., 1., 1.]]),
                'location': np.array([[0., 0., 0.]]),
                'rotation': np.array([[1., 0., 0., 0.]]),
                'score': np.array([0.])
            }
        ]
        dt_annos: List[Dict[str, np.ndarray]] = [
            {
                'name': np.array(['Car']),
                'dimensions': np.array([[1., 1., 1.,]]),
                'location': np.array([[0., 0., 0.]]),
                'rotation': np.array([[1., 0., 0., 0.]]),
                'score': np.array([1.0])
            }
        ]

        correction_metric_result: Dict = evaluate_correction_metric(
            gt_annos, dt_annos, CORRECTION_METRIC_CONFIG)
        expected_result: Dict = {
            'Car': {
                'correction_time': 0.0,
                'fp': 0,
                'fn': 0,
                'n_translation_errors': 0,
                'n_rotation_errors': 0,
                'n_scale_errors': 0,
                'n_translation_and_rotation_errors': 0,
                'n_rotation_and_scale_errors': 0,
                'n_translation_and_scale_errors': 0,
                'n_translation_and_rotation_and_scale_errors': 0,
                'n_classification_errors': 0,
                'tp': 1,
                'total_gt': 1,
                'total_dt': 1,
                'n_no_positional_errors': 1
            },
            'total_correction_time': 0.0,
            'correction_time_by_frame': 0.0,
            'correction_time_by_object': 0.0,
            'baseline_correction_time': 1.0,
            'correction_acceleration_ratio': 1.0
        }
        dict_comparer(correction_metric_result, expected_result)

    def test_single_complex_object_ideal_placement(self):
        annos: List[Dict[str, np.ndarray]] = [
            {
                'name': np.array(['Car']),
                'dimensions': np.array([[1.4020, 1.6825, 4.3136]]),
                'location': np.array([[3.3099, 2.3186, 34.6617]]),
                'rotation': np.array([[0.9512, -0.045, 0.1677, 0.2548]]),
                'score': np.array([1.])
            }
        ]

        correction_metric_result: Dict = evaluate_correction_metric(
            annos, annos, CORRECTION_METRIC_CONFIG)
        expected_result: Dict = {
            'Car': {
                'correction_time': 0.0,
                'fp': 0,
                'fn': 0,
                'n_translation_errors': 0,
                'n_rotation_errors': 0,
                'n_scale_errors': 0,
                'n_translation_and_rotation_errors': 0,
                'n_rotation_and_scale_errors': 0,
                'n_translation_and_scale_errors': 0,
                'n_translation_and_rotation_and_scale_errors': 0,
                'n_classification_errors': 0,
                'tp': 1,
                'total_gt': 1,
                'total_dt': 1,
                'n_no_positional_errors': 1
            },
            'total_correction_time': 0.0,
            'correction_time_by_frame': 0.0,
            'correction_time_by_object': 0.0,
            'baseline_correction_time': 1.0,
            'correction_acceleration_ratio': 1.0
        }
        dict_comparer(correction_metric_result, expected_result)

    def test_single_complex_object_almost_ideal_placement(self):
        gt_annos: List[Dict[str, np.ndarray]] = [
            {
                'name': np.array(['Car']),
                'dimensions': np.array([[1.4020, 1.6825, 4.3136]]),
                'location': np.array([[3.3099, 2.3186, 34.6617]]),
                'rotation': np.array([[0.9512, -0.045, 0.1677, 0.2548]]),
                'score': np.array([0.])
            }
        ]
        dt_annos: List[Dict[str, np.ndarray]] = [
            {
                'name': np.array(['Car']),
                'dimensions': np.array([[1.4020, 1.6825, 4.3136]]),
                'location': np.array([[3.30, 2.31, 34.66]]),
                'rotation': np.array([[0.9512, -0.045, 0.1677, 0.2548]]),
                'score': np.array([1.])
            }
        ]

        correction_metric_result: Dict = evaluate_correction_metric(
            gt_annos, dt_annos, CORRECTION_METRIC_CONFIG)
        expected_result: Dict = {
            'Car': {
                'correction_time': 0.0,
                'fp': 0,
                'fn': 0,
                'n_translation_errors': 0,
                'n_rotation_errors': 0,
                'n_scale_errors': 0,
                'n_translation_and_rotation_errors': 0,
                'n_rotation_and_scale_errors': 0,
                'n_translation_and_scale_errors': 0,
                'n_translation_and_rotation_and_scale_errors': 0,
                'n_classification_errors': 0,
                'tp': 1,
                'total_gt': 1,
                'total_dt': 1,
                'n_no_positional_errors': 1
            },
            'total_correction_time': 0.0,
            'correction_time_by_frame': 0.0,
            'correction_time_by_object': 0.0,
            'baseline_correction_time': 1.0,
            'correction_acceleration_ratio': 1.0
        }
        dict_comparer(correction_metric_result, expected_result)

    def test_multiple_complex_objects_ideal_placement(self):
        annos: List[Dict[str, np.ndarray]] = [
            {
                'name': np.array(['Car', 'Car']),
                'dimensions': np.array([[1.4020, 1.6825, 4.3136], [1.7018, 1.5782, 3.5694]]),
                'location': np.array([[3.3099, 2.3186, 34.6617], [3.3208, 1.7037, 9.2208]]),
                'rotation': np.array([[0.9512, -0.045, 0.1677, 0.2548], [0.9512, -0.045, 0.1677, 0.2548]]),
                'score': np.array([1., 1.])
            }
        ]

        correction_metric_result: Dict = evaluate_correction_metric(
            annos, annos, CORRECTION_METRIC_CONFIG)
        expected_result: Dict = {
            'Car': {
                'correction_time': 0.0,
                'fp': 0,
                'fn': 0,
                'n_translation_errors': 0,
                'n_rotation_errors': 0,
                'n_scale_errors': 0,
                'n_translation_and_rotation_errors': 0,
                'n_rotation_and_scale_errors': 0,
                'n_translation_and_scale_errors': 0,
                'n_translation_and_rotation_and_scale_errors': 0,
                'n_classification_errors': 0,
                'tp': 2,
                'total_gt': 2,
                'total_dt': 2,
                'n_no_positional_errors': 2
            },
            'total_correction_time': 0.0,
            'correction_time_by_frame': 0.0,
            'correction_time_by_object': 0.0,
            'baseline_correction_time': 2.0,
            'correction_acceleration_ratio': 1.0
        }
        dict_comparer(correction_metric_result, expected_result)

    def test_multiple_complex_objects_from_different_frames_ideal_placement(self):
        annos: List[Dict[str, np.ndarray]] = [
            {
                'name': np.array(['Car']),
                'dimensions': np.array([[1.4020, 1.6825, 4.3136]]),
                'location': np.array([[3.3099, 2.3186, 34.6617]]),
                'rotation': np.array([[0.9238, 0., 0., 0.3826]]),
                'score': np.array([1.])
            },
            {
                'name': np.array(['Car']),
                'dimensions': np.array([[1.7018, 1.5782, 3.5694]]),
                'location': np.array([[3.3208, 1.7037, 9.2208]]),
                'rotation': np.array([[0.9063, 0., 0., 0.4226]]),
                'score': np.array([1.])
            }
        ]

        correction_metric_result: Dict = evaluate_correction_metric(
            annos, annos, CORRECTION_METRIC_CONFIG)
        expected_result: Dict = {
            'Car': {
                'correction_time': 0.0,
                'fp': 0,
                'fn': 0,
                'n_translation_errors': 0,
                'n_rotation_errors': 0,
                'n_scale_errors': 0,
                'n_translation_and_rotation_errors': 0,
                'n_rotation_and_scale_errors': 0,
                'n_translation_and_scale_errors': 0,
                'n_translation_and_rotation_and_scale_errors': 0,
                'n_classification_errors': 0,
                'tp': 2,
                'total_gt': 2,
                'total_dt': 2,
                'n_no_positional_errors': 2
            },
            'total_correction_time': 0.0,
            'correction_time_by_frame': 0.0,
            'correction_time_by_object': 0.0,
            'baseline_correction_time': 2.0,
            'correction_acceleration_ratio': 1.0
        }
        dict_comparer(correction_metric_result, expected_result)

    def test_no_objects(self):
        gt_annos: List[Dict[str, np.ndarray]] = [
            {
                'name': np.array([]),
                'dimensions': np.array([]),
                'location': np.array([]),
                'rotation': np.array([]),
                'score': np.array([])
            }
        ]
        dt_annos: List[Dict[str, np.ndarray]] = [
            {
                'name': np.array([]),
                'dimensions': np.array([]),
                'location': np.array([]),
                'rotation': np.array([]),
                'score': np.array([])
            }
        ]

        correction_metric_result: Dict = evaluate_correction_metric(
            gt_annos, dt_annos, CORRECTION_METRIC_CONFIG)
        expected_result: Dict = {
            'Car': {
                'correction_time': 0.0,
                'fp': 0,
                'fn': 0,
                'n_translation_errors': 0,
                'n_rotation_errors': 0,
                'n_scale_errors': 0,
                'n_translation_and_rotation_errors': 0,
                'n_rotation_and_scale_errors': 0,
                'n_translation_and_scale_errors': 0,
                'n_translation_and_rotation_and_scale_errors': 0,
                'n_classification_errors': 0,
                'tp': 0,
                'total_gt': 0,
                'total_dt': 0,
                'n_no_positional_errors': 0
            },
            'total_correction_time': 0.0,
            'correction_time_by_frame': 0.0,
            'correction_time_by_object': 0.0,
            'baseline_correction_time': 0.0,
            'correction_acceleration_ratio': 1.0
        }
        dict_comparer(correction_metric_result, expected_result)

    def test_single_object_false_positive(self):
        gt_annos: List[Dict[str, np.ndarray]] = [
            {
                'name': np.array([]),
                'dimensions': np.array([]),
                'location': np.array([]),
                'rotation': np.array([]),
                'score': np.array([])
            }
        ]
        dt_annos: List[Dict[str, np.ndarray]] = [
            {
                'name': np.array(['Car']),
                'dimensions': np.array([[1., 1., 1.,]]),
                'location': np.array([[0., 0., 0.]]),
                'rotation': np.array([[1., 0., 0., 0.]]),
                'score': np.array([1.0])
            }
        ]

        correction_metric_result: Dict = evaluate_correction_metric(
            gt_annos, dt_annos, CORRECTION_METRIC_CONFIG)
        expected_result: Dict = {
            'Car': {
                'correction_time': 1.0,
                'fp': 1,
                'fn': 0,
                'n_translation_errors': 0,
                'n_rotation_errors': 0,
                'n_scale_errors': 0,
                'n_translation_and_rotation_errors': 0,
                'n_rotation_and_scale_errors': 0,
                'n_translation_and_scale_errors': 0,
                'n_translation_and_rotation_and_scale_errors': 0,
                'n_classification_errors': 0,
                'tp': 0,
                'total_gt': 0,
                'total_dt': 1,
                'n_no_positional_errors': 0
            },
            'total_correction_time': 1.0,
            'correction_time_by_frame': 1.0,
            'correction_time_by_object': 1.0,
            'baseline_correction_time': 0.0,
            'correction_acceleration_ratio': 0.0
        }
        dict_comparer(correction_metric_result, expected_result)

    def test_single_object_false_negative(self):
        gt_annos: List[Dict[str, np.ndarray]] = [
            {
                'name': np.array(['Car']),
                'dimensions': np.array([[1., 1., 1.]]),
                'location': np.array([[0., 0., 0.]]),
                'rotation': np.array([[1., 0., 0., 0.]]),
                'score': np.array([0.])
            }
        ]
        dt_annos: List[Dict[str, np.ndarray]] = [
            {
                'name': np.array([]),
                'dimensions': np.array([]),
                'location': np.array([]),
                'rotation': np.array([]),
                'score': np.array([])
            }
        ]

        correction_metric_result: Dict = evaluate_correction_metric(
            gt_annos, dt_annos, CORRECTION_METRIC_CONFIG)
        expected_result: Dict = {
            'Car': {
                'correction_time': 1.0,
                'fp': 0,
                'fn': 1,
                'n_translation_errors': 0,
                'n_rotation_errors': 0,
                'n_scale_errors': 0,
                'n_translation_and_rotation_errors': 0,
                'n_rotation_and_scale_errors': 0,
                'n_translation_and_scale_errors': 0,
                'n_translation_and_rotation_and_scale_errors': 0,
                'n_classification_errors': 0,
                'tp': 0,
                'total_gt': 1,
                'total_dt': 0,
                'n_no_positional_errors': 0
            },
            'total_correction_time': 1.0,
            'correction_time_by_frame': 1.0,
            'correction_time_by_object': 1.0,
            'baseline_correction_time': 1.0,
            'correction_acceleration_ratio': 0.0
        }
        dict_comparer(correction_metric_result, expected_result)

    def test_single_object_below_score_threshold(self):
        gt_annos: List[Dict[str, np.ndarray]] = [
            {
                'name': np.array(['Car']),
                'dimensions': np.array([[1., 1., 1.]]),
                'location': np.array([[0., 0., 0.]]),
                'rotation': np.array([[1., 0., 0., 0.]]),
                'score': np.array([0.])
            }
        ]
        dt_annos: List[Dict[str, np.ndarray]] = [
            {
                'name': np.array(['Car']),
                'dimensions': np.array([[1., 1., 1.]]),
                'location': np.array([[0., 0., 0.]]),
                'rotation': np.array([[1., 0., 0., 0.]]),
                'score': np.array([.9])
            }
        ]

        correction_metric_config = CorrectionMetricConfig(
            average_times_to_correct=AverageTimesToCorrect(
                false_positive=1.0,
                false_negative=1.0,
                translation=1.0,
                rotation=1.0,
                scale=1.0,
                translation_and_rotation=1.0,
                rotation_and_scale=1.0,
                translation_and_scale=1.0,
                translation_and_rotation_and_scale=1.0,
                classification=1.0),
            thresholds={
                'Car': Thresholds(
                    translation=0.1,
                    rotation=0.1,
                    scale=0.1)
            },
            classes=['Car'],
            iou_ths=0.0,
            score_ths=1.0 # override
        )

        correction_metric_result: Dict = evaluate_correction_metric(
            gt_annos, dt_annos, correction_metric_config)
        expected_result: Dict = {
            'Car': {
                'correction_time': 1.0,
                'fp': 0,
                'fn': 1,
                'n_translation_errors': 0,
                'n_rotation_errors': 0,
                'n_scale_errors': 0,
                'n_translation_and_rotation_errors': 0,
                'n_rotation_and_scale_errors': 0,
                'n_translation_and_scale_errors': 0,
                'n_translation_and_rotation_and_scale_errors': 0,
                'n_classification_errors': 0,
                'tp': 0,
                'total_gt': 1,
                'total_dt': 0,
                'n_no_positional_errors': 0
            },
            'total_correction_time': 1.0,
            'correction_time_by_frame': 1.0,
            'correction_time_by_object': 1.0,
            'baseline_correction_time': 1.0,
            'correction_acceleration_ratio': 0.0
        }
        dict_comparer(correction_metric_result, expected_result)

    def test_single_object_translation_error_below_translation_error_threshold(self):
        gt_annos: List[Dict[str, np.ndarray]] = [
            {
                'name': np.array(['Car']),
                'dimensions': np.array([[1., 1., 1.]]),
                'location': np.array([[0., 0., 0.]]),
                'rotation': np.array([[1., 0., 0., 0.]]),
                'score': np.array([0.])
            }
        ]
        dt_annos: List[Dict[str, np.ndarray]] = [
            {
                'name': np.array(['Car']),
                'dimensions': np.array([[1., 1., 1.]]),
                'location': np.array([[0., 0., 0.5]]),
                'rotation': np.array([[1., 0., 0., 0.]]),
                'score': np.array([1.])
            }
        ]

        correction_metric_config = CorrectionMetricConfig(
            average_times_to_correct=AverageTimesToCorrect(
                false_positive=1.0,
                false_negative=1.0,
                translation=1.0,
                rotation=1.0,
                scale=1.0,
                translation_and_rotation=1.0,
                rotation_and_scale=1.0,
                translation_and_scale=1.0,
                translation_and_rotation_and_scale=1.0,
                classification=1.0),
            thresholds={
                'Car': Thresholds(
                    translation=1.0,
                    rotation=0.1,
                    scale=0.1)
            },
            classes=['Car'],
            iou_ths=0.0,
            score_ths=0.5
        )

        correction_metric_result: Dict = evaluate_correction_metric(
            gt_annos, dt_annos, correction_metric_config)
        expected_result: Dict = {
            'Car': {
                'correction_time': 0.0,
                'fp': 0,
                'fn': 0,
                'n_translation_errors': 0,
                'n_rotation_errors': 0,
                'n_scale_errors': 0,
                'n_translation_and_rotation_errors': 0,
                'n_rotation_and_scale_errors': 0,
                'n_translation_and_scale_errors': 0,
                'n_translation_and_rotation_and_scale_errors': 0,
                'n_classification_errors': 0,
                'tp': 1,
                'total_gt': 1,
                'total_dt': 1,
                'n_no_positional_errors': 1
            },
            'total_correction_time': 0.0,
            'correction_time_by_frame': 0.0,
            'correction_time_by_object': 0.0,
            'baseline_correction_time': 1.0,
            'correction_acceleration_ratio': 1.0
        }
        dict_comparer(correction_metric_result, expected_result)

    def test_single_object_translation_error_above_translation_error_threshold(self):
        gt_annos: List[Dict[str, np.ndarray]] = [
            {
                'name': np.array(['Car']),
                'dimensions': np.array([[1., 1., 1.]]),
                'location': np.array([[0., 0., 0.]]),
                'rotation': np.array([[1., 0., 0., 0.]]),
                'score': np.array([0.])
            }
        ]
        dt_annos: List[Dict[str, np.ndarray]] = [
            {
                'name': np.array(['Car']),
                'dimensions': np.array([[1., 1., 1.]]),
                'location': np.array([[0., 0., 0.5]]),
                'rotation': np.array([[1., 0., 0., 0.]]),
                'score': np.array([1.])
            }
        ]

        correction_metric_result: Dict = evaluate_correction_metric(
            gt_annos, dt_annos, CORRECTION_METRIC_CONFIG)
        expected_result: Dict = {
            'Car': {
                'correction_time': 1.0,
                'fp': 0,
                'fn': 0,
                'n_translation_errors': 1,
                'n_rotation_errors': 0,
                'n_scale_errors': 0,
                'n_translation_and_rotation_errors': 0,
                'n_rotation_and_scale_errors': 0,
                'n_translation_and_scale_errors': 0,
                'n_translation_and_rotation_and_scale_errors': 0,
                'n_classification_errors': 0,
                'tp': 1,
                'total_gt': 1,
                'total_dt': 1,
                'n_no_positional_errors': 0
            },
            'total_correction_time': 1.0,
            'correction_time_by_frame': 1.0,
            'correction_time_by_object': 1.0,
            'baseline_correction_time': 1.0,
            'correction_acceleration_ratio': 0.0
        }
        dict_comparer(correction_metric_result, expected_result)

    def test_single_object_translation_error_below_iou_threshold(self):
        gt_annos: List[Dict[str, np.ndarray]] = [
            {
                'name': np.array(['Car']),
                'dimensions': np.array([[1., 1., 1.]]),
                'location': np.array([[0., 0., 0.]]),
                'rotation': np.array([[1., 0., 0., 0.]]),
                'score': np.array([0.])
            }
        ]
        dt_annos: List[Dict[str, np.ndarray]] = [
            {
                'name': np.array(['Car']),
                'dimensions': np.array([[1., 1., 1.]]),
                'location': np.array([[0., 0., 0.5]]),
                'rotation': np.array([[1., 0., 0., 0.]]),
                'score': np.array([1.])
            }
        ]

        correction_metric_config = CorrectionMetricConfig(
            average_times_to_correct=AverageTimesToCorrect(
                false_positive=1.0,
                false_negative=1.0,
                translation=1.0,
                rotation=1.0,
                scale=1.0,
                translation_and_rotation=1.0,
                rotation_and_scale=1.0,
                translation_and_scale=1.0,
                translation_and_rotation_and_scale=1.0,
                classification=1.0),
            thresholds={
                'Car': Thresholds(
                    translation=0.1,
                    rotation=0.1,
                    scale=0.1)
            },
            classes=['Car'],
            iou_ths=0.7, # override
            score_ths=0.5
        )

        correction_metric_result: Dict = evaluate_correction_metric(
            gt_annos, dt_annos, correction_metric_config)
        expected_result: Dict = {
            'Car': {
                'correction_time': 2.0,
                'fp': 1,
                'fn': 1,
                'n_translation_errors': 0,
                'n_rotation_errors': 0,
                'n_scale_errors': 0,
                'n_translation_and_rotation_errors': 0,
                'n_rotation_and_scale_errors': 0,
                'n_translation_and_scale_errors': 0,
                'n_translation_and_rotation_and_scale_errors': 0,
                'n_classification_errors': 0,
                'tp': 0,
                'total_gt': 1,
                'total_dt': 1,
                'n_no_positional_errors': 0
            },
            'total_correction_time': 2.0,
            'correction_time_by_frame': 2.0,
            'correction_time_by_object': 2.0,
            'baseline_correction_time': 1.0,
            'correction_acceleration_ratio': -1.0
        }
        dict_comparer(correction_metric_result, expected_result)

    def test_single_object_rotation_error_below_rotation_error_threshold(self):
        gt_annos: List[Dict[str, np.ndarray]] = [
            {
                'name': np.array(['Car']),
                'dimensions': np.array([[1., 1., 1.]]),
                'location': np.array([[0., 0., 0.]]),
                'rotation': np.array([[1., 0., 0., 0.]]),
                'score': np.array([0.])
            }
        ]
        dt_annos: List[Dict[str, np.ndarray]] = [
            {
                'name': np.array(['Car']),
                'dimensions': np.array([[1., 1., 1.]]),
                'location': np.array([[0., 0., 0.]]),
                'rotation': np.array([[0.9249, 0., 0., 0.3801]]),
                'score': np.array([1.])
            }
        ]

        correction_metric_config = CorrectionMetricConfig(
            average_times_to_correct=AverageTimesToCorrect(
                false_positive=1.0,
                false_negative=1.0,
                translation=1.0,
                rotation=1.0,
                scale=1.0,
                translation_and_rotation=1.0,
                rotation_and_scale=1.0,
                translation_and_scale=1.0,
                translation_and_rotation_and_scale=1.0,
                classification=1.0),
            thresholds={
                'Car': Thresholds(
                    translation=0.1,
                    rotation=1.0, # override
                    scale=0.1)
            },
            classes=['Car'],
            iou_ths=0.0,
            score_ths=0.5
        )

        correction_metric_result: Dict = evaluate_correction_metric(
            gt_annos, dt_annos, correction_metric_config)
        expected_result: Dict = {
            'Car': {
                'correction_time': 0.0,
                'fp': 0,
                'fn': 0,
                'n_translation_errors': 0,
                'n_rotation_errors': 0,
                'n_scale_errors': 0,
                'n_translation_and_rotation_errors': 0,
                'n_rotation_and_scale_errors': 0,
                'n_translation_and_scale_errors': 0,
                'n_translation_and_rotation_and_scale_errors': 0,
                'n_classification_errors': 0,
                'tp': 1,
                'total_gt': 1,
                'total_dt': 1,
                'n_no_positional_errors': 1
            },
            'total_correction_time': 0.0,
            'correction_time_by_frame': 0.0,
            'correction_time_by_object': 0.0,
            'baseline_correction_time': 1.0,
            'correction_acceleration_ratio': 1.0
        }
        dict_comparer(correction_metric_result, expected_result)

    def test_single_object_rotation_error_above_rotation_error_threshold(self):
        gt_annos: List[Dict[str, np.ndarray]] = [
            {
                'name': np.array(['Car']),
                'dimensions': np.array([[1., 1., 1.]]),
                'location': np.array([[0., 0., 0.]]),
                'rotation': np.array([[1., 0., 0., 0.]]),
                'score': np.array([0.])
            }
        ]
        dt_annos: List[Dict[str, np.ndarray]] = [
            {
                'name': np.array(['Car']),
                'dimensions': np.array([[1., 1., 1.]]),
                'location': np.array([[0., 0., 0.]]),
                'rotation': np.array([[0.9249, 0., 0., 0.3801]]),
                'score': np.array([1.])
            }
        ]

        correction_metric_result: Dict = evaluate_correction_metric(
            gt_annos, dt_annos, CORRECTION_METRIC_CONFIG)
        expected_result: Dict = {
            'Car': {
                'correction_time': 1.0,
                'fp': 0,
                'fn': 0,
                'n_translation_errors': 0,
                'n_rotation_errors': 1,
                'n_scale_errors': 0,
                'n_translation_and_rotation_errors': 0,
                'n_rotation_and_scale_errors': 0,
                'n_translation_and_scale_errors': 0,
                'n_translation_and_rotation_and_scale_errors': 0,
                'n_classification_errors': 0,
                'tp': 1,
                'total_gt': 1,
                'total_dt': 1,
                'n_no_positional_errors': 0
            },
            'total_correction_time': 1.0,
            'correction_time_by_frame': 1.0,
            'correction_time_by_object': 1.0,
            'baseline_correction_time': 1.0,
            'correction_acceleration_ratio': 0.0
        }
        dict_comparer(correction_metric_result, expected_result)

    def test_single_object_scale_error_below_scale_error_threshold(self):
        gt_annos: List[Dict[str, np.ndarray]] = [
            {
                'name': np.array(['Car']),
                'dimensions': np.array([[1., 1., 1.]]),
                'location': np.array([[0., 0., 0.]]),
                'rotation': np.array([[1., 0., 0., 0.]]),
                'score': np.array([0.])
            }
        ]
        dt_annos: List[Dict[str, np.ndarray]] = [
            {
                'name': np.array(['Car']),
                'dimensions': np.array([[2., 2., 2.]]),
                'location': np.array([[0., 0., 0.]]),
                'rotation': np.array([[1., 0., 0., 0.]]),
                'score': np.array([1.])
            }
        ]

        correction_metric_config = CorrectionMetricConfig(
            average_times_to_correct=AverageTimesToCorrect(
                false_positive=1.0,
                false_negative=1.0,
                translation=1.0,
                rotation=1.0,
                scale=1.0,
                translation_and_rotation=1.0,
                rotation_and_scale=1.0,
                translation_and_scale=1.0,
                translation_and_rotation_and_scale=1.0,
                classification=1.0),
            thresholds={
                'Car': Thresholds(
                    translation=0.1,
                    rotation=0.1,
                    scale=1.0) # override
            },
            classes=['Car'],
            iou_ths=0.0,
            score_ths=0.5
        )

        correction_metric_result: Dict = evaluate_correction_metric(
            gt_annos, dt_annos, correction_metric_config)
        expected_result: Dict = {
            'Car': {
                'correction_time': 0.0,
                'fp': 0,
                'fn': 0,
                'n_translation_errors': 0,
                'n_rotation_errors': 0,
                'n_scale_errors': 0,
                'n_translation_and_rotation_errors': 0,
                'n_rotation_and_scale_errors': 0,
                'n_translation_and_scale_errors': 0,
                'n_translation_and_rotation_and_scale_errors': 0,
                'n_classification_errors': 0,
                'tp': 1,
                'total_gt': 1,
                'total_dt': 1,
                'n_no_positional_errors': 1
            },
            'total_correction_time': 0.0,
            'correction_time_by_frame': 0.0,
            'correction_time_by_object': 0.0,
            'baseline_correction_time': 1.0,
            'correction_acceleration_ratio': 1.0
        }
        dict_comparer(correction_metric_result, expected_result)

    def test_single_object_scale_error_above_scale_error_threshold(self):
        gt_annos: List[Dict[str, np.ndarray]] = [
            {
                'name': np.array(['Car']),
                'dimensions': np.array([[1., 1., 1.]]),
                'location': np.array([[0., 0., 0.]]),
                'rotation': np.array([[1., 0., 0., 0.]]),
                'score': np.array([0.])
            }
        ]
        dt_annos: List[Dict[str, np.ndarray]] = [
            {
                'name': np.array(['Car']),
                'dimensions': np.array([[2., 2., 2.]]),
                'location': np.array([[0., 0., 0.]]),
                'rotation': np.array([[1., 0., 0., 0.]]),
                'score': np.array([1.])
            }
        ]

        correction_metric_result: Dict = evaluate_correction_metric(
            gt_annos, dt_annos, CORRECTION_METRIC_CONFIG)
        expected_result: Dict = {
            'Car': {
                'correction_time': 1.0,
                'fp': 0,
                'fn': 0,
                'n_translation_errors': 0,
                'n_rotation_errors': 0,
                'n_scale_errors': 1,
                'n_translation_and_rotation_errors': 0,
                'n_rotation_and_scale_errors': 0,
                'n_translation_and_scale_errors': 0,
                'n_translation_and_rotation_and_scale_errors': 0,
                'n_classification_errors': 0,
                'tp': 1,
                'total_gt': 1,
                'total_dt': 1,
                'n_no_positional_errors': 0
            },
            'total_correction_time': 1.0,
            'correction_time_by_frame': 1.0,
            'correction_time_by_object': 1.0,
            'baseline_correction_time': 1.0,
            'correction_acceleration_ratio': 0.0
        }
        dict_comparer(correction_metric_result, expected_result)

    def test_single_object_classification_error(self):
        gt_annos: List[Dict[str, np.ndarray]] = [
            {
                'name': np.array(['Car']),
                'dimensions': np.array([[1., 1., 1.]]),
                'location': np.array([[0., 0., 0.]]),
                'rotation': np.array([[1., 0., 0., 0.]]),
                'score': np.array([0.])
            }
        ]
        dt_annos: List[Dict[str, np.ndarray]] = [
            {
                'name': np.array(['Pedestrian']),
                'dimensions': np.array([[1., 1., 1.]]),
                'location': np.array([[0., 0., 0.]]),
                'rotation': np.array([[1., 0., 0., 0.]]),
                'score': np.array([1.])
            }
        ]

        correction_metric_config = CorrectionMetricConfig(
            average_times_to_correct=AverageTimesToCorrect(
                false_positive=1.0,
                false_negative=1.0,
                translation=1.0,
                rotation=1.0,
                scale=1.0,
                translation_and_rotation=1.0,
                rotation_and_scale=1.0,
                translation_and_scale=1.0,
                translation_and_rotation_and_scale=1.0,
                classification=1.0),
            thresholds={
                'Car': Thresholds(
                    translation=0.1,
                    rotation=0.1,
                    scale=0.1),
                'Pedestrian': Thresholds(
                    translation=0.1,
                    rotation=0.1,
                    scale=0.1),
            },
            classes=['Car', 'Pedestrian'], # override
            iou_ths=0.0,
            score_ths=0.5
        )

        correction_metric_result: Dict = evaluate_correction_metric(
            gt_annos, dt_annos, correction_metric_config)
        expected_result: Dict = {
            'Car': {
                'correction_time': 1.0,
                'fp': 0,
                'fn': 0,
                'n_translation_errors': 0,
                'n_rotation_errors': 0,
                'n_scale_errors': 0,
                'n_translation_and_rotation_errors': 0,
                'n_rotation_and_scale_errors': 0,
                'n_translation_and_scale_errors': 0,
                'n_translation_and_rotation_and_scale_errors': 0,
                'n_classification_errors': 1,
                'tp': 1,
                'total_gt': 1,
                'total_dt': 0,
                'n_no_positional_errors': 1
            },
            'Pedestrian': {
                'correction_time': 0.0,
                'fp': 0,
                'fn': 0,
                'n_translation_errors': 0,
                'n_rotation_errors': 0,
                'n_scale_errors': 0,
                'n_translation_and_rotation_errors': 0,
                'n_rotation_and_scale_errors': 0,
                'n_translation_and_scale_errors': 0,
                'n_translation_and_rotation_and_scale_errors': 0,
                'n_classification_errors': 0,
                'tp': 0,
                'total_gt': 0,
                'total_dt': 1,
                'n_no_positional_errors': 0
            },
            'total_correction_time': 1.0,
            'correction_time_by_frame': 1.0,
            'correction_time_by_object': 1.0,
            'baseline_correction_time': 1.0,
            'correction_acceleration_ratio': 0.0
        }
        dict_comparer(correction_metric_result, expected_result)

    def test_single_object_classification_error_ignored_detected_class(self):
        gt_annos: List[Dict[str, np.ndarray]] = [
            {
                'name': np.array(['Car']),
                'dimensions': np.array([[1., 1., 1.]]),
                'location': np.array([[0., 0., 0.]]),
                'rotation': np.array([[1., 0., 0., 0.]]),
                'score': np.array([0.])
            }
        ]
        dt_annos: List[Dict[str, np.ndarray]] = [
            {
                'name': np.array(['Ignored']),
                'dimensions': np.array([[1., 1., 1.]]),
                'location': np.array([[0., 0., 0.]]),
                'rotation': np.array([[1., 0., 0., 0.]]),
                'score': np.array([1.])
            }
        ]

        correction_metric_result: Dict = evaluate_correction_metric(
            gt_annos, dt_annos, CORRECTION_METRIC_CONFIG)
        expected_result: Dict = {
            'Car': {
                'correction_time': 1.0,
                'fp': 0,
                'fn': 0,
                'n_translation_errors': 0,
                'n_rotation_errors': 0,
                'n_scale_errors': 0,
                'n_translation_and_rotation_errors': 0,
                'n_rotation_and_scale_errors': 0,
                'n_translation_and_scale_errors': 0,
                'n_translation_and_rotation_and_scale_errors': 0,
                'n_classification_errors': 1,
                'tp': 1,
                'total_gt': 1,
                'total_dt': 0,
                'n_no_positional_errors': 1
            },
            'total_correction_time': 1.0,
            'correction_time_by_frame': 1.0,
            'correction_time_by_object': 1.0,
            'baseline_correction_time': 1.0,
            'correction_acceleration_ratio': 0.0
        }
        dict_comparer(correction_metric_result, expected_result)

    def test_single_object_classification_error_ignored_ground_truth_class(self):
        gt_annos: List[Dict[str, np.ndarray]] = [
            {
                'name': np.array(['Ignored']),
                'dimensions': np.array([[1., 1., 1.]]),
                'location': np.array([[0., 0., 0.]]),
                'rotation': np.array([[1., 0., 0., 0.]]),
                'score': np.array([0.])
            }
        ]
        dt_annos: List[Dict[str, np.ndarray]] = [
            {
                'name': np.array(['Car']),
                'dimensions': np.array([[1., 1., 1.]]),
                'location': np.array([[0., 0., 0.]]),
                'rotation': np.array([[1., 0., 0., 0.]]),
                'score': np.array([1.])
            }
        ]

        correction_metric_result: Dict = evaluate_correction_metric(
            gt_annos, dt_annos, CORRECTION_METRIC_CONFIG)
        expected_result: Dict = {
            'Car': {
                'correction_time': 1.0,
                'fp': 1,
                'fn': 0,
                'n_translation_errors': 0,
                'n_rotation_errors': 0,
                'n_scale_errors': 0,
                'n_translation_and_rotation_errors': 0,
                'n_rotation_and_scale_errors': 0,
                'n_translation_and_scale_errors': 0,
                'n_translation_and_rotation_and_scale_errors': 0,
                'n_classification_errors': 0,
                'tp': 0,
                'total_gt': 0,
                'total_dt': 1,
                'n_no_positional_errors': 0
            },
            'total_correction_time': 1.0,
            'correction_time_by_frame': 1.0,
            'correction_time_by_object': 1.0,
            'baseline_correction_time': 0.0,
            'correction_acceleration_ratio': 0.0
        }
        dict_comparer(correction_metric_result, expected_result)

    def test_single_object_classification_error_both_classes_ignored(self):
        gt_annos: List[Dict[str, np.ndarray]] = [
            {
                'name': np.array(['Ignored']),
                'dimensions': np.array([[1., 1., 1.]]),
                'location': np.array([[0., 0., 0.]]),
                'rotation': np.array([[1., 0., 0., 0.]]),
                'score': np.array([0.])
            }
        ]
        dt_annos: List[Dict[str, np.ndarray]] = [
            {
                'name': np.array(['Ignored']),
                'dimensions': np.array([[1., 1., 1.]]),
                'location': np.array([[0., 0., 0.]]),
                'rotation': np.array([[1., 0., 0., 0.]]),
                'score': np.array([1.])
            }
        ]

        correction_metric_result: Dict = evaluate_correction_metric(
            gt_annos, dt_annos, CORRECTION_METRIC_CONFIG)
        expected_result: Dict = {
            'Car': {
                'correction_time': 0.0,
                'fp': 0,
                'fn': 0,
                'n_translation_errors': 0,
                'n_rotation_errors': 0,
                'n_scale_errors': 0,
                'n_translation_and_rotation_errors': 0,
                'n_rotation_and_scale_errors': 0,
                'n_translation_and_scale_errors': 0,
                'n_translation_and_rotation_and_scale_errors': 0,
                'n_classification_errors': 0,
                'tp': 0,
                'total_gt': 0,
                'total_dt': 0,
                'n_no_positional_errors': 0
            },
            'total_correction_time': 0.0,
            'correction_time_by_frame': 0.0,
            'correction_time_by_object': 0.0,
            'baseline_correction_time': 0.0,
            'correction_acceleration_ratio': 1.0
        }
        dict_comparer(correction_metric_result, expected_result)

    def test_single_object_translation_and_rotation_error(self):
        gt_annos: List[Dict[str, np.ndarray]] = [
            {
                'name': np.array(['Car']),
                'dimensions': np.array([[1., 1., 1.]]),
                'location': np.array([[0., 0., 0.]]),
                'rotation': np.array([[1., 0., 0., 0.]]),
                'score': np.array([0.])
            }
        ]
        dt_annos: List[Dict[str, np.ndarray]] = [
            {
                'name': np.array(['Car']),
                'dimensions': np.array([[1., 1., 1.]]),
                'location': np.array([[0., 0., 0.5]]),
                'rotation': np.array([[0.9249, 0., 0., 0.3801]]),
                'score': np.array([1.])
            }
        ]

        correction_metric_result: Dict = evaluate_correction_metric(
            gt_annos, dt_annos, CORRECTION_METRIC_CONFIG)
        expected_result: Dict = {
            'Car': {
                'correction_time': 1.0,
                'fp': 0,
                'fn': 0,
                'n_translation_errors': 0,
                'n_rotation_errors': 0,
                'n_scale_errors': 0,
                'n_translation_and_rotation_errors': 1,
                'n_rotation_and_scale_errors': 0,
                'n_translation_and_scale_errors': 0,
                'n_translation_and_rotation_and_scale_errors': 0,
                'n_classification_errors': 0,
                'tp': 1,
                'total_gt': 1,
                'total_dt': 1,
                'n_no_positional_errors': 0
            },
            'total_correction_time': 1.0,
            'correction_time_by_frame': 1.0,
            'correction_time_by_object': 1.0,
            'baseline_correction_time': 1.0,
            'correction_acceleration_ratio': 0.0
        }
        dict_comparer(correction_metric_result, expected_result)

    def test_single_object_rotation_and_scale_error(self):
        gt_annos: List[Dict[str, np.ndarray]] = [
            {
                'name': np.array(['Car']),
                'dimensions': np.array([[1., 1., 1.]]),
                'location': np.array([[0., 0., 0.]]),
                'rotation': np.array([[1., 0., 0., 0.]]),
                'score': np.array([0.])
            }
        ]
        dt_annos: List[Dict[str, np.ndarray]] = [
            {
                'name': np.array(['Car']),
                'dimensions': np.array([[4., 4., 4.]]),
                'location': np.array([[0., 0., 0.0]]),
                'rotation': np.array([[0.9249, 0., 0., 0.3801]]),
                'score': np.array([1.])
            }
        ]

        correction_metric_result: Dict = evaluate_correction_metric(
            gt_annos, dt_annos, CORRECTION_METRIC_CONFIG)
        expected_result: Dict = {
            'Car': {
                'correction_time': 1.0,
                'fp': 0,
                'fn': 0,
                'n_translation_errors': 0,
                'n_rotation_errors': 0,
                'n_scale_errors': 0,
                'n_translation_and_rotation_errors': 0,
                'n_rotation_and_scale_errors': 1,
                'n_translation_and_scale_errors': 0,
                'n_translation_and_rotation_and_scale_errors': 0,
                'n_classification_errors': 0,
                'tp': 1,
                'total_gt': 1,
                'total_dt': 1,
                'n_no_positional_errors': 0
            },
            'total_correction_time': 1.0,
            'correction_time_by_frame': 1.0,
            'correction_time_by_object': 1.0,
            'baseline_correction_time': 1.0,
            'correction_acceleration_ratio': 0.0
        }
        dict_comparer(correction_metric_result, expected_result)

    def test_single_object_translation_and_scale_error(self):
        gt_annos: List[Dict[str, np.ndarray]] = [
            {
                'name': np.array(['Car']),
                'dimensions': np.array([[1., 1., 1.]]),
                'location': np.array([[0., 0., 0.]]),
                'rotation': np.array([[1., 0., 0., 0.]]),
                'score': np.array([0.])
            }
        ]
        dt_annos: List[Dict[str, np.ndarray]] = [
            {
                'name': np.array(['Car']),
                'dimensions': np.array([[4., 4., 4.]]),
                'location': np.array([[0., 0., 0.5]]),
                'rotation': np.array([[1., 0., 0., 0.]]),
                'score': np.array([1.])
            }
        ]

        correction_metric_result: Dict = evaluate_correction_metric(
            gt_annos, dt_annos, CORRECTION_METRIC_CONFIG)
        expected_result: Dict = {
            'Car': {
                'correction_time': 1.0,
                'fp': 0,
                'fn': 0,
                'n_translation_errors': 0,
                'n_rotation_errors': 0,
                'n_scale_errors': 0,
                'n_translation_and_rotation_errors': 0,
                'n_rotation_and_scale_errors': 0,
                'n_translation_and_scale_errors': 1,
                'n_translation_and_rotation_and_scale_errors': 0,
                'n_classification_errors': 0,
                'tp': 1,
                'total_gt': 1,
                'total_dt': 1,
                'n_no_positional_errors': 0
            },
            'total_correction_time': 1.0,
            'correction_time_by_frame': 1.0,
            'correction_time_by_object': 1.0,
            'baseline_correction_time': 1.0,
            'correction_acceleration_ratio': 0.0
        }
        dict_comparer(correction_metric_result, expected_result)

    def test_single_object_translation_and_rotation_and_scale_error(self):
        gt_annos: List[Dict[str, np.ndarray]] = [
            {
                'name': np.array(['Car']),
                'dimensions': np.array([[1., 1., 1.]]),
                'location': np.array([[0., 0., 0.]]),
                'rotation': np.array([[1., 0., 0., 0.]]),
                'score': np.array([0.])
            }
        ]
        dt_annos: List[Dict[str, np.ndarray]] = [
            {
                'name': np.array(['Car']),
                'dimensions': np.array([[4., 4., 4.]]),
                'location': np.array([[0., 0., 0.5]]),
                'rotation': np.array([[0.9249, 0., 0., 0.3801]]),
                'score': np.array([1.])
            }
        ]

        correction_metric_result: Dict = evaluate_correction_metric(
            gt_annos, dt_annos, CORRECTION_METRIC_CONFIG)
        expected_result: Dict = {
            'Car': {
                'correction_time': 1.0,
                'fp': 0,
                'fn': 0,
                'n_translation_errors': 0,
                'n_rotation_errors': 0,
                'n_scale_errors': 0,
                'n_translation_and_rotation_errors': 0,
                'n_rotation_and_scale_errors': 0,
                'n_translation_and_scale_errors': 0,
                'n_translation_and_rotation_and_scale_errors': 1,
                'n_classification_errors': 0,
                'tp': 1,
                'total_gt': 1,
                'total_dt': 1,
                'n_no_positional_errors': 0
            },
            'total_correction_time': 1.0,
            'correction_time_by_frame': 1.0,
            'correction_time_by_object': 1.0,
            'baseline_correction_time': 1.0,
            'correction_acceleration_ratio': 0.0
        }
        dict_comparer(correction_metric_result, expected_result)

    def test_single_object_translation_and_rotation_and_scale_error_different_config(self):
        gt_annos: List[Dict[str, np.ndarray]] = [
            {
                'name': np.array(['Car']),
                'dimensions': np.array([[1., 1., 1.]]),
                'location': np.array([[0., 0., 0.]]),
                'rotation': np.array([[1., 0., 0., 0.]]),
                'score': np.array([0.])
            }
        ]
        dt_annos: List[Dict[str, np.ndarray]] = [
            {
                'name': np.array(['Car']),
                'dimensions': np.array([[4., 4., 4.]]),
                'location': np.array([[0., 0., 0.5]]),
                'rotation': np.array([[0.9249, 0., 0., 0.3801]]),
                'score': np.array([1.])
            }
        ]

        correction_metric_config = CorrectionMetricConfig(
            average_times_to_correct=AverageTimesToCorrect(
                false_positive=1.0,
                false_negative=1.0,
                translation=1.0,
                rotation=1.0,
                scale=1.0,
                translation_and_rotation=1.0,
                rotation_and_scale=1.0,
                translation_and_scale=1.0,
                translation_and_rotation_and_scale=100.0, # override
                classification=1.0),
            thresholds={
                'Car': Thresholds(
                    translation=0.1,
                    rotation=0.1,
                    scale=0.1)
            },
            classes=['Car'],
            iou_ths=0.0,
            score_ths=0.5
        )

        correction_metric_result: Dict = evaluate_correction_metric(
            gt_annos, dt_annos, correction_metric_config)
        expected_result: Dict = {
            'Car': {
                'correction_time': 100.0,
                'fp': 0,
                'fn': 0,
                'n_translation_errors': 0,
                'n_rotation_errors': 0,
                'n_scale_errors': 0,
                'n_translation_and_rotation_errors': 0,
                'n_rotation_and_scale_errors': 0,
                'n_translation_and_scale_errors': 0,
                'n_translation_and_rotation_and_scale_errors': 1,
                'n_classification_errors': 0,
                'tp': 1,
                'total_gt': 1,
                'total_dt': 1,
                'n_no_positional_errors': 0
            },
            'total_correction_time': 100.0,
            'correction_time_by_frame': 100.0,
            'correction_time_by_object': 100.0,
            'baseline_correction_time': 1.0,
            'correction_acceleration_ratio': -99.0
        }
        dict_comparer(correction_metric_result, expected_result)

    def test_single_object_translation_and_rotation_and_scale_error_plus_classification_error(self):
        gt_annos: List[Dict[str, np.ndarray]] = [
            {
                'name': np.array(['Car']),
                'dimensions': np.array([[1., 1., 1.]]),
                'location': np.array([[0., 0., 0.]]),
                'rotation': np.array([[1., 0., 0., 0.]]),
                'score': np.array([0.])
            }
        ]
        dt_annos: List[Dict[str, np.ndarray]] = [
            {
                'name': np.array(['Ignored']),
                'dimensions': np.array([[4., 4., 4.]]),
                'location': np.array([[0., 0., 0.5]]),
                'rotation': np.array([[0.9249, 0., 0., 0.3801]]),
                'score': np.array([1.])
            }
        ]

        correction_metric_result: Dict = evaluate_correction_metric(
            gt_annos, dt_annos, CORRECTION_METRIC_CONFIG)
        expected_result: Dict = {
            'Car': {
                'correction_time': 2.0,
                'fp': 0,
                'fn': 0,
                'n_translation_errors': 0,
                'n_rotation_errors': 0,
                'n_scale_errors': 0,
                'n_translation_and_rotation_errors': 0,
                'n_rotation_and_scale_errors': 0,
                'n_translation_and_scale_errors': 0,
                'n_translation_and_rotation_and_scale_errors': 1,
                'n_classification_errors': 1,
                'tp': 1,
                'total_gt': 1,
                'total_dt': 0,
                'n_no_positional_errors': 0
            },
            'total_correction_time': 2.0,
            'correction_time_by_frame': 2.0,
            'correction_time_by_object': 2.0,
            'baseline_correction_time': 1.0,
            'correction_acceleration_ratio': -1.0
        }
        dict_comparer(correction_metric_result, expected_result)

    def test_single_object_translation_and_rotation_and_scale_error_additional_empty_frame(self):
        gt_annos: List[Dict[str, np.ndarray]] = [
            {
                'name': np.array(['Car']),
                'dimensions': np.array([[1., 1., 1.]]),
                'location': np.array([[0., 0., 0.]]),
                'rotation': np.array([[1., 0., 0., 0.]]),
                'score': np.array([0.])
            },
            {
                'name': np.array([]),
                'dimensions': np.array([]),
                'location': np.array([]),
                'rotation': np.array([]),
                'score': np.array([])
            }
        ]
        dt_annos: List[Dict[str, np.ndarray]] = [
            {
                'name': np.array(['Car']),
                'dimensions': np.array([[4., 4., 4.]]),
                'location': np.array([[0., 0., 0.5]]),
                'rotation': np.array([[0.9249, 0., 0., 0.3801]]),
                'score': np.array([1.])
            },
            {
                'name': np.array([]),
                'dimensions': np.array([]),
                'location': np.array([]),
                'rotation': np.array([]),
                'score': np.array([])
            }
        ]

        correction_metric_result: Dict = evaluate_correction_metric(
            gt_annos, dt_annos, CORRECTION_METRIC_CONFIG)
        expected_result: Dict = {
            'Car': {
                'correction_time': 1.0,
                'fp': 0,
                'fn': 0,
                'n_translation_errors': 0,
                'n_rotation_errors': 0,
                'n_scale_errors': 0,
                'n_translation_and_rotation_errors': 0,
                'n_rotation_and_scale_errors': 0,
                'n_translation_and_scale_errors': 0,
                'n_translation_and_rotation_and_scale_errors': 1,
                'n_classification_errors': 0,
                'tp': 1,
                'total_gt': 1,
                'total_dt': 1,
                'n_no_positional_errors': 0
            },
            'total_correction_time': 1.0,
            'correction_time_by_frame': 0.5,
            'correction_time_by_object': 1.0,
            'baseline_correction_time': 1.0,
            'correction_acceleration_ratio': 0.0
        }
        dict_comparer(correction_metric_result, expected_result)

    def test_multiple_objects_same_error(self):
        gt_annos: List[Dict[str, np.ndarray]] = [
            {
                'name': np.array(['Car', 'Car']),
                'dimensions': np.array([[1., 1., 1.], [1., 1., 1.]]),
                'location': np.array([[0., 0., 0.], [10., 10., 10.]]),
                'rotation': np.array([[1., 0., 0., 0.], [1., 0., 0., 0.]]),
                'score': np.array([0., 0.])
            }
        ]
        dt_annos: List[Dict[str, np.ndarray]] = [
            {
                'name': np.array(['Car', 'Car']),
                'dimensions': np.array([[1., 1., 1.], [1., 1., 1.]]),
                'location': np.array([[0., 0., 0.5], [10., 10., 10.5]]),
                'rotation': np.array([[1., 0., 0., 0.], [1., 0., 0., 0.]]),
                'score': np.array([1., 1.])
            }
        ]

        correction_metric_result: Dict = evaluate_correction_metric(
            gt_annos, dt_annos, CORRECTION_METRIC_CONFIG)
        expected_result: Dict = {
            'Car': {
                'correction_time': 2.0,
                'fp': 0,
                'fn': 0,
                'n_translation_errors': 2,
                'n_rotation_errors': 0,
                'n_scale_errors': 0,
                'n_translation_and_rotation_errors': 0,
                'n_rotation_and_scale_errors': 0,
                'n_translation_and_scale_errors': 0,
                'n_translation_and_rotation_and_scale_errors': 0,
                'n_classification_errors': 0,
                'tp': 2,
                'total_gt': 2,
                'total_dt': 2,
                'n_no_positional_errors': 0
            },
            'total_correction_time': 2.0,
            'correction_time_by_frame': 2.0,
            'correction_time_by_object': 1.0,
            'baseline_correction_time': 2.0,
            'correction_acceleration_ratio': 0.0
        }
        dict_comparer(correction_metric_result, expected_result)

    def test_multiple_objects_single_error(self):
        gt_annos: List[Dict[str, np.ndarray]] = [
            {
                'name': np.array(['Car', 'Car']),
                'dimensions': np.array([[1., 1., 1.], [1., 1., 1.]]),
                'location': np.array([[0., 0., 0.], [10., 10., 10.]]),
                'rotation': np.array([[1., 0., 0., 0.], [1., 0., 0., 0.]]),
                'score': np.array([0., 0.])
            }
        ]
        dt_annos: List[Dict[str, np.ndarray]] = [
            {
                'name': np.array(['Car', 'Car']),
                'dimensions': np.array([[1., 1., 1.], [1., 1., 1.]]),
                'location': np.array([[0., 0., 0.], [10., 10., 10.5]]),
                'rotation': np.array([[1., 0., 0., 0.], [1., 0., 0., 0.]]),
                'score': np.array([1., 1.])
            }
        ]

        correction_metric_result: Dict = evaluate_correction_metric(
            gt_annos, dt_annos, CORRECTION_METRIC_CONFIG)
        expected_result: Dict = {
            'Car': {
                'correction_time': 1.0,
                'fp': 0,
                'fn': 0,
                'n_translation_errors': 1,
                'n_rotation_errors': 0,
                'n_scale_errors': 0,
                'n_translation_and_rotation_errors': 0,
                'n_rotation_and_scale_errors': 0,
                'n_translation_and_scale_errors': 0,
                'n_translation_and_rotation_and_scale_errors': 0,
                'n_classification_errors': 0,
                'tp': 2,
                'total_gt': 2,
                'total_dt': 2,
                'n_no_positional_errors': 1
            },
            'total_correction_time': 1.0,
            'correction_time_by_frame': 1.0,
            'correction_time_by_object': 0.5,
            'baseline_correction_time': 2.0,
            'correction_acceleration_ratio': 0.5
        }
        dict_comparer(correction_metric_result, expected_result)

    def test_multiple_objects_different_classes_different_thresholds(self):
        gt_annos: List[Dict[str, np.ndarray]] = [
            {
                'name': np.array(['Car', 'Pedestrian']),
                'dimensions': np.array([[1., 1., 1.], [1., 1., 1.]]),
                'location': np.array([[0., 0., 0.], [10., 10., 10.]]),
                'rotation': np.array([[1., 0., 0., 0.], [1., 0., 0., 0.]]),
                'score': np.array([0., 0.])
            }
        ]
        dt_annos: List[Dict[str, np.ndarray]] = [
            {
                'name': np.array(['Car', 'Pedestrian']),
                'dimensions': np.array([[1., 1., 1.], [1., 1., 1.]]),
                'location': np.array([[0., 0., 0.], [10., 10., 10.]]),
                'rotation': np.array([[0.9249, 0., 0., 0.3801], [0.9249, 0., 0., 0.3801]]),
                'score': np.array([1., 1.])
            }
        ]

        correction_metric_config = CorrectionMetricConfig(
            average_times_to_correct=AverageTimesToCorrect(
                false_positive=1.0,
                false_negative=1.0,
                translation=1.0,
                rotation=1.0,
                scale=1.0,
                translation_and_rotation=1.0,
                rotation_and_scale=1.0,
                translation_and_scale=1.0,
                translation_and_rotation_and_scale=1.0,
                classification=1.0),
            thresholds={
                'Car': Thresholds(
                    translation=0.1,
                    rotation=0.1,
                    scale=0.1),
                'Pedestrian': Thresholds(
                    translation=0.1,
                    rotation=1.0, # override
                    scale=0.1)
            },
            classes=['Car', 'Pedestrian'],
            iou_ths=0.0,
            score_ths=0.5
        )

        correction_metric_result: Dict = evaluate_correction_metric(
            gt_annos, dt_annos, correction_metric_config)
        expected_result: Dict = {
            'Car': {
                'correction_time': 1.0,
                'fp': 0,
                'fn': 0,
                'n_translation_errors': 0,
                'n_rotation_errors': 1,
                'n_scale_errors': 0,
                'n_translation_and_rotation_errors': 0,
                'n_rotation_and_scale_errors': 0,
                'n_translation_and_scale_errors': 0,
                'n_translation_and_rotation_and_scale_errors': 0,
                'n_classification_errors': 0,
                'tp': 1,
                'total_gt': 1,
                'total_dt': 1,
                'n_no_positional_errors': 0
            },
            'Pedestrian': {
                'correction_time': 0.0,
                'fp': 0,
                'fn': 0,
                'n_translation_errors': 0,
                'n_rotation_errors': 0,
                'n_scale_errors': 0,
                'n_translation_and_rotation_errors': 0,
                'n_rotation_and_scale_errors': 0,
                'n_translation_and_scale_errors': 0,
                'n_translation_and_rotation_and_scale_errors': 0,
                'n_classification_errors': 0,
                'tp': 1,
                'total_gt': 1,
                'total_dt': 1,
                'n_no_positional_errors': 1
            },
            'total_correction_time': 1.0,
            'correction_time_by_frame': 1.0,
            'correction_time_by_object': 0.5,
            'baseline_correction_time': 2.0,
            'correction_acceleration_ratio': 0.5
        }
        dict_comparer(correction_metric_result, expected_result)

    def test_multiple_objects_different_errors(self):
        gt_annos: List[Dict[str, np.ndarray]] = [
            {
                'name': np.array(['Car', 'Car']),
                'dimensions': np.array([[1., 1., 1.], [1., 1., 1.]]),
                'location': np.array([[0., 0., 0.], [10., 10., 10.]]),
                'rotation': np.array([[1., 0., 0., 0.], [1., 0., 0., 0.]]),
                'score': np.array([0., 0.])
            }
        ]
        dt_annos: List[Dict[str, np.ndarray]] = [
            {
                'name': np.array(['Car', 'Car']),
                'dimensions': np.array([[1., 1., 1.], [1., 1., 1.]]),
                'location': np.array([[0., 0., 0.5], [10., 10., 10.]]),
                'rotation': np.array([[1., 0., 0., 0.], [0.9249, 0., 0., 0.3801]]),
                'score': np.array([1., 1.])
            }
        ]

        correction_metric_result: Dict = evaluate_correction_metric(
            gt_annos, dt_annos, CORRECTION_METRIC_CONFIG)
        expected_result: Dict = {
            'Car': {
                'correction_time': 2.0,
                'fp': 0,
                'fn': 0,
                'n_translation_errors': 1,
                'n_rotation_errors': 1,
                'n_scale_errors': 0,
                'n_translation_and_rotation_errors': 0,
                'n_rotation_and_scale_errors': 0,
                'n_translation_and_scale_errors': 0,
                'n_translation_and_rotation_and_scale_errors': 0,
                'n_classification_errors': 0,
                'tp': 2,
                'total_gt': 2,
                'total_dt': 2,
                'n_no_positional_errors': 0
            },
            'total_correction_time': 2.0,
            'correction_time_by_frame': 2.0,
            'correction_time_by_object': 1.0,
            'baseline_correction_time': 2.0,
            'correction_acceleration_ratio': 0.0
        }
        dict_comparer(correction_metric_result, expected_result)

    def test_multiple_objects_different_errors_different_config(self):
        gt_annos: List[Dict[str, np.ndarray]] = [
            {
                'name': np.array(['Car', 'Car']),
                'dimensions': np.array([[1., 1., 1.], [1., 1., 1.]]),
                'location': np.array([[0., 0., 0.], [10., 10., 10.]]),
                'rotation': np.array([[1., 0., 0., 0.], [1., 0., 0., 0.]]),
                'score': np.array([0., 0.])
            }
        ]
        dt_annos: List[Dict[str, np.ndarray]] = [
            {
                'name': np.array(['Car', 'Car']),
                'dimensions': np.array([[1., 1., 1.], [1., 1., 1.]]),
                'location': np.array([[0., 0., 0.5], [10., 10., 10.]]),
                'rotation': np.array([[1., 0., 0., 0.], [0.9249, 0., 0., 0.3801]]),
                'score': np.array([1., 1.])
            }
        ]

        correction_metric_config = CorrectionMetricConfig(
            average_times_to_correct=AverageTimesToCorrect(
                false_positive=1.0,
                false_negative=1.0,
                translation=10.0, # override
                rotation=20.0, # override
                scale=1.0,
                translation_and_rotation=1.0,
                rotation_and_scale=1.0,
                translation_and_scale=1.0,
                translation_and_rotation_and_scale=1.0,
                classification=1.0),
            thresholds={
                'Car': Thresholds(
                    translation=0.1,
                    rotation=0.1,
                    scale=0.1)
            },
            classes=['Car'],
            iou_ths=0.0,
            score_ths=0.5
        )

        correction_metric_result: Dict = evaluate_correction_metric(
            gt_annos, dt_annos, correction_metric_config)
        expected_result: Dict = {
            'Car': {
                'correction_time': 30.0,
                'fp': 0,
                'fn': 0,
                'n_translation_errors': 1,
                'n_rotation_errors': 1,
                'n_scale_errors': 0,
                'n_translation_and_rotation_errors': 0,
                'n_rotation_and_scale_errors': 0,
                'n_translation_and_scale_errors': 0,
                'n_translation_and_rotation_and_scale_errors': 0,
                'n_classification_errors': 0,
                'tp': 2,
                'total_gt': 2,
                'total_dt': 2,
                'n_no_positional_errors': 0
            },
            'total_correction_time': 30.0,
            'correction_time_by_frame': 30.0,
            'correction_time_by_object': 15.0,
            'baseline_correction_time': 2.0,
            'correction_acceleration_ratio': -14.0
        }
        dict_comparer(correction_metric_result, expected_result)

    def test_multiple_objects_same_complex_error(self):
        gt_annos: List[Dict[str, np.ndarray]] = [
            {
                'name': np.array(['Car', 'Car']),
                'dimensions': np.array([[1., 1., 1.], [1., 1., 1.]]),
                'location': np.array([[0., 0., 0.], [10., 10., 10.]]),
                'rotation': np.array([[1., 0., 0., 0.], [1., 0., 0., 0.]]),
                'score': np.array([0., 0.])
            }
        ]
        dt_annos: List[Dict[str, np.ndarray]] = [
            {
                'name': np.array(['Car', 'Car']),
                'dimensions': np.array([[1., 1., 1.], [1., 1., 1.]]),
                'location': np.array([[0., 0., 0.5], [10., 10., 10.5]]),
                'rotation': np.array([[0.9249, 0., 0., 0.3801], [0.9249, 0., 0., 0.3801]]),
                'score': np.array([1., 1.])
            }
        ]

        correction_metric_result: Dict = evaluate_correction_metric(
            gt_annos, dt_annos, CORRECTION_METRIC_CONFIG)
        expected_result: Dict = {
            'Car': {
                'correction_time': 2.0,
                'fp': 0,
                'fn': 0,
                'n_translation_errors': 0,
                'n_rotation_errors': 0,
                'n_scale_errors': 0,
                'n_translation_and_rotation_errors': 2,
                'n_rotation_and_scale_errors': 0,
                'n_translation_and_scale_errors': 0,
                'n_translation_and_rotation_and_scale_errors': 0,
                'n_classification_errors': 0,
                'tp': 2,
                'total_gt': 2,
                'total_dt': 2,
                'n_no_positional_errors': 0
            },
            'total_correction_time': 2.0,
            'correction_time_by_frame': 2.0,
            'correction_time_by_object': 1.0,
            'baseline_correction_time': 2.0,
            'correction_acceleration_ratio': 0.0
        }
        dict_comparer(correction_metric_result, expected_result)

    def test_multiple_objects_different_complex_errors(self):
        gt_annos: List[Dict[str, np.ndarray]] = [
            {
                'name': np.array(['Car', 'Car']),
                'dimensions': np.array([[1., 1., 1.], [1., 1., 1.]]),
                'location': np.array([[0., 0., 0.], [10., 10., 10.]]),
                'rotation': np.array([[1., 0., 0., 0.], [1., 0., 0., 0.]]),
                'score': np.array([0., 0.])
            }
        ]
        dt_annos: List[Dict[str, np.ndarray]] = [
            {
                'name': np.array(['Car', 'Car']),
                'dimensions': np.array([[1., 1., 1.], [4., 4., 4.]]),
                'location': np.array([[0., 0., 0.5], [10., 10., 10.]]),
                'rotation': np.array([[0.9249, 0., 0., 0.3801], [0.9249, 0., 0., 0.3801]]),
                'score': np.array([1., 1.])
            }
        ]

        correction_metric_result: Dict = evaluate_correction_metric(
            gt_annos, dt_annos, CORRECTION_METRIC_CONFIG)
        expected_result: Dict = {
            'Car': {
                'correction_time': 2.0,
                'fp': 0,
                'fn': 0,
                'n_translation_errors': 0,
                'n_rotation_errors': 0,
                'n_scale_errors': 0,
                'n_translation_and_rotation_errors': 1,
                'n_rotation_and_scale_errors': 1,
                'n_translation_and_scale_errors': 0,
                'n_translation_and_rotation_and_scale_errors': 0,
                'n_classification_errors': 0,
                'tp': 2,
                'total_gt': 2,
                'total_dt': 2,
                'n_no_positional_errors': 0
            },
            'total_correction_time': 2.0,
            'correction_time_by_frame': 2.0,
            'correction_time_by_object': 1.0,
            'baseline_correction_time': 2.0,
            'correction_acceleration_ratio': 0.0
        }
        dict_comparer(correction_metric_result, expected_result)

    def test_multiple_objects_different_complex_errors_different_config(self):
        gt_annos: List[Dict[str, np.ndarray]] = [
            {
                'name': np.array(['Car', 'Car']),
                'dimensions': np.array([[1., 1., 1.], [1., 1., 1.]]),
                'location': np.array([[0., 0., 0.], [10., 10., 10.]]),
                'rotation': np.array([[1., 0., 0., 0.], [1., 0., 0., 0.]]),
                'score': np.array([0., 0.])
            }
        ]
        dt_annos: List[Dict[str, np.ndarray]] = [
            {
                'name': np.array(['Car', 'Car']),
                'dimensions': np.array([[1., 1., 1.], [4., 4., 4.]]),
                'location': np.array([[0., 0., 0.5], [10., 10., 10.]]),
                'rotation': np.array([[0.9249, 0., 0., 0.3801], [0.9249, 0., 0., 0.3801]]),
                'score': np.array([1., 1.])
            }
        ]

        correction_metric_config = CorrectionMetricConfig(
            average_times_to_correct=AverageTimesToCorrect(
                false_positive=1.0,
                false_negative=1.0,
                translation=1.0,
                rotation=1.0,
                scale=1.0,
                translation_and_rotation=100., # override
                rotation_and_scale=200., # override
                translation_and_scale=1.0,
                translation_and_rotation_and_scale=1.0,
                classification=1.0),
            thresholds={
                'Car': Thresholds(
                    translation=0.1,
                    rotation=0.1,
                    scale=0.1)
            },
            classes=['Car'],
            iou_ths=0.0,
            score_ths=0.5
        )

        correction_metric_result: Dict = evaluate_correction_metric(
            gt_annos, dt_annos, correction_metric_config)
        expected_result: Dict = {
            'Car': {
                'correction_time': 300.0,
                'fp': 0,
                'fn': 0,
                'n_translation_errors': 0,
                'n_rotation_errors': 0,
                'n_scale_errors': 0,
                'n_translation_and_rotation_errors': 1,
                'n_rotation_and_scale_errors': 1,
                'n_translation_and_scale_errors': 0,
                'n_translation_and_rotation_and_scale_errors': 0,
                'n_classification_errors': 0,
                'tp': 2,
                'total_gt': 2,
                'total_dt': 2,
                'n_no_positional_errors': 0
            },
            'total_correction_time': 300.0,
            'correction_time_by_frame': 300.0,
            'correction_time_by_object': 150.0,
            'baseline_correction_time': 2.0,
            'correction_acceleration_ratio': -149.0
        }
        dict_comparer(correction_metric_result, expected_result)

    def test_multiple_objects_from_different_frames_different_errors(self):
        gt_annos: List[Dict[str, np.ndarray]] = [
            {
                'name': np.array(['Car']),
                'dimensions': np.array([[1., 1., 1.]]),
                'location': np.array([[0., 0., 0.]]),
                'rotation': np.array([[1., 0., 0., 0.]]),
                'score': np.array([0.])
            },
            {
                'name': np.array(['Car']),
                'dimensions': np.array([[1., 1., 1.]]),
                'location': np.array([[10., 10., 10.]]),
                'rotation': np.array([[1., 0., 0., 0.]]),
                'score': np.array([0.])
            }
        ]
        dt_annos: List[Dict[str, np.ndarray]] = [
            {
                'name': np.array(['Car']),
                'dimensions': np.array([[1., 1., 1.]]),
                'location': np.array([[0., 0., 0.5]]),
                'rotation': np.array([[1., 0., 0., 0.]]),
                'score': np.array([1.])
            },
            {
                'name': np.array(['Car']),
                'dimensions': np.array([[1., 1., 1.]]),
                'location': np.array([[10., 10., 10.]]),
                'rotation': np.array([[0.9249, 0., 0., 0.3801]]),
                'score': np.array([1.])
            }
        ]

        correction_metric_result: Dict = evaluate_correction_metric(
            gt_annos, dt_annos, CORRECTION_METRIC_CONFIG)
        expected_result: Dict = {
            'Car': {
                'correction_time': 2.0,
                'fp': 0,
                'fn': 0,
                'n_translation_errors': 1,
                'n_rotation_errors': 1,
                'n_scale_errors': 0,
                'n_translation_and_rotation_errors': 0,
                'n_rotation_and_scale_errors': 0,
                'n_translation_and_scale_errors': 0,
                'n_translation_and_rotation_and_scale_errors': 0,
                'n_classification_errors': 0,
                'tp': 2,
                'total_gt': 2,
                'total_dt': 2,
                'n_no_positional_errors': 0
            },
            'total_correction_time': 2.0,
            'correction_time_by_frame': 1.0,
            'correction_time_by_object': 1.0,
            'baseline_correction_time': 2.0,
            'correction_acceleration_ratio': 0.0
        }
        dict_comparer(correction_metric_result, expected_result)

    def test_number_of_frames_for_ground_truth_and_detections_not_equal_fail(self):
        gt_annos: List[Dict[str, np.ndarray]] = [
            {}, {}
        ]
        dt_annos: List[Dict[str, np.ndarray]] = [
            {}
        ]

        with pytest.raises(AssertionError) as exc_info:
            evaluate_correction_metric(gt_annos, dt_annos, CORRECTION_METRIC_CONFIG)
        assert exc_info.type == AssertionError

    def test_multiple_objects_from_different_classes_different_errors(self):
        gt_annos: List[Dict[str, np.ndarray]] = [
            {
                'name': np.array(['Car', 'Pedestrian']),
                'dimensions': np.array([[1., 1., 1.], [1., 1., 1.]]),
                'location': np.array([[0., 0., 0.], [10., 10., 10.]]),
                'rotation': np.array([[1., 0., 0., 0.], [1., 0., 0., 0.]]),
                'score': np.array([0., 0.])
            }
        ]
        dt_annos: List[Dict[str, np.ndarray]] = [
            {
                'name': np.array(['Car', 'Pedestrian']),
                'dimensions': np.array([[1., 1., 1.], [1., 1., 1.]]),
                'location': np.array([[0., 0., 0.5], [10., 10., 10.]]),
                'rotation': np.array([[1., 0., 0., 0.], [0.9249, 0., 0., 0.3801]]),
                'score': np.array([1., 1.])
            }
        ]

        correction_metric_config = CorrectionMetricConfig(
            average_times_to_correct=AverageTimesToCorrect(
                false_positive=1.0,
                false_negative=1.0,
                translation=1.0,
                rotation=1.0,
                scale=1.0,
                translation_and_rotation=1.0,
                rotation_and_scale=1.0,
                translation_and_scale=1.0,
                translation_and_rotation_and_scale=1.0,
                classification=1.0),
            thresholds={
                'Car': Thresholds(
                    translation=0.1,
                    rotation=0.1,
                    scale=0.1),
                'Pedestrian': Thresholds(
                    translation=0.1,
                    rotation=0.1,
                    scale=0.1)
            },
            classes=['Car', 'Pedestrian'],
            iou_ths=0.0,
            score_ths=0.5
        )

        correction_metric_result: Dict = evaluate_correction_metric(
            gt_annos, dt_annos, correction_metric_config)
        expected_result: Dict = {
            'Car': {
                'correction_time': 1.0,
                'fp': 0,
                'fn': 0,
                'n_translation_errors': 1,
                'n_rotation_errors': 0,
                'n_scale_errors': 0,
                'n_translation_and_rotation_errors': 0,
                'n_rotation_and_scale_errors': 0,
                'n_translation_and_scale_errors': 0,
                'n_translation_and_rotation_and_scale_errors': 0,
                'n_classification_errors': 0,
                'tp': 1,
                'total_gt': 1,
                'total_dt': 1,
                'n_no_positional_errors': 0
            },
            'Pedestrian': {
                'correction_time': 1.0,
                'fp': 0,
                'fn': 0,
                'n_translation_errors': 0,
                'n_rotation_errors': 1,
                'n_scale_errors': 0,
                'n_translation_and_rotation_errors': 0,
                'n_rotation_and_scale_errors': 0,
                'n_translation_and_scale_errors': 0,
                'n_translation_and_rotation_and_scale_errors': 0,
                'n_classification_errors': 0,
                'tp': 1,
                'total_gt': 1,
                'total_dt': 1,
                'n_no_positional_errors': 0
            },
            'total_correction_time': 2.0,
            'correction_time_by_frame': 2.0,
            'correction_time_by_object': 1.0,
            'baseline_correction_time': 2.0,
            'correction_acceleration_ratio': 0.0
        }
        dict_comparer(correction_metric_result, expected_result)

class TestEvaluateCorrectionMetric():

    def test_evaluate_correction_metric(self):
        correction_metric_config = CorrectionMetricConfig(
            average_times_to_correct=AverageTimesToCorrect(
                false_positive=1.0,
                false_negative=1.0,
                translation=1.0,
                rotation=1.0,
                scale=1.0,
                translation_and_rotation=1.0,
                rotation_and_scale=1.0,
                translation_and_scale=1.0,
                translation_and_rotation_and_scale=1.0,
                classification=1.0),
            thresholds={
                'car': Thresholds(
                    translation=0.1,
                    rotation=0.1,
                    scale=0.1),
                'pedestrian': Thresholds(
                    translation=0.1,
                    rotation=0.1,
                    scale=0.1)
            },
            classes=['car', 'pedestrian'],
            iou_ths=0.0,
            score_ths=0.5
        )
        gt_json = test_data.gt_json
        dt_json = test_data.dt_json
        gt_annos, dt_annos = read_annotations_car(gt_json, dt_json)
        correction_metric_result: Dict = evaluate_correction_metric(gt_annos, dt_annos, correction_metric_config)
        expected_result: Dict = {
            'car': {
                'correction_time': 1.0,
                'fp': 0,
                'fn': 0,
                'n_translation_errors': 1,
                'n_rotation_errors': 0,
                'n_scale_errors': 0,
                'n_translation_and_rotation_errors': 0,
                'n_rotation_and_scale_errors': 0,
                'n_translation_and_scale_errors': 0,
                'n_translation_and_rotation_and_scale_errors': 0,
                'n_classification_errors': 0,
                'tp': 1,
                'total_gt': 1,
                'total_dt': 1,
                'n_no_positional_errors': 0
            },
            'pedestrian': {
                'correction_time': 1.0,
                'fp': 0,
                'fn': 0,
                'n_translation_errors': 0,
                'n_rotation_errors': 1,
                'n_scale_errors': 0,
                'n_translation_and_rotation_errors': 0,
                'n_rotation_and_scale_errors': 0,
                'n_translation_and_scale_errors': 0,
                'n_translation_and_rotation_and_scale_errors': 0,
                'n_classification_errors': 0,
                'tp': 1,
                'total_gt': 1,
                'total_dt': 1,
                'n_no_positional_errors': 0
            },
            'total_correction_time': 2.0,
            'correction_time_by_frame': 2.0,
            'correction_time_by_object': 1.0,
            'baseline_correction_time': 2.0,
            'correction_acceleration_ratio': 0.0
        }
        dict_comparer(correction_metric_result, expected_result)
