# pylint: disable=missing-module-docstring, missing-class-docstring
import pytest
import numpy as np
from numpy.testing import assert_almost_equal
from pyquaternion import Quaternion #type: ignore
from merge_annotations import get_sequences_by_model, get_frames_by_model, get_ious_pairs_to_calculate,\
    get_ious, get_ious_by_model, merge_all, get_passed_indices, filter_out_passed_indices, flatten,\
    set_intersection_left, filtered_indices_right, set_intersection_right, set_intersection_left_right,\
    merge_boxes_strategy, choose_one_indices_of_boxes, delete_duplicates, get_boxes_to_merge,\
    update_boxes_passed, format_rotation, get_rotation_difference, average_rotation, average_box,\
    merge_boxes, merge_with_strategy

class TestGetSequencesByModel():

    @pytest.mark.usefixtures('annotations', 'sequence_by_model')
    def test_get_first_sequence(self, annotations, sequence_by_model):
        result = get_sequences_by_model(annotations, 0)
        assert result == sequence_by_model

class TestGetFramesByModel():

    @pytest.mark.usefixtures('sequence_by_model', 'frame_by_model')
    def test_get_first_frame(self, sequence_by_model, frame_by_model):
        result = get_frames_by_model(sequence_by_model, 0)
        assert result == frame_by_model

class TestGetIousPairsToCalculate():

    @pytest.mark.usefixtures('strategy_list')
    def test_get_ious_pairs_to_calculate(self, strategy_list):
        result = get_ious_pairs_to_calculate(strategy_list)
        assert result == ['MODEL_1+MODEL_2', 'MODEL_1+MODEL_3', 'MODEL_3+MODEL_3']

class TestGetIous():

    @pytest.mark.usefixtures('frame_by_model', 'ious_by_model')
    def test_get_ious(self, frame_by_model, ious_by_model):
        result = get_ious(frame_by_model['MODEL_1'], frame_by_model['MODEL_2'])
        assert_almost_equal(result, ious_by_model['MODEL_1+MODEL_2'], decimal=2)
        result = get_ious(frame_by_model['MODEL_1'], frame_by_model['MODEL_3'])
        assert_almost_equal(result, ious_by_model['MODEL_1+MODEL_3'], decimal=2)
        result = get_ious(frame_by_model['MODEL_3'], frame_by_model['MODEL_3'])
        assert_almost_equal(result, ious_by_model['MODEL_3+MODEL_3'], decimal=2)

class TestGetIousByModel():

    @pytest.mark.usefixtures('frame_by_model', 'strategy_list', 'ious_by_model')
    def test_get_ious_by_model(self, frame_by_model, strategy_list, ious_by_model):
        ious_pairs_to_calculate = get_ious_pairs_to_calculate(strategy_list)
        result = get_ious_by_model(frame_by_model, ious_pairs_to_calculate)
        assert result.keys() == ious_by_model.keys()
        for key, ious_list in ious_by_model.items():
            assert_almost_equal(result[key], ious_list, decimal=2)

class TestMergeAll():

    @pytest.mark.usefixtures('frame_by_model', 'merge_all_frame')
    def test_merge_all(self, frame_by_model, merge_all_frame):
        result = merge_all(frame_by_model)
        assert result == merge_all_frame

class TestGetPassedIndices():

    @pytest.mark.usefixtures('ious_by_model')
    def test_get_passed_indices(self, ious_by_model):
        result = get_passed_indices(ious_by_model['MODEL_1+MODEL_2'], 0.5)
        assert result == {0: [(1, 0.53), (2, 0.56)]}
        result = get_passed_indices(ious_by_model['MODEL_1+MODEL_2'], 0.3)
        assert result == {0: [(1, 0.53), (2, 0.56)], 1: [(0, 0.33)], 2: [(0, 0.33)]}
        result = get_passed_indices(ious_by_model['MODEL_1+MODEL_2'], 0.6)
        assert len(result.keys()) == 0
        result = get_passed_indices(ious_by_model['MODEL_3+MODEL_3'], 0.99)
        assert result == {0: [(0, 1)], 1: [(1, 1)]}

class TestFilterOutPassedIndices():

    @pytest.mark.usefixtures('ious_by_model')
    def test_filter_out_passed_indices(self, ious_by_model):
        passed_indices = get_passed_indices(ious_by_model['MODEL_1+MODEL_2'], 0.5)
        result = filter_out_passed_indices(passed_indices, [], [2])
        assert result == {0: [(1, 0.53)]}
        result = filter_out_passed_indices(passed_indices, [], [2, 1])
        assert result == {}
        result = filter_out_passed_indices(passed_indices, [0], [])
        assert result == {}
        result = filter_out_passed_indices(passed_indices, [0], [1, 5])
        assert result == {}

        passed_indices = get_passed_indices(ious_by_model['MODEL_1+MODEL_2'], 0.3)
        result = filter_out_passed_indices(passed_indices, [], [0])
        assert result == {0: [(1, 0.53), (2, 0.56)]}
        passed_indices = get_passed_indices(ious_by_model['MODEL_3+MODEL_3'], 0.99)
        result = filter_out_passed_indices(passed_indices, [0, 1], [0, 1])
        assert result == {}

class TestFlatten():
    def test_flatten(self):
        result = flatten([0, 1, [2, 3, 4], 5])
        assert result == [0, 1, 2, 3, 4, 5]

class TestSetIntersectionLeft():
    def test_set_intersection_left(self):
        result_left, result_right = set_intersection_left({0: [(1, 0.56), (2, 0.56)], 1: [(0, 0.33)], 2: [(0, 0.33)]}, {1: [(0, 0.33)], 2: [(0, 0.33)], 3: [(10, 0.1)]})
        assert result_left == result_right == {1: [(0, 0.33)], 2: [(0, 0.33)]}
        result_left, result_right = set_intersection_left({}, {1: [(0, 0.33)], 2: [(0, 0.33)]})
        assert result_left == result_right == {}
        result_left, result_right = set_intersection_left({1: [(0, 0.33)], 2: [(0, 0.33)]}, {})
        assert result_left == result_right == {}
        result_left, result_right = set_intersection_left({1: [(0, 0.33)], 2: [(0, 0.33)]}, {3: [(24, 0.2)]})
        assert result_left == result_right == {}

class TestFilteredIndicesRight():
    def test_filtered_indices_right(self):
        result = filtered_indices_right([], {1: [(0, 0.33)], 2: [(0, 0.33)], 3: [(10, 0.1)]})
        assert result == {}
        result = filtered_indices_right([0], {1: [(0, 0.33)], 2: [(0, 0.33)], 3: [(10, 0.1)]})
        assert result == {1: [(0, 0.33)], 2: [(0, 0.33)]}
        result = filtered_indices_right([10], {1: [(0, 0.33)], 2: [(0, 0.33)], 3: [(10, 0.1)]})
        assert result == {3: [(10, 0.1)]}
        result = filtered_indices_right([10], {1: [(0, 0.33), (10, 0.2)], 2: [(0, 0.33)], 3: [(10, 0.1)]})
        assert result == {3: [(10, 0.1)], 1: [(10, 0.2)]}

class TestSetIntersectionRight():
    def test_set_intersection_right(self):
        result_left, result_right = set_intersection_right({0: [(1, 0.56), (2, 0.56)], 1: [(0, 0.33)], 2: [(0, 0.33)]}, {1: [(0, 0.33)], 2: [(0, 0.33)], 3: [(10, 0.1)]})
        assert result_left == result_right == {1: [(0, 0.33)], 2: [(0, 0.33)]}
        result_left, result_right = set_intersection_right({10: [(1, 0.56), (2, 0.56)], 8: [(0, 0.33)], 4: [(0, 0.33)]}, {1: [(0, 0.33)], 2: [(1, 0.33)], 3: [(10, 0.1)]})
        assert result_left == {10: [(1, 0.56)], 8: [(0, 0.33)], 4: [(0, 0.33)]}
        assert result_right == {1: [(0, 0.33)], 2: [(1, 0.33)]}

class TestSetIntersectionLeftRight():
    def test_set_intersection_left_right(self):
        result_left, result_right = set_intersection_left_right({0: [(1, 0.56), (2, 0.56)], 1: [(10, 0.33)], 2: [(0, 0.33)]}, {1: [(0, 0.33)], 2: [(0, 0.33)], 3: [(10, 0.1)]})
        assert result_left == {0: [(1, 0.56), (2, 0.56)]}
        assert result_right == {1: [(0, 0.33)], 2: [(0, 0.33)]}
        result_left, result_right = set_intersection_left_right({0: [(1, 0.56), (2, 0.56)], 1: [(10, 0.33)], 2: [(0, 0.33)]}, {1: [(0, 0.33)], 2: [(0, 0.33)], 3: [(0, 0.5), (10, 0.1), (1, 0.2)], 4:[(10, 0.6)]})
        assert result_left == {0: [(1, 0.56), (2, 0.56)], 1: [(10, 0.33)]}
        assert result_right == {1: [(0, 0.33)], 2: [(0, 0.33)], 3: [(0, 0.5), (1, 0.2)]}

class TestMergeBoxesStrategy():
    @pytest.mark.usefixtures('indices_of_boxes_strategy_passed_list', 'indices_of_boxes_strategy_passed_result_list')
    def test_merge_boxes_strategy(self, indices_of_boxes_strategy_passed_list, indices_of_boxes_strategy_passed_result_list):
        for index, indices_of_boxes_strategy_passed in enumerate(indices_of_boxes_strategy_passed_list):
            result = merge_boxes_strategy(indices_of_boxes_strategy_passed)
            assert result == indices_of_boxes_strategy_passed_result_list[index]

class TestChooseOneIndicesOfBoxes():
    @pytest.mark.usefixtures('ious_by_model')
    def test_choose_one_indices_of_boxes(self, ious_by_model):
        result = choose_one_indices_of_boxes([(1, 0), (1, 1)], ious_by_model, ['MODEL_1', 'MODEL_2'])
        assert result == (1, 0)
        result = choose_one_indices_of_boxes([(0, 1, 0), (0, 2, 0)], ious_by_model, ['MODEL_1', 'MODEL_2', 'MODEL_3'])
        assert result == (0, 2, 0)

class TestDeleteDuplicates():
    @pytest.mark.usefixtures('ious_by_model')
    def test_delete_duplicates(self, ious_by_model):
        result = delete_duplicates(['MODEL_1', 'MODEL_2'], {'MODEL_1': [(1, 3)], 'MODEL_2': []}, [(1, 0), (1, 1), (2, 4), (1, 2)], ious_by_model, False)
        assert result == [(2, 4), (1, 0)]
        result = delete_duplicates(['MODEL_1', 'MODEL_2', 'MODEL_3'], {'MODEL_1': [(1, 3)], 'MODEL_2': [], 'MODEL_3':[(0, 3)]}, [(1, 0, 0), (1, 1, 0), (2, 4, 1), (1, 2, 0)], ious_by_model, False)
        assert result == [(2, 4, 1), (1, 0, 0)]
        result = delete_duplicates(['MODEL_1', 'MODEL_2', 'MODEL_3'], {'MODEL_1': [(1, 2)], 'MODEL_2': [], 'MODEL_3':[(0, 3)]}, [(1, 0, 0), (1, 1, 0), (2, 2, 0)], ious_by_model, False)
        assert result == [(1, 0, 0)]

class TestGetBoxesToMerge():
    @pytest.mark.usefixtures('indices_of_boxes_strategy_passed_list', 'ious_by_model_list')
    def test_indices_of_boxes_strategy_passed_list(self, indices_of_boxes_strategy_passed_list, ious_by_model_list):
        for index, indices_of_boxes_strategy_passed in enumerate(indices_of_boxes_strategy_passed_list):
            ious_by_model = ious_by_model_list[index]
            for models_name, indices_boxes in indices_of_boxes_strategy_passed.items():
                ious_of_models = ious_by_model[models_name]
                for left_index_box, list_right_indices in indices_boxes.items():
                    for right_box_index, iou in list_right_indices:
                        assert ious_of_models[left_index_box][right_box_index] == iou

    @pytest.mark.usefixtures('indices_of_boxes_strategy_passed_list', 'ious_by_model_list', 'order_of_models_list', 'indices_of_boxes_to_merge_list')
    def test_get_boxes_to_merge(self, indices_of_boxes_strategy_passed_list, ious_by_model_list, order_of_models_list, indices_of_boxes_to_merge_list):
        for index, indices_of_boxes_strategy_passed in enumerate(indices_of_boxes_strategy_passed_list):
            order_of_models, indices_of_boxes_to_merge = get_boxes_to_merge(indices_of_boxes_strategy_passed, ious_by_model_list[index], False)
            assert order_of_models == order_of_models_list[index]
            assert indices_of_boxes_to_merge == indices_of_boxes_to_merge_list[index]

class TestUpdateBoxesPassed():
    @pytest.mark.usefixtures('order_of_models_list', 'indices_of_boxes_to_merge_list', 'indices_of_boxes_passed_list')
    def test_update_boxes_passed(self, order_of_models_list, indices_of_boxes_to_merge_list, indices_of_boxes_passed_list):
        for index, indices_of_boxes_to_merge in enumerate(indices_of_boxes_to_merge_list):
            order_of_models = order_of_models_list[index]
            indices_of_boxes_passed = {}
            for model_name in order_of_models:
                indices_of_boxes_passed[model_name] = []
            result = update_boxes_passed(indices_of_boxes_passed, order_of_models, indices_of_boxes_to_merge)
            assert result == indices_of_boxes_passed_list[index]

        with pytest.raises(Exception) as e:
            indices_of_boxes_passed_list[0][order_of_models[0]] = [0]
            update_boxes_passed(indices_of_boxes_passed_list[0], order_of_models_list[0], indices_of_boxes_to_merge_list[0])

class TestFormatRotation():
    def test_format_rotation(self):
        seq = np.arange(0, 360 * np.pi / 180, 10 * np.pi / 180)
        for i in seq:
            rotation = Quaternion(axis=[0, 0, 1], radians=i)
            result = format_rotation(rotation)
            assert result >=0
            assert result < np.pi
        for i in seq:
            rotation = Quaternion(axis=[0, 0, 1], radians=-i)
            result = format_rotation(rotation)
            assert result >=0
            assert result < np.pi
        assert_almost_equal(format_rotation(Quaternion(axis=[0, 0, 1], radians=np.pi/2.)), np.pi/2., decimal=3)
        assert_almost_equal(format_rotation(Quaternion(axis=[0, 0, 1], radians=-np.pi/2.)), np.pi/2., decimal=3)

class TestGetRotationDifference():
    def test_get_rotation_difference(self):
        assert_almost_equal(get_rotation_difference(0, np.pi/2.), np.pi/2., decimal=3)
        assert_almost_equal(get_rotation_difference(0, np.pi/2. + 0.017), np.pi/2. - 0.017, decimal=3)
        assert_almost_equal(get_rotation_difference(0, np.pi - 0.017), 0.017, decimal=3)
        assert_almost_equal(get_rotation_difference(np.pi/2., 0), np.pi/2., decimal=3)
        assert_almost_equal(get_rotation_difference(np.pi/2. + 0.017, 0), np.pi/2. - 0.017, decimal=3)
        assert_almost_equal(get_rotation_difference(np.pi - 0.017, 0), 0.017, decimal=3)

class TestAverageRotation():
    def test_average_rotation(self):
        assert_almost_equal(average_rotation([0, np.pi/2.]), np.pi/4., decimal=3)
        assert_almost_equal(average_rotation([np.pi/2., 0]), np.pi/4., decimal=3)
        assert_almost_equal(average_rotation([0, np.pi/2., np.pi/2.]), np.pi/4., decimal=3)
        assert_almost_equal(average_rotation([np.pi/2., np.pi/2., 0]), np.pi/4., decimal=3)
        assert_almost_equal(average_rotation([0, np.pi - 0.01]), np.pi - 0.008, decimal=3) # 0.008 is 0.5 degrees
        assert_almost_equal(average_rotation([0, np.pi/2. - 0.01]), np.pi/4. - 0.008, decimal=3)
        assert_almost_equal(average_rotation([0, np.pi - 0.01, np.pi/2. - 0.01]), np.pi/4. - 0.008, decimal=3)
        assert_almost_equal(average_rotation([0, np.pi/4. , np.pi/2.]), np.pi/4., decimal=3)
        assert_almost_equal(average_rotation([0, 3 * np.pi/4. , np.pi/2.]), 3 * np.pi/4., decimal=3)

class TestAverageBox():
    def test_average_box(self):
        rotation = Quaternion(axis=[0, 0, 1], radians=0)
        box_1 = {'translation': [10., 10., 0.],'size': [3, 4, 2], 'rotation': [rotation[0], rotation[1], rotation[2], rotation[3]], 'detection_name':'car'}
        rotation = Quaternion(axis=[0, 0, 1], radians=np.pi/2.)
        box_2 = {'translation': [20., 16., 2.],'size': [4, 5, 3], 'rotation': [rotation[0], rotation[1], rotation[2], rotation[3]]}
        result = average_box([box_1, box_2])
        assert result['translation'] == [15.0, 13.0, 1.0]
        assert result['size'] == [3.5, 4.5, 2.5]
        assert_almost_equal(Quaternion(result['rotation']).yaw_pitch_roll[0], np.pi/4., decimal=3)
        rotation = Quaternion(axis=[0, 0, 1], radians=np.pi)
        box_3 = {'translation': [30., 4., 4.],'size': [3.5, 4.5, 2.5], 'rotation': [rotation[0], rotation[1], rotation[2], rotation[3]]}
        result = average_box([box_1, box_2, box_3])
        assert result['translation'] == [20.0, 10.0, 2.0]
        assert result['size'] == [3.5, 4.5, 2.5]
        assert_almost_equal(Quaternion(result['rotation']).yaw_pitch_roll[0], np.pi/4., decimal=3)

        rotation = Quaternion(axis=[0, 0, 1], radians=0)
        box_1 = {'translation': [10., 10., 0.],'size': [3, 4, 2], 'rotation': [rotation[0], rotation[1], rotation[2], rotation[3]], 'detection_name':'car'}
        rotation = Quaternion(axis=[0, 0, 1], radians=np.pi/2.)
        box_2 = {'translation': [20., 16., 2.],'size': [4, 5, 3], 'rotation': [rotation[0], rotation[1], rotation[2], rotation[3]]}
        rotation = Quaternion(axis=[0, 0, 1], radians=3 * np.pi/4.)
        box_3 = {'translation': [30., 4., 4.],'size': [3.5, 4.5, 2.5], 'rotation': [rotation[0], rotation[1], rotation[2], rotation[3]]}
        result = average_box([box_1, box_2, box_3])
        assert result['translation'] == [20.0, 10.0, 2.0]
        assert result['size'] == [3.5, 4.5, 2.5]
        assert_almost_equal(Quaternion(result['rotation']).yaw_pitch_roll[0], 3 * np.pi/4., decimal=3)

class TestMergeBoxes():
    @pytest.mark.usefixtures('frame_by_model')
    def test_merge_boxes(self, frame_by_model):
        result = merge_boxes([(0, 0), (1, 1), (2, 2)], ['MODEL_1', 'MODEL_1'], frame_by_model, False)
        assert result == frame_by_model['MODEL_1']['boxes']
        result = merge_boxes([(0, 0), (1, 1)], ['MODEL_1', 'MODEL_1'], frame_by_model, False)
        assert result == frame_by_model['MODEL_1']['boxes'][:2]
        result = merge_boxes([(0, 1), (1, 0)], ['MODEL_1', 'MODEL_2'], frame_by_model, False)
        assert result == [average_box([frame_by_model['MODEL_1']['boxes'][0], frame_by_model['MODEL_2']['boxes'][1]]),
                          average_box([frame_by_model['MODEL_1']['boxes'][1], frame_by_model['MODEL_2']['boxes'][0]])]
        result = merge_boxes([(0, 1, 1)], ['MODEL_1', 'MODEL_2', 'MODEL_3'], frame_by_model, False)
        assert result == [average_box([frame_by_model['MODEL_1']['boxes'][0], frame_by_model['MODEL_2']['boxes'][1], frame_by_model['MODEL_3']['boxes'][1]])]

class TestMergeWithStrategy():
    @pytest.mark.usefixtures('strategy_list', 'ious_by_model', 'frame_by_model')
    def test_merge_with_strategy(self, strategy_list, ious_by_model, frame_by_model):
        result = merge_with_strategy(strategy_list, ious_by_model, frame_by_model, False, False)
        box_1 = merge_boxes([(0, 2, 0)], ['MODEL_1', 'MODEL_2', 'MODEL_3'], frame_by_model, False)[0]
        box_2 = merge_boxes([(1, 1)], ['MODEL_3', 'MODEL_3'], frame_by_model, False)[0]
        assert result == [box_1, box_2]