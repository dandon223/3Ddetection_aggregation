# pylint: disable=missing-module-docstring, redefined-outer-name
import pytest
import numpy as np

BOX1 = {'translation': [1185., 438., 1.],
        'size': [1.92, 4.49, 1.60],
        'rotation': [-0.7549002692305175, 0.007293097360446562, -0.008942727560232594, -0.6557380741350674],
        'detection_name':'car'
        }
BOX11 = {'translation': [1186.1, 438., 1.],
        'size': [1.92, 4.49, 1.60],
        'rotation': [-0.7549002692305175, 0.007293097360446562, -0.008942727560232594, -0.6557380741350674],
        'detection_name':'car'
        }
BOX12 = {'translation': [1184., 438., 1.],
        'size': [1.92, 4.49, 1.60],
        'rotation': [-0.7549002692305175, 0.007293097360446562, -0.008942727560232594, -0.6557380741350674],
        'detection_name':'car'
        }
BOX13 = {'translation': [1185., 439., 1.],
        'size': [1.92, 4.49, 1.60],
        'rotation': [-0.7549002692305175, 0.007293097360446562, -0.008942727560232594, -0.6557380741350674],
        'detection_name':'car'
        }
BOX2 = {'translation': [1141., 411., 1.],
        'size': [1.94, 4.54, 1.55],
        'rotation': [-0.26438674070742374, 0.004904961427492157, 0.010445238088253915, 0.9643477016578512],
        'detection_name':'car'
        }
BOX21 = {'translation': [1142., 411., 1.],
        'size': [1.94, 4.54, 1.55],
        'rotation': [-0.26438674070742374, 0.004904961427492157, 0.010445238088253915, 0.9643477016578512],
        'detection_name':'car'
        }
BOX22 = {'translation': [1140., 411., 1.],
        'size': [1.94, 4.54, 1.55],
        'rotation': [-0.26438674070742374, 0.004904961427492157, 0.010445238088253915, 0.9643477016578512],
        'detection_name':'car'
        }
BOX3 = {'translation': [1145., 415., 1.],
        'size': [1.94, 4.54, 1.55],
        'rotation': [-0.26438674070742374, 0.004904961427492157, 0.010445238088253915, 0.9643477016578512],
        'detection_name':'car'
        }
@pytest.fixture
def annotations():
    return {'MODEL_1': [[{'frame_token': '0', 'boxes':[BOX1, BOX21, BOX22]}, {'frame_token': '1', 'boxes':[]}]],
            'MODEL_2': [[{'frame_token': '0', 'boxes':[BOX2, BOX11, BOX12]}, {'frame_token': '1', 'boxes':[]}]],
            'MODEL_3': [[{'frame_token': '0', 'boxes':[BOX13, BOX3]}, {'frame_token': '1', 'boxes':[]}]]}

@pytest.fixture
def sequence_by_model():
    return {'MODEL_1': [{'frame_token': '0', 'boxes':[BOX1, BOX21, BOX22]}, {'frame_token': '1', 'boxes':[]}],
            'MODEL_2': [{'frame_token': '0', 'boxes':[BOX2, BOX11, BOX12]}, {'frame_token': '1', 'boxes':[]}],
            'MODEL_3': [{'frame_token': '0', 'boxes':[BOX13, BOX3]}, {'frame_token': '1', 'boxes':[]}]}

@pytest.fixture
def frame_by_model():
    return {'MODEL_1': {'frame_token': '0', 'boxes':[BOX1, BOX21, BOX22]},
            'MODEL_2': {'frame_token': '0', 'boxes':[BOX2, BOX11, BOX12]},
            'MODEL_3': {'frame_token': '0', 'boxes':[BOX13, BOX3]}}

@pytest.fixture
def strategy_list():
    return [['MODEL_1', 'MODEL_2', 'MODEL_3', 0.3],
            ['MODEL_3']]

@pytest.fixture
def ious_by_model():
    return {'MODEL_1+MODEL_2': [[0.0, 0.53, 0.56], [0.33, 0.0, 0.0], [0.33, 0.0, 0.0]], 'MODEL_1+MODEL_3': [[0.30, 0.0], [0.0, 0.0], [0.0, 0.0]], 'MODEL_3+MODEL_3': [[1, 0], [0, 1]]}

@pytest.fixture
def merge_all_frame():
    return [BOX1, BOX21, BOX22, BOX2, BOX11, BOX12, BOX13, BOX3]

@pytest.fixture
def indices_of_boxes_strategy_passed_list():
    return [{'MODEL_1+MODEL_2':{0: [(3, 0.6), (0, 0.4)], 1: [(2, 0.7)], 2: [(4, 0.6)], 3: [(0, 0.5)]},
            'MODEL_1+MODEL_3':{0: [(0, 0.6)], 1: [(2, 0.7)], 2: [(3, 0.7)]},
            'MODEL_2+MODEL_4':{0: [(2, 0.6)], 2: [(1, 0.7)], 1:[(0, 0.4)]},
            'MODEL_5+MODEL_3':{0: [(0, 0.2)], 2: [(2, 0.7), (4, 0.5)], 1: [(1, 0.4)]},
            'MODEL_6+MODEL_5':{0: [(0, 0.4), (1, 0.1), (2, 0.3)], 1: [(2, 0.7)]}},
            {'MODEL_1+MODEL_2':{0: [(3, 0.6), (0, 0.4)], 1: [(2, 0.7)], 2: [(1, 0.6)], 3: [(0, 0.5)]},
             'MODEL_1+MODEL_3':{0: [(1, 0.6)], 1: [(2, 0.7)], 2: [(0, 0.7)]}},
            {'MODEL_1+MODEL_2':{0: [(3, 0.6), (0, 0.4)], 1: [(2, 0.7)], 2: [(1, 0.6)], 3: [(0, 0.5)]},
             'MODEL_3+MODEL_2':{0: [(3, 0.6)], 1: [(2, 0.7)], 2: [(4, 0.7)], 3: [(0, 0.1)]}},
             {'MODEL_1+MODEL_2':{1: [(3, 0.6), (0, 0.4)], 5: [(2, 0.7)], 6: [(1, 0.6)], 0: [(0, 0.5)]},
             'MODEL_3+MODEL_1':{0: [(1, 0.6)], 1: [(2, 0.7)], 2: [(4, 0.7)], 3: [(0, 0.1), (3, 0.5)]}},
             {'MODEL_3+MODEL_1':{0: [(3, 0.6), (0, 0.4), (4, 0.2)], 1: [(2, 0.7)], 2: [(6, 0.6)], 3: [(0, 0.5)]},
             'MODEL_1+MODEL_2':{1: [(0, 0.6)], 5: [(1, 0.7)], 0: [(2, 0.7), (3, 0.3)], 3: [(4, 0.1)]}}]

@pytest.fixture
def ious_by_model_list():
     return [{'MODEL_1+MODEL_2':[[0.4, 0, 0, 0.6, 0], [0, 0, 0.7, 0, 0], [0, 0, 0, 0, 0.6], [0.5, 0, 0, 0, 0]],
              'MODEL_1+MODEL_3':[[0.6, 0, 0, 0, 0], [0, 0, 0.7, 0, 0], [0, 0, 0, 0.7, 0], [0, 0, 0, 0, 0]],
              'MODEL_2+MODEL_4':[[0, 0, 0.6], [0.4, 0, 0], [0, 0.7, 0], [0, 0, 0], [0, 0, 0]],
              'MODEL_5+MODEL_3':[[0.2, 0, 0, 0, 0], [0, 0.4, 0, 0, 0], [0, 0, 0.7, 0, 0.5]],
              'MODEL_6+MODEL_5':[[0.4, 0.1, 0.3], [0, 0, 0.7]]},
             {'MODEL_1+MODEL_2':[[0.4, 0, 0, 0.6], [0, 0, 0.7, 0], [0, 0.6, 0, 0], [0.5, 0, 0, 0]],
              'MODEL_1+MODEL_3':[[0, 0.6, 0], [0, 0, 0.7], [0.7, 0, 0]]},
             {'MODEL_1+MODEL_2':[[0.4, 0, 0, 0.6, 0], [0, 0, 0.7, 0, 0], [0, 0.6, 0, 0, 0], [0.5, 0, 0, 0, 0]],
              'MODEL_3+MODEL_2':[[0, 0, 0, 0.6, 0], [0, 0, 0.7, 0, 0], [0, 0, 0, 0, 0.7], [0.1, 0, 0, 0, 0]]},
             {'MODEL_1+MODEL_2':[[0.5, 0, 0, 0], [0.4, 0, 0, 0.6], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0.7, 0], [0, 0.6, 0, 0]],
              'MODEL_3+MODEL_1':[[0, 0.6, 0, 0, 0, 0, 0], [0, 0, 0.7, 0, 0, 0, 0], [0, 0, 0, 0, 0.7, 0, 0], [0.1, 0, 0, 0.5, 0, 0, 0]]},
             {'MODEL_3+MODEL_1':[[0.4, 0, 0, 0.6, 0.2, 0, 0], [0, 0, 0.7, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0.6], [0.5, 0, 0, 0, 0, 0, 0]],
              'MODEL_1+MODEL_2':[[0, 0, 0.7, 0.3, 0], [0.6, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0.1], [0, 0, 0, 0, 0], [0, 0.7, 0, 0, 0], [0, 0, 0, 0, 0]]}]
@pytest.fixture
def indices_of_boxes_strategy_passed_result_list():
        return [{'MODEL_1+MODEL_2': {1: [(2, 0.7)], 0: [(0, 0.4)]}, 'MODEL_1+MODEL_3': {1: [(2, 0.7)], 0: [(0, 0.6)]}, 'MODEL_2+MODEL_4': {2: [(1, 0.7)], 0: [(2, 0.6)]}, 'MODEL_5+MODEL_3': {0: [(0, 0.2)], 2: [(2, 0.7)]}, 'MODEL_6+MODEL_5': {0: [(0, 0.4), (2, 0.3)], 1: [(2, 0.7)]}},
                {'MODEL_1+MODEL_2': {0: [(3, 0.6), (0, 0.4)], 1: [(2, 0.7)], 2: [(1, 0.6)]}, 'MODEL_1+MODEL_3': {0: [(1, 0.6)], 1: [(2, 0.7)], 2: [(0, 0.7)]}},
                {'MODEL_1+MODEL_2': {0: [(3, 0.6), (0, 0.4)], 1: [(2, 0.7)], 3: [(0, 0.5)]}, 'MODEL_3+MODEL_2': {0: [(3, 0.6)], 1: [(2, 0.7)], 3: [(0, 0.1)]}},
                {'MODEL_1+MODEL_2': {0: [(0, 0.5)], 1: [(3, 0.6), (0, 0.4)]}, 'MODEL_3+MODEL_1': {0: [(1, 0.6)], 3: [(0, 0.1)]}},
                {'MODEL_3+MODEL_1': {0: [(3, 0.6), (0, 0.4)], 3: [(0, 0.5)]}, 'MODEL_1+MODEL_2': {0: [(2, 0.7), (3, 0.3)], 3: [(4, 0.1)]}}]

@pytest.fixture
def order_of_models_list():
     return [['MODEL_1', 'MODEL_2', 'MODEL_3', 'MODEL_4', 'MODEL_5', 'MODEL_6'],
             ['MODEL_1', 'MODEL_2', 'MODEL_3'],
             ['MODEL_1', 'MODEL_2', 'MODEL_3'],
             ['MODEL_1', 'MODEL_2', 'MODEL_3'],
             ['MODEL_3', 'MODEL_1', 'MODEL_2']
             ]

@pytest.fixture
def indices_of_boxes_to_merge_list():
     return [[(0, 0, 0, 2, 0, 0), (1, 2, 2, 1, 2, 1)],
             [(1, 2, 2), (2, 1, 0), (0, 3, 1)],
             [(1, 2, 1), (3, 0, 3), (0, 3, 0)],
             [(0, 0, 3), (1, 3, 0)],
             [(0, 3, 4), (3, 0, 2)]]

@pytest.fixture
def indices_of_boxes_passed_list():
     return [{'MODEL_1': [0, 1], 'MODEL_2': [0, 2], 'MODEL_3': [0, 2], 'MODEL_4': [2, 1], 'MODEL_5': [0, 2], 'MODEL_6': [0, 1]},
             {'MODEL_1': [1, 2, 0], 'MODEL_2': [2, 1, 3], 'MODEL_3': [2, 0, 1]},
             {'MODEL_1': [1, 3, 0], 'MODEL_2': [2, 0, 3], 'MODEL_3': [1, 3, 0]},
             {'MODEL_1': [0, 1], 'MODEL_2': [0, 3], 'MODEL_3': [3, 0]},
             {'MODEL_3': [0, 3], 'MODEL_1': [3, 0], 'MODEL_2': [4, 2]}]