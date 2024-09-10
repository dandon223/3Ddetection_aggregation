"""Tests for method _get_box_corners"""
import numpy as np
from pyquaternion import Quaternion #type: ignore
from car_metric.calculate_iou import _get_box_corners

class TestGetBoxCorners():
    """Test class"""

    def test_no_rotation(self):
        dimensions: np.ndarray = np.array([1., 2., 3.])
        location: np.ndarray = np.array([10., 20., 30.])
        rotation: Quaternion = Quaternion(axis=[1, 0, 0], angle=0)

        box_corners: np.ndarray = _get_box_corners(dimensions, location, rotation)
        expected_box_corners: np.ndarray = np.array([
            [9.5, 19, 28.5],
            [10.5, 19, 28.5],
            [10.5, 21, 28.5],
            [9.5, 21, 28.5],
            [9.5, 19, 31.5],
            [10.5, 19, 31.5],
            [10.5, 21, 31.5],
            [9.5, 21, 31.5],
        ])
        np.testing.assert_array_equal(box_corners, expected_box_corners)

    def test_rotation_yaw_90_degrees(self):
        dimensions: np.ndarray = np.array([2., 4., 6.])
        location: np.ndarray = np.array([1., 1., 1.])
        rotation: Quaternion = Quaternion(w=0.7071, x=0, y=0, z=0.7071)

        box_corners: np.ndarray = _get_box_corners(dimensions, location, rotation)
        expected_box_corners: np.ndarray = np.array([
            [3., 0., -2.],
            [3., 2., -2.],
            [-1., 2., -2.],
            [-1., 0., -2.],
            [3., 0., 4.],
            [3., 2., 4.],
            [-1., 2., 4.],
            [-1., 0., 4.]
        ])
        np.testing.assert_allclose(box_corners, expected_box_corners, atol=1e-3)

    def test_rotation_pitch_and_roll_90_degrees(self):
        dimensions: np.ndarray = np.array([2., 4., 6.])
        location: np.ndarray = np.array([1., 1., 1.])
        rotation: Quaternion = Quaternion(w=0.5, x=0.5, y=0.5, z=-0.5)

        box_corners: np.ndarray = _get_box_corners(dimensions, location, rotation)
        expected_box_corners: np.ndarray = np.array([
            [-1., 4., 2.],
            [-1., 4., 0.],
            [3., 4., 0.],
            [3., 4., 2.],
            [-1., -2., 2.],
            [-1., -2., 0.],
            [3., -2., 0.],
            [3., -2., 2.]])
        np.testing.assert_allclose(box_corners, expected_box_corners, atol=1e-3)

    def test_rotation_yaw_pitch_and_roll_90_degrees(self):
        dimensions: np.ndarray = np.array([2., 4., 6.])
        location: np.ndarray = np.array([1., 1., 1.])
        rotation: Quaternion = Quaternion(w=0.7071, x=0, y=0.7071, z=0)

        box_corners: np.ndarray = _get_box_corners(dimensions, location, rotation)
        expected_box_corners: np.ndarray = np.array([
            [-2., -1., 2.],
            [-2., -1., 0.],
            [-2., 3., 0.],
            [-2., 3., 2.],
            [4., -1., 2.],
            [4., -1., 0.],
            [4., 3., 0.],
            [4., 3., 2.]])
        np.testing.assert_allclose(box_corners, expected_box_corners, atol=1e-3)
