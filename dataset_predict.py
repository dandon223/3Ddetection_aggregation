import glob
import os
import pathlib
from logging import Logger
from typing import Dict, List, Optional, Tuple, Type, Union
from nuscenes.utils import splits

import numpy as np
import numpy.typing as npt
from easydict import EasyDict  # type: ignore
import nuscenes  # type: ignore
from pcdet.datasets.dataset import DatasetTemplate  # type: ignore
from pcdet.datasets.kitti.kitti_dataset import (  # type: ignore
    KittiDataset as KittiDatasetPCDet,
)
from pcdet.datasets.nuscenes.nuscenes_dataset import (  # type: ignore
    NuScenesDataset as NuScenesDatasetPCDet,
)
from pcdet.datasets.nuscenes.nuscenes_utils import (  # type: ignore
    boxes_lidar_to_nusenes,
    lidar_nusc_box_to_global,
)
from torch import Tensor


class InvalidPointCloudException(Exception):
    """
    Raised when the input file does not contain a valid point cloud
    """

    pass


class Dataset(DatasetTemplate):
    """
    Base class for datasets used for prediction
    """

    def __init__(
        self,
        dataset_cfg: Optional[EasyDict],
        class_names: List[str],
        root_path: pathlib.Path,
        logger: Optional[Logger] = None,
        ext: str = ".bin",
    ):
        """Constructor, initializes class members, searches for files containing point clouds

        :param dataset_cfg: dataset config
        :type dataset_cfg: Optional[EasyDict]
        :param class_names: objects belonging to classes in this list are to be detected
        :type class_names: List[str]
        :param root_path: The path to the root of the dataset
        :type root_path: pathlib.Path
        :param logger: logger, defaults to None
        :type logger: Optional[Logger], optional
        :param ext: point cloud file extension, defaults to ".bin"
        :type ext: str, optional
        """
        super().__init__(
            dataset_cfg=dataset_cfg,
            class_names=class_names,
            training=False,
            root_path=root_path,
            logger=logger,
        )
        self.root_path = root_path
        self.root_split_path = self.root_path

        self.ext = ext

        data_file_list = (
            glob.glob(str(self.root_path / f"*{self.ext}"))
            if self.root_path.is_dir()
            else [self.root_path.as_posix()]
        )

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def _make_data_dict(
        self, index: Union[int, str], points: npt.NDArray[np.float32]
    ) -> Dict[str, Union[str, bool, Tuple[int, int], npt.NDArray]]:
        """Prepare a dictionary that describes the frame

        :param index: frame ID
        :type index: Union[int, str]
        :param points: point cloud
        :type points: npt.NDArray[np.float32]
        :return: dictionary that describes the frame
        :rtype: Dict[str, Union[str, bool, Tuple[int, int], npt.NDArray]]
        """
        input_dict = {
            "points": points,
            "frame_id": index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict

    def __len__(self) -> int:
        return len(self.sample_file_list)

    def __getitem__(self, index):
        raise NotImplementedError


class KittiDataset(Dataset, KittiDatasetPCDet):
    """
    Dataset to generate annotations in KITTI format
    """

    def __init__(
        self,
        dataset_cfg: Optional[EasyDict],
        class_names: List[str],
        root_path: pathlib.Path,
        logger: Optional[Logger] = None,
    ):
        """Constructor, initializes class members, searches for files containing point clouds

        :param dataset_cfg: dataset config
        :type dataset_cfg: Optional[EasyDict]
        :param class_names: objects belonging to classes in this list are to be detected
        :type class_names: List[str]
        :param root_path: The path to the root of the dataset
        :type root_path: pathlib.Path
        :param logger: logger, defaults to None
        :type logger: Optional[Logger], optional
        """
        super().__init__(
            dataset_cfg,
            class_names,
            root_path=root_path,
            logger=logger,
            ext=".bin",
        )

    def __getitem__(
        self, index: int
    ) -> Dict[str, Union[str, bool, Tuple[int, int], npt.NDArray]]:
        """
        Returns a dictionary describing the specified frame

        :param index: frame number
        :type index: int
        :raises InvalidPointCloudException: when the point cloud file is not valid
        :return: dictionary describing the specified frame
        :rtype: Dict[str, Union[str, bool, Tuple[int, int], npt.NDArray]]
        """

        try:
            points = np.fromfile(
                self.sample_file_list[index], dtype=np.float32
            ).reshape(-1, 4)
        except FileNotFoundError as exc:
            if self.logger is not None:
                self.logger.warning(str(exc))
            raise InvalidPointCloudException from exc
        except ValueError as exc:
            if self.logger is not None:
                self.logger.warning(
                    "File %s does not contain a valid point cloud",
                    self.sample_file_list[index],
                )
            raise InvalidPointCloudException from exc

        data_dict = self._make_data_dict(index, points)
        data_dict.update(
            {
                "image_shape": (1242, 375),
                "calib": self.get_calib(
                    self.sample_file_list[index].split("/")[-1].split(self.ext)[0]
                ),
            }
        )
        return data_dict


class NuscenesDataset(Dataset, NuScenesDatasetPCDet):
    """
    Dataset to generate annotations in NuScenes format
    """

    def __init__(
        self,
        dataset_cfg: EasyDict,
        class_names: List[str],
        detector: str,
        root_path: Optional[pathlib.Path] = None,
        logger: Optional[Logger] = None,
    ):
        """Constructor, initializes class members, searches for files containing point clouds

        :param dataset_cfg: dataset config
        :type dataset_cfg: EasyDict
        :param class_names: objects belonging to classes in this list are to be detected
        :type class_names: List[str]
        :param root_path: The path to the root of the dataset, defaults to None
        :type root_path: Optional[pathlib.Path], optional
        :param logger: logger, defaults to None
        :type logger: Optional[Logger], optional
        """
        if root_path is None:
            root_path = pathlib.Path(dataset_cfg.DATA_PATH)

        super().__init__(
            dataset_cfg,
            class_names,
            root_path=root_path,
            logger=None,
            ext=".bin",
        )
        self.logger = logger

        try:
            self.nusc = nuscenes.NuScenes(
                dataroot=root_path.parent, version=root_path.name, verbose=False
            )
        except (AssertionError, FileNotFoundError) as exc:
            if logger is not None:
                logger.error(str(exc))
            raise ValueError from exc
        self.detector = detector
        
    def is_scene_from_validation_set(self, frame_token: str):
        scene_token = self.nusc.get("sample", frame_token)['scene_token']
        scene_name =  self.nusc.get("scene", scene_token)['name']
        return scene_name in splits.val

    def transform_det_annos_to_nusc_annos(
        self, det_annos: List[Dict[str, Union[str, npt.NDArray]]], frame_id: str
    ) -> Dict[str, Dict[str, Union[str, float, List[float], None]]]:
        """Transforms the annotations from the detector to the NuScenes format

        Copied from OpenPCDet nuscenes.utils and slightly modified.
        To change to our box format

        :param det_annos: annotations from the detector
        :type det_annos: List[Dict[str, Union[str, npt.NDArray]]]
        :param frame_id: frame identifier associated with the annotations
        :type frame_id: str
        :return: annotations in NuScenes format
        :rtype: Dict[str, Dict[str, Union[str, float, List[float], None]]]
        """
        nusc_annos: Dict[str, Dict[str, Union[str, float, List[float], None]]] = {
            "results": {},
            "meta": {},
        }

        for det in det_annos:
            annos = []
            box_list = boxes_lidar_to_nusenes(det)
            box_list = lidar_nusc_box_to_global(
                nusc=self.nusc, boxes=box_list, sample_token=frame_id
            )

            for k, box in enumerate(box_list):
                name = det["name"][k]
                nusc_anno = {
                    "sample_token": frame_id,
                    "translation": box.center.tolist(),
                    "size": box.wlh.tolist(),
                    "rotation": box.orientation.elements.tolist(),
                    "velocity": box.velocity[:2].tolist(),
                    "detection_name": name,
                    "detection_score": box.score,
                    "attribute_name": None,
                }
                annos.append(nusc_anno)

            scene_id = self.nusc.get("sample", frame_id)["scene_token"]
            nusc_annos["results"].update({scene_id: annos})

        return nusc_annos

    def generate_prediction_dicts(
        self,
        data_dict: Dict[str, Union[str, bool, Tuple[int, int], npt.NDArray]],
        pred_dicts: List[Dict[str, Tensor]],
        class_names: List[str],
    ) -> Dict[str, Dict[str, Union[str, float, List[float], None]]]:
        """Transforms the results from the detector into annotations

        :param data_dict: dictionary describing the specified frame
        :type data_dict: Dict[str, Union[str, bool, Tuple[int, int], npt.NDArray]]
        :param pred_dicts: results from detector
        :type pred_dicts: List[Dict[str, Tensor]]
        :param class_names: objects belonging to the classes in this list may have been detected
        :type class_names: List[str]
        :return: annotations in NuScenes format
        :rtype: Dict[str, Dict[str, Union[str, float, List[float], None]]]
        """
        annos = super().generate_prediction_dicts(data_dict, pred_dicts, class_names)

        return self.transform_det_annos_to_nusc_annos(
            annos, data_dict["frame_id"].item()  # type: ignore
        )

    def __len__(self) -> int:
        return len(self.nusc.sample)

    def __getitem__(
        self, index: int
    ) -> Dict[str, Union[str, bool, Tuple[int, int], npt.NDArray]]:
        """
        Returns a dictionary describing the specified frame

        :param index: frame number
        :type index: int
        :raises InvalidPointCloudException: when the point cloud file is not valid
        :return: dictionary describing the specified frame
        :rtype: Dict[str, Union[str, bool, Tuple[int, int], npt.NDArray]]
        """
        sample = self.nusc.sample[index]

        lidar_data_token = sample["data"]["LIDAR_TOP"]
        point_cloud_filename = self.nusc.get("sample_data", lidar_data_token)[
            "filename"
        ]
        point_cloud_path = os.path.join(self.nusc.dataroot, point_cloud_filename)

        try:
            points = np.fromfile(point_cloud_path, dtype=np.float32).reshape(-1, 5)
            if self.detector=="TED":
                points = points[:,:4]
            elif self.detector=="SPHEREFORMER":
                points = points[:,:5]
        except FileNotFoundError as exc:
            msg = str(exc)
            if self.logger is not None:
                self.logger.warning(msg)
            raise InvalidPointCloudException(msg) from exc
        except ValueError as exc:
            if self.logger is not None:
                self.logger.warning(
                    "File %s does not contain a valid point cloud", point_cloud_filename
                )
            raise InvalidPointCloudException from exc

        return self._make_data_dict(sample["token"], points)


DATASETS: Dict[str, Type[Dataset]] = {
    "KittiDataset": KittiDataset,
    "NuScenesDataset": NuscenesDataset,
}
