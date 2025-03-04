import typing as T
import numpy as np
import pykitti
from ..transforms.rotation_rpy import RotationRPY
from ..geo.geo_position_lla import GeoPositionLLA
from ..geo.geo_position_xyz import GeoPositionXYZ
from pylib.geo.geo_lla_xyz_converter import GeoLlaXyzConverter
from .localization import Localization, build_localizations


class KittiDatasetAdaptor:
    def __init__(self, kitti_dataset: pykitti.raw):
        assert isinstance(kitti_dataset, pykitti.raw)
        self._kitti_dataset = kitti_dataset

    @property
    def kitti_dataset(self) -> pykitti.raw:
        return self._kitti_dataset

    def read_geo_positions_lla(self) -> T.List[GeoPositionLLA]:
        geo_positions_lla: T.List[GeoPositionLLA] = []
        for oxt in self._kitti_dataset.oxts:
            geo_positions_lla.append(
                GeoPositionLLA(
                    latitude=oxt.packet.lat,
                    longitude=oxt.packet.lon,
                    altitude=oxt.packet.alt))
        return geo_positions_lla

    def read_orientations_rpy(self) -> T.List[RotationRPY]:
        rpys: T.List[RotationRPY] = []
        for pose_idx, oxt in enumerate(self._kitti_dataset.oxts):
            roll, pitch, yaw = oxt.packet.roll, oxt.packet.pitch, oxt.packet.yaw
            rpys.append(RotationRPY(roll=roll, pitch=pitch, yaw=yaw))
        return rpys

    def build_localizations(self, reference_point: GeoPositionLLA) -> T.List[Localization]:
        # Initializing converter for conversions between LLA geo positions and XYZ geo positions in
        # mercator system
        geo_lla_xyz_converter = GeoLlaXyzConverter(reference_point)

        # Reading LLA geo positions of GPS sensor
        geo_positions_lla = self.read_geo_positions_lla()

        # Converting LLA geo positions to XYZ geo positions in mercator system
        geo_positions_xyz: T.List[GeoPositionXYZ] = []
        for geo_position_lla in geo_positions_lla:
            geo_positions_xyz.append(geo_lla_xyz_converter.convert_lla_to_xyz(geo_position_lla))

        # Reading GPS sensor orientations in mercator system
        orientations_rpy = self.read_orientations_rpy()

        # Формируем показания локализации (поза в проекции меркатора + LLA-геопозиция)
        return build_localizations(
            geo_positions_lla,
            geo_positions_xyz,
            reference_point,
            orientations_rpy)

    @property
    def num_lidar_clouds(self) -> int:
        return len(self._kitti_dataset.velo_files)

    def read_lidar_cloud_xyzi(self, lidar_cloud_idx: int) -> np.ndarray:
        assert lidar_cloud_idx < self.num_lidar_clouds
        return self._kitti_dataset.get_velo(lidar_cloud_idx)

    def read_lidar_cloud_xyz(self, lidar_cloud_idx: int) -> np.ndarray:
        return np.array(self.read_lidar_cloud_xyzi(lidar_cloud_idx)[:, :3])

    def read_lidar_clouds_xyzi(self) -> T.List[np.ndarray]:
        return list(self._kitti_dataset.velo)
