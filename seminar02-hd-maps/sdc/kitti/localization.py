import typing as T
from collections import namedtuple
from ..geo.geo_position_xyz import GeoPositionXYZ
from ..geo.geo_position_lla import GeoPositionLLA
from ..transforms.rotation_rpy import RotationRPY


Localization = namedtuple(
    'Localization',
    ['geo_position_lla', 'geo_position_xyz', 'reference_point', 'orientation_rpy'])


def build_localization(
        geo_position_lla: GeoPositionLLA,
        geo_position_xyz: GeoPositionXYZ,
        reference_point: GeoPositionLLA,
        orientation_rpy: RotationRPY) -> Localization:
    assert isinstance(geo_position_lla, GeoPositionLLA)
    assert isinstance(geo_position_xyz, GeoPositionXYZ)
    assert isinstance(reference_point, GeoPositionLLA)
    assert isinstance(orientation_rpy, RotationRPY)
    return Localization(
        geo_position_lla,
        geo_position_xyz,
        reference_point,
        orientation_rpy)


def build_localizations(
        geo_positions_lla: T.List[GeoPositionLLA],
        geo_positions_xyz: T.List[GeoPositionXYZ],
        reference_point: GeoPositionLLA,
        orientations_rpy: T.List[RotationRPY]):
    num_poses = len(geo_positions_lla)
    assert len(geo_positions_xyz) == num_poses
    assert len(orientations_rpy) == num_poses

    localizations: T.List[Localization] = []
    for pose_idx in range(num_poses):
        localizations.append(build_localization(
            geo_positions_lla[pose_idx],
            geo_positions_xyz[pose_idx],
            reference_point,
            orientations_rpy[pose_idx]))
    return localizations
