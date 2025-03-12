import typing as T
import numpy as np
from ..transforms.convert_transform import convert_xyz_and_rpy_to_transform_matrix
from .localization import Localization


def convert_localization_to_transform_matrix(localization: Localization) -> np.ndarray:
    position_xyz = np.array((
        localization.geo_position_xyz.x,
        localization.geo_position_xyz.y,
        localization.geo_position_xyz.z))
    return convert_xyz_and_rpy_to_transform_matrix(position_xyz, localization.orientation_rpy)


def convert_localizations_to_transform_matrices(
        localizations: T.List[Localization]) -> T.List[np.ndarray]:
    return [
        convert_localization_to_transform_matrix(localization)
        for localization in localizations]
