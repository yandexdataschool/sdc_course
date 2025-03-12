import copy
import pyproj
from .geo_position_lla import GeoPositionLLA
from .geo_position_xyz import GeoPositionXYZ


class GeoLlaXyzConverter:
    """Converter between mercator projection and longlat projection"""

    def __init__(self, reference_point: GeoPositionLLA):
        self._reference_point = copy.deepcopy(reference_point)
        self._mercator_projector = pyproj.Proj(
            '+proj=merc '
            f'+lat_0={reference_point.latitude} '
            f'+lon_0={reference_point.longitude} '
            '+ellps=WGS84')
        self._latlong_projector = pyproj.Proj('+proj=longlat +ellps=WGS84')

    @property
    def mercator_projector(self) -> pyproj.Proj:
        return self._mercator_projector

    @property
    def latlong_projector(self) -> pyproj.Proj:
        return self._latlong_projector

    @property
    def reference_point(self) -> GeoPositionLLA:
        return self._reference_point

    def convert_xyz_to_lla(self, geo_position_xyz: GeoPositionXYZ) -> GeoPositionLLA:
        assert isinstance(geo_position_xyz, GeoPositionXYZ)
        longitude, latitude, altitude = pyproj.transform(
            self._mercator_projector, self._latlong_projector,
            geo_position_xyz.x, geo_position_xyz.y, geo_position_xyz.z)
        return GeoPositionLLA(
            latitude=latitude + self._reference_point.latitude,
            longitude=longitude,
            altitude=altitude + self._reference_point.altitude)

    def convert_lla_to_xyz(self, geo_position_lla: GeoPositionLLA) -> GeoPositionXYZ:
        assert isinstance(geo_position_lla, GeoPositionLLA)
        x, y, z = pyproj.transform(
            self._latlong_projector, self._mercator_projector,
            geo_position_lla.longitude,
            geo_position_lla.latitude - self._reference_point.latitude,
            geo_position_lla.altitude - self._reference_point.altitude)
        return GeoPositionXYZ(x, y, z)
