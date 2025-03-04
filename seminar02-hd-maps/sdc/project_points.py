import numpy as np


def project_points_on_image(
        image,
        cloud,
        lidar_to_camera_transform,
        rectification_matrix,
        projection_matrix,
        only_valid=False):
    """
    :param image: Numpy array of shape (H, W, C) or (H, W)
    :param cloud: Numpy array of shape (N, 3) or (N, 4)
    :param lidar_to_camera_transform: Numpy array of shape (4, 4)
    :param rectification_matrix: Numpy array of shape (4, 4)
    :param projection_matrix: Numpy array of shape (4, 3)
    :returns: point_to_image_map, depths, valid_points_mask
    """
    assert cloud.ndim == 2
    assert lidar_to_camera_transform.shape == (4, 4)
    assert projection_matrix.shape == (3, 4)
    assert rectification_matrix.shape == (4, 4)

    num_points = cloud.shape[0]
    if cloud.shape[1] == 3:
        # Only X, Y, Z are provided
        points_xyz1 = np.hstack([cloud, np.ones((num_points, 1))])
    elif cloud.shape[1] == 4:
        # X, Y, Z and I are provided
        points_xyz1 = np.array(cloud, copy=True)
        points_xyz1[:, 3] = 1.
    assert points_xyz1.shape == (num_points, 4)

    projection_transform = np.dot(projection_matrix, rectification_matrix)
    points_xyz1_in_camera = np.dot(lidar_to_camera_transform, points_xyz1.T).T
    projected_points = np.dot(projection_transform, points_xyz1_in_camera.T).T
    assert projected_points.shape == (num_points, 3)

    depths = projected_points[:, 2]
    assert depths.shape == (num_points,)

    point_to_image_map = projected_points[:, :2] / depths[:, None]
    assert point_to_image_map.shape == (num_points, 2)

    image_width, image_height = image.size
    valid_points_mask =\
        (np.round(point_to_image_map[:, 0]) > 0.) &\
        (np.round(point_to_image_map[:, 0]) < image_width) &\
        (np.round(point_to_image_map[:, 1]) > 0) &\
        (np.round(point_to_image_map[:, 1]) < image_height) &\
        (depths > 0.)
    assert valid_points_mask.shape == (num_points,)

    if only_valid:
        return point_to_image_map[valid_points_mask], depths[valid_points_mask], valid_points_mask
    return point_to_image_map, depths, valid_points_mask


def project_points_on_cylinder(points, attributes, num_lines=64, num_columns=640):
    """
    :param points: Points in lidar frame which are to be projected on cylinder.
        Numpy array of shape (N, 3) or (N, 4) with the first 3 columns containing X, Y, Z
        coordinates.
    :param attributes:
    :param num_lines:
    :param num_columns:

    :returns: scan, points_to_cylinder_map
    """
    assert points.ndim == 2
    assert points.shape[1] == 3 or points.shape[1] == 4
    num_points = points.shape[0]

    heights = points[:, 2]
    xy_distances = np.linalg.norm(points[:, :2], axis=1)

    # We need to revert both axis
    z_angles = np.arctan2(-heights, xy_distances)
    xy_angles = np.arctan2(-points[:, 1], points[:, 0])

    z_angle_min, z_angle_max = z_angles.min(), z_angles.max()
    xy_angle_min, xy_angle_max = xy_angles.min(), xy_angles.max()
    assert z_angle_max > z_angle_min
    assert xy_angle_max > xy_angle_min

    # TODO(siri3us) Probably min and max angles should be external parameters
    us = (z_angles - z_angle_min) / (z_angle_max - z_angle_min)
    vs = (xy_angles - xy_angle_min) / (xy_angle_max - xy_angle_min)

    us = np.round((num_lines - 1) * us)
    vs = np.round((num_columns - 1) * vs)
    point_to_cylinder_map = np.stack([vs, us], axis=1)
    assert point_to_cylinder_map.shape == (num_points, 2)

    # [u, v, attributes + validity]
    # Last attribute is a number of lidar points in the cell
    scan = np.zeros([len(attributes) + 1, num_lines, num_columns])
    for point_idx in range(num_points):
        v, u = np.round(point_to_cylinder_map[point_idx])
        v, u = int(v), int(u)
        # Marking this cell as a filled one
        scan[-1, u, v] += 1
        for attribute_idx in range(len(attributes)):
            scan[attribute_idx, u, v] = attributes[attribute_idx][point_idx]
    return scan, point_to_cylinder_map
