import typing as T
import numpy as np
from scipy.spatial import KDTree


def apply_radius_outlier_removal(
        cloud: np.ndarray,
        search_radius: float,
        min_num_neighbors_in_radius: int,
        num_workers: int = -1) -> T.Tuple[np.ndarray, T.List[int]]:
    assert cloud.ndim == 2 and cloud.shape[0] > 0 and cloud.shape[1] == 3
    assert search_radius > 0. and min_num_neighbors_in_radius > 0
    tree = KDTree(cloud)
    preserved_indices = []
    for point_idx, neighbors_indices in enumerate(
            tree.query_ball_point(cloud, r=search_radius, workers=num_workers)):
        assert len(neighbors_indices) >= 1  # One neighbor is the point itself
        if len(neighbors_indices) >= min_num_neighbors_in_radius + 1:
            preserved_indices.append(point_idx)
    del tree
    preserved_cloud = np.array(cloud[preserved_indices], copy=True)
    return preserved_cloud, preserved_indices
