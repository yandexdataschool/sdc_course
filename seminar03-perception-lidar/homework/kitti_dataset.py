import copy
from pathlib import Path
import random
from typing import TypedDict, Union, Literal, List, Tuple

import numpy as np
from numpy.typing import NDArray
import torch
import torchvision
import torch.utils.data as data


g_type2class={
    'Car': 0,
    'Van': 1,
    'Truck': 2,
    'Pedestrian': 3,
    'Person_sitting': 4,
    'Cyclist': 5,
    'Tram': 6,
    'Misc': 7
}
g_class2type = {g_type2class[t]: t for t in g_type2class}


class Calibration(TypedDict):
    P0: NDArray[np.float32]  # (3, 4)
    P1: NDArray[np.float32]  # (3, 4)
    P2: NDArray[np.float32]  # (3, 4)
    P3: NDArray[np.float32]  # (3, 4)
    R0_rect: NDArray[np.float32]  # (4, 4)
    Tr_velo_to_cam: NDArray[np.float32]  # (4, 4)
    Tr_imu_to_velo: NDArray[np.float32]  # (4, 4)


LabelType = Union[
    Literal['Car'],
    Literal['Van'],
    Literal['Truck'],
    Literal['Pedestrian'],
    Literal['Person_sitting'],
    Literal['Cyclist'],
    Literal['Tram'],
    Literal['Misc'],
    Literal['DontCare'],
]


OccludedType = Union[
    Literal[0],  # fully visible
    Literal[1],  # partly occluded
    Literal[2],  # largely occluded
    Literal[3],  # unknown
]


class Label(TypedDict):
    type: LabelType
    truncated: float
    occluded: OccludedType
    alpha: float
    bbox: List[int]
    dimensions: List[float]
    location: List[float]
    rotation_y: float
    score: float


def _pad_matrix_to_44(mtx):
    padded = np.eye(4, dtype=mtx.dtype)
    padded[:mtx.shape[0], :mtx.shape[1]] = mtx
    return padded


def split_train_val(dataset_root, folder='training', fraction=0.25):
    dataset_root = Path(dataset_root)
    items = [x.stem for x in (dataset_root / folder / 'velodyne').iterdir()]
    random.seed(42)
    val_items = random.sample(items, int(len(items) * fraction))
    val_items_set = set(val_items)
    train_items = [x for x in items if x not in val_items_set]
    return train_items, val_items


class KittiDataset(data.Dataset):
    def __init__(self, dataset_root, items=None, split='training', only_easy=True):
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.only_easy = only_easy
        if not items:
            self.items = [x.stem for x in (self.dataset_root / self.split / 'velodyne').iterdir()]
        else:
            self.items = items
        self.labels = [self._load_label(item) for item in self.items]
        self.image = [self._load_image(item) for item in self.items]
        self.cloud = [self._load_cloud(item) for item in self.items]
        self.calibration = [self._load_calibration(item) for item in self.items]

    def _get_path(self, data_type, item, suffix):
        return (self.dataset_root / self.split / data_type / item).with_suffix(suffix)

    @staticmethod
    def _parse_schema(line, schema):
        parts = line.split()
        assert len(parts) == len(schema)
        ans = {}
        for field, part in zip(schema, parts):
            ans[field['name']] = field['type'](part)
        return ans

    def _load_label(self, item):
        schema = [
            {'name': 'type', 'type': str},
            {'name': 'truncated', 'type': float},
            {'name': 'occluded', 'type': int},
            {'name': 'alpha', 'type': float},
            {'name': 'bbox_left', 'type': float},
            {'name': 'bbox_top', 'type': float},
            {'name': 'bbox_right', 'type': float},
            {'name': 'bbox_bottom', 'type': float},
            {'name': 'dimensions_height', 'type': float},
            {'name': 'dimensions_width', 'type': float},
            {'name': 'dimensions_length', 'type': float},
            {'name': 'location_x', 'type': float},
            {'name': 'location_y', 'type': float},
            {'name': 'location_z', 'type': float},
            {'name': 'rotation_y', 'type': float},
        ]
        entries = []
        with self._get_path('label_2', item, '.txt').open('r') as inf:
            for line in inf.read().strip().split('\n'):
                entry = self._parse_schema(line, schema)
                structured_entry = {
                    'type': entry['type'],
                    'truncated': entry['truncated'],
                    'occluded': entry['occluded'],
                    'alpha': entry['alpha'],
                    'bbox': [entry['bbox_left'], entry['bbox_top'], entry['bbox_right'], entry['bbox_bottom']],
                    'dimensions': [entry['dimensions_length'], entry['dimensions_width'], entry['dimensions_height']],
                    'location': [entry['location_x'], entry['location_y'], entry['location_z']],
                    'rotation_y': entry['rotation_y']
                }
                entries.append(structured_entry)
        return entries

    def _load_image(self, item: str):
        img_path = self._get_path('image_2', item, '.png')
        return torchvision.io.read_image(str(img_path))

    def _load_cloud(self, item: str):
        cloud_path = self._get_path('velodyne', item, '.bin')
        scan = np.fromfile(str(cloud_path), dtype=np.float32)
        return scan.reshape((-1, 4))

    def _load_calibration(self, item: str):
        ans = {}
        with self._get_path('calib', item, '.txt').open('r') as inf:
            for line in inf.read().strip().split('\n'):
                mtx_name, *coeff = line.split()
                coeff = np.array(list(map(float, coeff))).astype(np.float32)
                ans[mtx_name.rstrip(':')] = coeff

        ans['P0'] = ans['P0'].reshape((3, 4))
        ans['P1'] = ans['P1'].reshape((3, 4))
        ans['P2'] = ans['P2'].reshape((3, 4))
        ans['P3'] = ans['P3'].reshape((3, 4))
        ans['R0_rect'] = _pad_matrix_to_44(ans['R0_rect'].reshape((3, 3)))
        ans['Tr_velo_to_cam'] = _pad_matrix_to_44(ans['Tr_velo_to_cam'].reshape((3, 4)))
        ans['Tr_imu_to_velo'] = _pad_matrix_to_44(ans['Tr_imu_to_velo'].reshape((3, 4)))
        return ans

    def __len__(self):
        return len(self.items)

    def _is_easy_label(self, label: Label):
        return label['bbox'][3] - label['bbox'][1] >= 50 and label['occluded'] == 0 and label['truncated'] <= 0.15

    def __getitem__(self, idx):
        ans = {
            'image': self.image[idx],
            'cloud': self.cloud[idx],
            'calibration': self.calibration[idx],
        }
        ans['labels'] = copy.deepcopy(self.labels[idx])
        if self.only_easy:
            for label in ans['labels']:
                if not self._is_easy_label(label):
                    label['type'] = 'DontCare'
        return ans
