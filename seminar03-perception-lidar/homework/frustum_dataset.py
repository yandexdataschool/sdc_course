import copy
import random
from typing import TypedDict, Union, Literal, List, Tuple

import numpy as np
import torch
import torchvision
import torch.utils.data as data
import tqdm

from kitti_dataset import Calibration, g_type2class, g_class2type


NUM_HEADING_BIN = 12
NUM_SIZE_CLUSTER = 8
FRUSTRUM_CLOUD_SZ = 2048
# length, width, height
g_type_mean_size = {'Car': np.array([3.88311640418,1.62856739989,1.52563191462]),
                    'Van': np.array([5.06763659,1.9007158,2.20532825]),
                    'Truck': np.array([10.13586957,2.58549199,3.2520595]),
                    'Pedestrian': np.array([0.84422524,0.66068622,1.76255119]),
                    'Person_sitting': np.array([0.80057803,0.5983815,1.27450867]),
                    'Cyclist': np.array([1.76282397,0.59706367,1.73698127]),
                    'Tram': np.array([16.17150617,2.53246914,3.53079012]),
                    'Misc': np.array([3.64300781,1.54298177,1.92320313])}

g_mean_size_arr = np.zeros((NUM_SIZE_CLUSTER, 3)) # size clustrs
for i in range(NUM_SIZE_CLUSTER):
    g_mean_size_arr[i] = g_type_mean_size[g_class2type[i]]


class FrustumDataset(data.Dataset):
    def __init__(self, kitti_dataset, detector_2d_wrapper: 'Detector2DWrapper', projector: 'Projector', cuda=True):
        self.kitti_dataset = kitti_dataset
        self.projector = projector
        self.detector_2d_wrapper = detector_2d_wrapper
        print('Creating frustum dataset for split', self.kitti_dataset.split)
        self.frustums = []
        self.cuda = cuda
        if self.cuda:
            self.detector_2d_wrapper.to_cuda()
        for idx, scene in enumerate(tqdm.tqdm(self.kitti_dataset)):
            if self.kitti_dataset.split == 'training':
                if not len(scene['labels']):
                    continue
                self.frustums.extend(self._generate_frustums_from_single_scene_training(scene, idx))
            else:
                self.frustums.extend(self._generate_frustums_from_single_scene_testing(scene, idx))
        if self.cuda:
            self.detector_2d_wrapper.to_cpu()

        self.frustums = [entry for entry in self.frustums if entry['cloud'].size]

    def __len__(self):
        return len(self.frustums)

    def __getitem__(self, idx):
        ans = self.frustums[idx]
        ans = {x: y for x, y in ans.items()}  # shallow copy every item

        if 'cloud_segmentation' in ans:  # resample GT as well
            ans['cloud'], ans['cloud_segmentation'] = self._resample_cloud(
                ans['cloud'], FRUSTRUM_CLOUD_SZ, labels=ans['cloud_segmentation'])
        else:
            ans['cloud'] = self._resample_cloud(ans['cloud'], FRUSTRUM_CLOUD_SZ)
        return ans

    @staticmethod
    def _rot_22_by_angle(angle):
        return np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]]
        )

    @classmethod
    def _rot_44_by_angle_world(cls, angle):
        # around Z-axis
        rot = np.eye(4, dtype=np.float32)
        rot[:2, :2] = cls._rot_22_by_angle(angle)
        return rot

    @classmethod
    def _rot_44_by_angle_camera(cls, angle):
        # around Y-axis
        return np.array([
            [np.cos(angle), 0, -np.sin(angle), 0],
            [0,             1, 0,              0],
            [np.sin(angle), 0, np.cos(angle),  0],
            [0,             0, 0,              1]
        ], dtype=np.float32)
        return rot

    def _get_bbox_rotation_matrix(self, bbox, calibration: Calibration):
        # rotation mtx to make bbox center always be in the same location
        camera_pts = np.array([[(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2, 1.0]])
        cloud_center_camera_frame = self.projector.projection_to_camera(camera_pts, calibration)
        cloud_center_world = self.projector.camera_to_world(cloud_center_camera_frame, calibration)

        cloud_center_x, cloud_center_y = [cloud_center_world[0][0], cloud_center_world[0][1]]
        neg_angle = -np.arctan2(cloud_center_y, cloud_center_x)
        rot = self._rot_44_by_angle_world(neg_angle)
        return rot, neg_angle

    def _get_rot_idx(self, angle):
        angle = (angle + 5 * np.pi) % (2 * np.pi) # [0, 2 * pi)
        angle_step = (2 * np.pi / NUM_HEADING_BIN)
        angle_bin = int(angle // angle_step)
        angle_residuals = []
        for i in range(NUM_HEADING_BIN):
            cur_bin_center = i * angle_step + angle_step / 2
            angle_bin_residual = angle - cur_bin_center
            if angle_bin_residual >= np.pi:
                angle_bin_residual -= 2 * np.pi
            if angle_bin_residual < -np.pi:
                angle_bin_residual += 2 * np.pi
            angle_residuals.append(angle_bin_residual)

        assert abs(angle_residuals[angle_bin]) < angle_step + 1e-9, \
            f"Incorrect bin/residual: angle {angle}, bin {angle_bin}, residual {angle_residuals[angle_bin]} "
        assert 0 <= angle_bin < NUM_HEADING_BIN, \
            f"Incorrect bin/residual: angle {angle}, bin {angle_bin}, residual {angle_residuals[angle_bin]} "
        return angle_bin, np.array(angle_residuals)

    def _get_box_idx(self, dimensions, class_name):
        size_bin = g_type2class[class_name]
        size_mean = g_type_mean_size[class_name]
        ans = []
        for i in range(NUM_SIZE_CLUSTER):
            ans.extend(dimensions / g_type_mean_size[g_class2type[i]] - 1.0)
        return size_bin, np.array(ans)

    @staticmethod
    def _resample_cloud(cloud, target_size, labels=None):
        if not cloud.size:
            if labels is not None:
                return cloud, labels
            return cloud
        indices = np.random.randint(cloud.shape[0], size=(target_size,))
        if labels is not None:
            return cloud[indices], labels[indices]
        return cloud[indices]

    def _generate_single_frustum(self, kitti_scene, kitti_scene_idx, bbox, label=None, min_z=1, max_z=200):
        points = kitti_scene['cloud']
        calibration = kitti_scene['calibration']

        points_xyzw = self.projector.to_homogenous_coords(points[:, :3])  # NOTE: drop intensity here
        camera_pts = self.projector.world_to_camera(points_xyzw, calibration)
        projected_pts = self.projector.camera_to_projection(camera_pts, calibration)

        projected_dist = projected_pts[:, -1][..., None]
        projected_pts = projected_pts / projected_dist
        bbox = bbox.numpy()
        frustum_mask = (projected_pts[:, 0] >= bbox[0]) \
            & (projected_pts[:, 0] <= bbox[2]) \
            & (projected_pts[:, 1] >= bbox[1]) \
            & (projected_pts[:, 1] <= bbox[3]) \
            & (projected_dist[:, 0] >= min_z) \
            & (projected_dist[:, 0] <= max_z)

        rot, neg_angle = self._get_bbox_rotation_matrix(bbox, calibration)
        bbox = np.floor(bbox).astype(np.int32)
        rotated_points = points_xyzw @ rot.T
        rotated_points[:, -1] = points[:, -1]  # NOTE: copy intensity from source points
        selected_points = rotated_points[frustum_mask]

        ans = {
            'cloud': selected_points,
            'image': torchvision.transforms.functional.resized_crop(
                kitti_scene['image'],
                bbox[1], bbox[0], bbox[3] - bbox[1] + 1, bbox[2] - bbox[0] + 1,
                size=(64, 64)  # demonstration purposes only!
            ),
            'frustum_rotation_angle': neg_angle,
            'kitti_scene_idx': kitti_scene_idx,
            # NOTE: some of the calibration matrices are only valid for source images (before crop and resize),
            # and we do not want to bother recalculating them, hence they're not returned
        }
        if label:
            location = np.pad(np.array(label['location']), [0, 1], constant_values=1).reshape((1, 4))

            # let's build transformation from arbitrary object to object with location=0 and rotation=0
            target_translate = np.eye(4)
            target_translate[:3, 3] = -location[0, :3]
            target_rot = self._rot_44_by_angle_camera(label['rotation_y'])
            selected_points_in_camera_frame = self.projector.world_to_camera(points_xyzw[frustum_mask], calibration)
            selected_points_in_object_frame = (target_rot @ target_translate @ selected_points_in_camera_frame.T).T

            # Using margins of 1.15, because some cuboids do not cover entire objects.
            # I looked into it for quite some time and am pretty sure that everything up to this point was done right;
            # however, boxes sometimes do not cover entire objects' point clouds
            selected_obj_mask = \
                    (selected_points_in_object_frame[:, 0] > -label['dimensions'][0] / 2 * 1.15) & \
                    (selected_points_in_object_frame[:, 0] < label['dimensions'][0] / 2 * 1.15) & \
                    (selected_points_in_object_frame[:, 1] > -label['dimensions'][2] * 1.15) & \
                    (selected_points_in_object_frame[:, 1] < label['dimensions'][2] * 0.15) & \
                    (selected_points_in_object_frame[:, 2] > -label['dimensions'][1] / 2 * 1.15) & \
                    (selected_points_in_object_frame[:, 2] < label['dimensions'][1] / 2 * 1.15)

            world_location = self.projector.camera_to_world(location, calibration)
            world_location = (world_location @ rot.T)[..., :3]
            label = copy.deepcopy(label)

            # encode box in world frame instead of camera frame
            ans['world_location'] = world_location[0, :]
            ans['cloud_segmentation'] = selected_obj_mask
            ans['size_idx'], ans['size_residual'] = self._get_box_idx(label['dimensions'], label['type'])
            rotation_y_camera = label['rotation_y'] - neg_angle
            ans['heading_idx'], ans['heading_residual'] = self._get_rot_idx(np.pi / 2 - rotation_y_camera)
        return ans

    def _generate_frustums_from_single_scene_training(self, kitti_scene, kitti_scene_idx):
        predictions_bboxes = self.detector_2d_wrapper.evaluate(kitti_scene['image'])
        interesting_labels = copy.deepcopy([x for x in kitti_scene['labels'] if x['type'] != 'DontCare'])
        if not len(interesting_labels):
            return
        for label in self.detector_2d_wrapper.match_with_gt_objects(interesting_labels, predictions_bboxes):
            yield self._generate_single_frustum(
                kitti_scene,
                kitti_scene_idx,
                torch.tensor(label['bbox']),
                label=label)

    def _generate_frustums_from_single_scene_testing(self, kitti_scene, kitti_scene_idx):
        predictions_bboxes = self.detector_2d_wrapper.evaluate(kitti_scene['image'])
        for bbox in predictions_bboxes:
            yield self._generate_single_frustum(kitti_scene, kitti_scene_idx, bbox, label=None)


def get_detection3d_corner_points(center, dims, rotation_y):
    pts = []
    for sz in [-0, 1]:
        for sx in [-0.5, 0.5]:
            for sy in [-0.5, 0.5]:
                pts.append((dims[0] * sx, dims[1] * sy, dims[2] * sz))
    pts = np.array(pts)
    rot = FrustumDataset._rot_44_by_angle_world(rotation_y)
    pts = np.pad(pts, [[0, 0], [0, 1]], mode='constant', constant_values=1)
    pts = pts @ rot.T
    pts = pts[:, :3] + center[None, :].numpy()
    return pts

