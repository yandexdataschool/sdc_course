import shapely
import scipy.optimize
import torch
import numpy as np
import tqdm

from frustum_dataset import get_detection3d_corner_points, g_class2type, g_type2class


DontCareType = -1


def shapely_iou(convex_a, convex_b):
    return convex_a.intersection(convex_b).area / (convex_a.union(convex_b).area + 1e-12)


def pairwise_iou(gt_cls_convexes_xy, pred_cls_convexes_xy):
    ans = torch.zeros(len(gt_cls_convexes_xy), len(pred_cls_convexes_xy), dtype=torch.float32)
    for i, gt_convex in enumerate(gt_cls_convexes_xy):
        for j, pred_convex in enumerate(pred_cls_convexes_xy):
            ans[i, j] = shapely_iou(gt_convex, pred_convex)
    return ans


def get_min_iou_config():
    ans = {i: 0.5 for i in range(len(g_class2type))}
    ans[g_type2class['Pedestrian']] = 0.25
    ans[g_type2class['Cyclist']] = 0.5
    return ans


def match_bboxes_single_scene(gt_bboxes, gt_labels, pred_bboxes, pred_labels, min_iou=None):
    # NOTE: mAP can't be calculated, because confidence can only be retrieved from camera detections
    # let's calculate confusion matrix for fixed confidence and check some metrics

    if min_iou is None:
        min_iou = get_min_iou_config()

    gt_convexes_xy = [shapely.convex_hull(shapely.MultiPoint([(p[0], p[1]) for p in bbox])) for bbox in gt_bboxes]
    pred_convexes_xy = [shapely.convex_hull(shapely.MultiPoint([(p[0], p[1]) for p in bbox])) for bbox in pred_bboxes]

    ans = [{} for _ in range(len(g_type2class))]
    for cls in range(len(g_type2class)):
        tp, fp, fn = 0, 0, 0
        # Match with DontCare to support weak labeling and easy mode evaluation
        gt_cls_mask = (np.array(gt_labels) == cls) | (np.array(gt_labels) == DontCareType)
        pred_cls_mask = np.array(pred_labels) == cls

        masked_gt_labels = np.array(gt_labels)[gt_cls_mask]

        # print(gt_convexes_xy)
        gt_cls_convexes_xy = [el for idx, el in enumerate(gt_convexes_xy) if gt_cls_mask[idx]]
        pred_cls_convexes_xy = [el for idx, el in enumerate(pred_convexes_xy) if pred_cls_mask[idx]]

        iou_mtx = pairwise_iou(gt_cls_convexes_xy, pred_cls_convexes_xy)
        gt_match_indices, pred_match_indices = scipy.optimize.linear_sum_assignment(iou_mtx, maximize=True)

        # count unmatched gt and pred as fn and fp respectively

        # do not count DontCare FN as we do not need to detect them
        fn += (masked_gt_labels[list(set(list(range(len(gt_cls_convexes_xy)))).difference(gt_match_indices))] != DontCareType).sum()
        # no need for same thing for FP as we never predict DontCare
        fp += len(set(list(range(len(pred_cls_convexes_xy)))).difference(pred_match_indices))

        # count matched gt with high IOU as tp and the rest as both fp/fn
        matched_ious = iou_mtx[gt_match_indices, pred_match_indices]
        tp = (matched_ious >= min_iou[cls])[masked_gt_labels[gt_match_indices] != DontCareType].sum()
        fn += (matched_ious < min_iou[cls])[masked_gt_labels[gt_match_indices] != DontCareType].sum()
        fp += (matched_ious < min_iou[cls])[masked_gt_labels[gt_match_indices] != DontCareType].sum()
        ans[cls] = {
            'tp': tp,
            'fp': fp,
            'fn': fn,
        }
    return ans


def compute_confusion_mtx(kitti_dataset, predicted_scenes, projector: 'Projector'):
    confusion_matrices = []
    for scene_idx, scene in enumerate(tqdm.tqdm(kitti_dataset)):
        gt_cuboids = []
        gt_labels = []
        for label in scene['labels']:
            # if label['type'] == 'DontCare':
            #     continue

            gt_world_location = projector.from_homogenous_coords(
                projector.camera_to_world(np.array(label['location']), scene['calibration']))
            cuboid_pts = get_detection3d_corner_points(
                torch.tensor(gt_world_location),
                torch.tensor(label['dimensions']),
                np.pi / 2 - label['rotation_y'])
            gt_cuboids.append(cuboid_pts)
            if label['type'] == 'DontCare':
                gt_labels.append(DontCareType)
            else:
                gt_labels.append(g_type2class[label['type']])
        confusion_mtx = match_bboxes_single_scene(
            gt_cuboids, gt_labels,
            [x[0] for x in predicted_scenes.get(scene_idx, [])],
            [x[1] for x in predicted_scenes.get(scene_idx, [])],
        )
        confusion_matrices.append(confusion_mtx)
    total_mtx = [
        {
            key: sum([el[cls][key] for el in confusion_matrices])
            for key in ['tp', 'fp', 'fn']
        }
        for cls in range(len(g_type2class))
    ]
    return total_mtx

