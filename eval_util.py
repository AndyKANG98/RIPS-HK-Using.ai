import collections
import functools
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import xml.dom.minidom
import six
import tensorflow as tf





def calc_iou_individual(pred_box, gt_box):
    """Calculate IoU of single predicted and ground truth box
    Args:
        pred_box (list of floats): location of predicted object as
            [xmin, ymin, xmax, ymax]
        gt_box (list of floats): location of ground truth object as
            [xmin, ymin, xmax, ymax]
    Returns:
        float: value of the IoU for the two boxes.
    Raises:
        AssertionError: if the box is obviously malformed
    """

    x1_t, y1_t, x2_t, y2_t = gt_box
    y1_p, x1_p, y2_p, x2_p = pred_box

    if (x1_p > x2_p) or (y1_p > y2_p):
        raise AssertionError(
            "Prediction box is malformed? pred box: {}".format(pred_box))
    if (x1_t > x2_t) or (y1_t > y2_t):
        raise AssertionError(
            "Ground Truth box is malformed? true box: {}".format(gt_box))

    if (x2_t < x1_p or x2_p < x1_t or y2_t < y1_p or y2_p < y1_t):
        return 0.0

    far_x = np.min([x2_t, x2_p])
    near_x = np.max([x1_t, x1_p])
    far_y = np.min([y2_t, y2_p])
    near_y = np.max([y1_t, y1_p])

    inter_area = (far_x - near_x + 1) * (far_y - near_y + 1)
    true_box_area = (x2_t - x1_t + 1) * (y2_t - y1_t + 1)
    pred_box_area = (x2_p - x1_p + 1) * (y2_p - y1_p + 1)
    iou = inter_area / (true_box_area + pred_box_area - inter_area)
    return iou

def label_image(name):
    path = "./test_images/label/"


    #open xml
    dom = xml.dom.minidom.parse(path + name + '.xml')

    #Get element of the root node 
    root = dom.documentElement

    # Create lists for wheels
    xmins = []
    ymins = []
    xmaxs = []
    ymaxs = []

    # Get the all the tags for x,y
    xmins_tag = root.getElementsByTagName('xmin')
    ymins_tag = root.getElementsByTagName('ymin')
    xmaxs_tag = root.getElementsByTagName('xmax')
    ymaxs_tag = root.getElementsByTagName('ymax')
    width = root.getElementsByTagName('width')
    height = root.getElementsByTagName('height')

    
    # Fill the lists for wheels
    for i in xmins_tag:
        xmins.append(float(float(i.firstChild.data)/float(width[0].firstChild.data)))
    for i in ymins_tag:
        ymins.append(float(float(i.firstChild.data)/float(height[0].firstChild.data)))
    for i in xmaxs_tag:
        xmaxs.append(float(float(i.firstChild.data)/float(width[0].firstChild.data)))
    for i in ymaxs_tag:
        ymaxs.append(float(float(i.firstChild.data)/float(height[0].firstChild.data)))

    # Merge four lists into a matrix
    d1 = np.vstack((xmins, ymins))
    d2 = np.vstack((xmaxs, ymaxs))
    d = np.vstack((d1,d2))
    gt_boxes = d.T
    return gt_boxes


def get_single_image_results(gt_boxes, pred_boxes, iou_thr):
    """Calculates number of true_pos, false_pos, false_neg from single batch of boxes.
    Args:
        gt_boxes (list of list of floats): list of locations of ground truth
            objects as [xmin, ymin, xmax, ymax]
        pred_boxes (dict): dict of dicts of 'boxes' (formatted like `gt_boxes`)
            and 'scores'
        iou_thr (float): value of IoU to consider as threshold for a
            true prediction.
    Returns:
        dict: true positives (int), false positives (int), false negatives (int)
    """
    
    all_pred_indices = range(len(pred_boxes))
    all_gt_indices = range(len(gt_boxes))
    if len(all_pred_indices) == 0:
        tp = 0
        fp = 0
        fn = len(gt_boxes)
        return {'true_pos': tp, 'false_pos': fp, 'false_neg': fn}
    if len(all_gt_indices) == 0:
        tp = 0
        fp = len(pred_boxes)
        fn = 0
        return {'true_pos': tp, 'false_pos': fp, 'false_neg': fn}

    gt_idx_thr = []
    pred_idx_thr = []
    ious = []
    for ipb, pred_box in enumerate(pred_boxes):
        for igb, gt_box in enumerate(gt_boxes):
            iou = calc_iou_individual(pred_boxes[pred_box]['boxes'], gt_box)
            if iou > iou_thr:
                gt_idx_thr.append(igb)
                pred_idx_thr.append(ipb)
                ious.append(iou)

    args_desc = np.argsort(ious)[::-1]
    if len(args_desc) == 0:
        # No matches
        tp = 0
        fp = len(pred_boxes)
        fn = len(gt_boxes)
    else:
        gt_match_idx = []
        pred_match_idx = []
        for idx in args_desc:
            gt_idx = gt_idx_thr[idx]
            pr_idx = pred_idx_thr[idx]
            # If the boxes are unmatched, add them to matches
            if (gt_idx not in gt_match_idx) and (pr_idx not in pred_match_idx):
                gt_match_idx.append(gt_idx)
                pred_match_idx.append(pr_idx)
        tp = len(gt_match_idx)
        fp = len(pred_boxes) - len(pred_match_idx)
        fn = len(gt_boxes) - len(gt_match_idx)

    return {'true_pos': tp, 'false_pos': fp, 'false_neg': fn}


def calc_precision_recall(img_results):
    """Calculates precision and recall from the set of images
    Args:
        img_results (dict): dictionary formatted like:
            {
                'img_id1': {'true_pos': int, 'false_pos': int, 'false_neg': int},
                'img_id2': ...
                ...
            }
    Returns:
        tuple: of floats of (precision, recall)
    """
    true_pos = 0; false_pos = 0; false_neg = 0
    for _, res in img_results.items():
        true_pos += res['true_pos']
        false_pos += res['false_pos']
        false_neg += res['false_neg']

    try:
        precision = float(float(true_pos)/(float(true_pos) + float(false_pos)))
    except ZeroDivisionError:
        precision = 0.0
    try:
        recall = float(float(true_pos)/(float(true_pos) + float(false_neg)))
    except ZeroDivisionError:
        recall = 0.0

    return (precision, recall)