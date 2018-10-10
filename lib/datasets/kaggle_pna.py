from __future__ import print_function
from __future__ import absolute_import
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# Modified by Ankoor Bhagat for Kaggle Competition
# --------------------------------------------------------

import os
import numpy as np
import pandas as pd
import scipy.sparse
import pickle
from .pna_imdb import imdb
from model.utils.config import cfg


try:
    xrange  # Python 2
except NameError:
    xrange = range  # Python 3


class kaggle_pna(imdb):
    def __init__(self, image_set, year, devkit_path=None):
        imdb.__init__(self, 'pna_' + year + '_' + image_set)
        self._year = year
        self._image_set = image_set
        self._devkit_path = self._get_default_path() if devkit_path is None else devkit_path
        self._data_path = self._devkit_path
        # TODO: Change _classes for binary
        self._classes = ('__background__',  # background - always index 0
                         # 'no lung opacity / not normal',
                         # 'normal',
                         # 'lung opacity')
                         'normal',
                         'opacity')
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = '.dcm'
        self._image_index = self._load_image_set_index()
        self._roidb_handler = self.gt_roidb
        self._df = pd.read_csv(os.path.join(self._data_path, 'Annotations', 'train_labels_bboxes.csv'))
        self._salt = None  # TODO
        self._comp_id = None  # TODO
        self.config = {'cleanup': True,
                       'use_salt': False,  # TODO
                       'use_diff': False,  # TODO
                       'matlab_eval': False,  # TODO
                       'rpn_file': None,
                       'min_size': 2,
                       'mode': "train"}  # Default is 'train'

        assert os.path.exists(self._devkit_path), 'Path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), 'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, pid):
        """
        Return the absolute path to image pid.dcm in the image sequence.
        """
        return self.image_path_from_index(self._image_index[pid])

    def image_id_at(self, pid):
        """
        Return the image pid in the image sequence.
        """
        return pid

    def image_path_from_index(self, pid):
        """
        Construct an image path from the image's "index" identifier.
        """
        #dcm_dir = "DCMImagesTest" if self.config['mode'] == 'test' else "DCMImagesTrain"
        #print('dcm_dir: ', dcm_dir)
        image_path = os.path.join(self._data_path, "DCMImagesTrain", pid + self._image_ext)
        assert os.path.exists(image_path), 'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        Load the pids listed in this dataset's image set file.
        """
        # Example path to image set file: self._devkit_path + /PNAdevkit/PNA2018/ImageSets/test.txt
        image_set_file = os.path.join(self._data_path, 'ImageSets',
                                      self._image_set + '.txt')
        assert os.path.exists(image_set_file), 'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index

    def _get_default_path(self):
        """
        Return the default path where Kaggle Pneumonia is expected to be installed.
        """
        return os.path.join(cfg.DATA_DIR, 'PNAdevkit', 'PNA' + self._year)

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')  # self.cache_path in imdb class
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = pickle.load(fid)
            print('{} ground-truth roidb loaded from {}'.format(self.name, cache_file))
            return roidb

        gt_roidb = [self._load_pna_annnotation(index) for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote ground-truth roidb to {}'.format(cache_file))

        return gt_roidb

    def selective_search_roidb(self):
        # TODO: Implement if needed. As we are using RPN, selective search is not needed.
        pass

    def rpn_roidb(self):
        if self._image_set != "test":
            gt_roidb = self.gt_roidb()
            rpn_roidb = self._load_rpn_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)
        else:
            roidb = self._load_rpn_roidb(None)

        return roidb

    def _load_rpn_roidb(self, gt_roidb):
        filename = self.config['rpn_file']
        print('loading {}'.format(filename))
        assert os.path.exists(filename), 'rpn data not found at: {}'.format(filename)
        with open(filename, 'rb') as f:
            box_list = pickle.load(f)
        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_selective_search_roidb(self, gt_roidb):
        # TODO: Implement if needed. As we are using RPN, selective search is not needed.
        pass

    def _load_pna_annnotation(self, pid):
        objs = self._df.query('patientId=="{}"'.format(pid))
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)  # Not used
        seg_areas = np.zeros((num_objs), dtype=np.float32)  # Not used
        ishards = np.zeros((num_objs), dtype=np.int32)  # Not used

        objs.reset_index(drop=True, inplace=True)
        objs = pd.DataFrame(objs)
        for ix, obj in objs.iterrows():
            # Get bounding box coordinates (use width and height to get x2, y2)
            x1 = float(obj['x'])
            y1 = float(obj['y'])
            x2 = float(x1 + obj['width'])
            y2 = float(y1 + obj['height'])
            boxes[ix, :] = [x1, y1, x2, y2]

            ishards[ix] = 0  # Difficult is set as 0 as no information available
            cls = self._class_to_ind[obj['category'].lower()]  # TODO: Change here for binary "class"
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0  # TODO: Check if 0.0 makes any difference, check pascal data xml
            seg_areas[ix] = 0.0

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes': boxes,
                'gt_classes': gt_classes,
                'gt_ishard': ishards,
                'gt_overlaps': overlaps,
                'flipped': False,
                'seg_areas': seg_areas}

    def _get_comp_id(self):
        comp_id = (self._comp_id + '_' + self._salt if self.config['use_salt']
                   else self._comp_id)
        return comp_id

    # def _get_pna_results_file_template(self):
    #     # PNAdevkit/results/PNA2018/<comp_id>_det_test_normal.txt
    #     filename = self._get_comp_id() + '_det_' + self._image_set + '_{:s}.txt'
    #     filedir = os.path.join(self._devkit_path, 'results', 'PNA' + self._year)
    #     if not os.path.exists(filedir):
    #         os.makedirs(filedir)
    #     path = os.path.join(filedir, filename)
    #     return path
    #
    # def _write_pna_results_file(self, all_boxes):
    #     for cls_idx, cls in enumerate(self.classes):
    #         if cls == '__background__':
    #             continue
    #         print('Writing "{}" PNA results file'.format(cls))
    #         filename = self._get_pna_results_file_template().format(cls)
    #         with open(filename, 'wt') as f:
    #             for im_idx, index in enumerate(self.image_index):
    #                 dets = all_boxes[cls_idx][im_idx]
    #                 if dets == []:
    #                     continue
    #                 for k in range(dets.shape[0]):
    #                     f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
    #                             format(index, dets[k, -1],  # prob
    #                                    dets[k, 0], dets[k, 1],  # x, y
    #                                    dets[k, 2] - dets[k, 0],  # width
    #                                    dets[k, 3] - dets[k, 1]))  # height

    # def _get_kaggle_submission_file_template(self):
    #     # PNAdevkit/results/PNA2018/<comp_id>_det_test_normal.txt
    #     filename = self._get_comp_id() + '_det_' + self._image_set + '_{:s}.txt'
    #     filedir = os.path.join(self._devkit_path, 'results', 'PNA' + self._year)
    #     if not os.path.exists(filedir):
    #         os.makedirs(filedir)
    #     path = os.path.join(filedir, filename)
    #     return path

    def write_kaggle_submission_file(self, all_boxes, file_path, min_conf=0.5):
        for cls_idx, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            filepath = file_path.format(str(cls_idx))
            with open(filepath, 'w') as f:
                header = 'patientId,PredictionString'
                f.write(header + '\n')
                for im_idx, index in enumerate(self.image_index):
                    dets = all_boxes[cls_idx][im_idx]
                    valid_dets = dets[dets[:, -1] >= min_conf]  # 4th col is probability
                    num_dets = len(valid_dets)
                    out_str = ""
                    out_str += index
                    out_str += ","
                    for n in range(num_dets):
                        prob = valid_dets[n, -1]
                        x1 = valid_dets[n, 0]
                        y1 = valid_dets[n, 1]
                        width = valid_dets[n, 2] - x1
                        height = valid_dets[n, 3] - y1
                        bboxes_str = "{:.3f} {:.1f} {:.1f} {:.1f} {:.1f} ".format(prob, x1, y1, width, height)
                        out_str += bboxes_str
                    f.write(out_str + "\n")
            print('Submission file created at "{}"'.format(filepath))

    def _do_python_eval(self, output_dir='output'):
        # TODO: Implement if needed
        pass

    def _do_matlab_eval(self, output_dir='output'):
        pass

    def evaluate_detections(self, all_boxes, output_dir=None):
        # TODO: Implement this based on Kaggle metric and Arvind's code
        pass

    def competition_mode(self, on):
        if on:
            self.config['mode'] = "test"

# --------------------------------------------------------------------------------
# Kaggle Metrics
# --------------------------------------------------------------------------------

"""
Function to calculate competition metric. Primarily 
from: https://www.kaggle.com/maxjeblick/keras-competition-metric-for-segmentation
"""


def iou_bbox(box1, box2):

    # Notation for Vij, is that V refers to the direction and i notes the bounding
    # box and j=1 refers to the smaller value and j=2 refers to the larger value.
    x11, y11, w1, h1 = box1
    x21, y21, w2, h2 = box2
    assert w1 * h1 > 0
    assert w2 * h2 > 0
    x12, y12 = x11 + w1, y11 + h1
    x22, y22 = x21 + w2, y21 + h2

    area1, area2 = w1 * h1, w2 * h2

    # In this case, in the notation, `i` refers to the intercept coordinates
    xi1, yi1, xi2, yi2 = max([x11, x21]), max([y11, y21]), min([x12, x22]), min([y12, y22])

    if xi2 <= xi1 or yi2 <= yi1:
        return 0
    else:
        intersect = float((xi2 - xi1) * (yi2 - yi1))
        union = area1 + area2 - intersect
        return intersect / union


def map_iou(boxes_true, boxes_pred, thresholds=(0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75)):
    """
    Mean average precision at different intersection over union (IoU) threshold
     input:
        boxes_true: Nx4 numpy array of ground true bounding boxes of one image. bbox format: (x1, y1, w, h), where
                    (x1,y2) are the upper left corner of the bounding box
        boxes_pred: Nx4 numpy array of predicted bounding boxes of one image. bbox format: (x1, y1, w, h), where
                    (x1,y2) are the upper left corner of the bounding box
        thresholds: IoU thresholds to evaluate mean average precision on
    output:
        map: mean average precision of the image
    """
    # According to the introduction, images with no ground truth bboxes will not be
    # included in the map score unless there is a false positive detection
    # reference available here: https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/discussion/64270
    # return np.nan if both are empty, don't count the image in final evaluation
    if len(boxes_true) == 0 and len(boxes_pred) == 0:
        return np.nan
    map_total = 0

    # loop over thresholds
    for t in thresholds:
        matched_bt = set()
        tp, fn = 0, 0
        for i, bt in enumerate(boxes_true):
            matched = False
            for j, bp in enumerate(boxes_pred):
                miou = iou_bbox(bt, bp)
                if miou >= t and not matched and j not in matched_bt:
                    matched = True
                    tp += 1  # bt is matched for the first time, count as TP
                    matched_bt.add(j)
            if not matched:
                fn += 1  # bt has no match, count as FN
        fp = len(boxes_pred) - len(matched_bt)  # FP is the bp that not matched to any bt
        m = tp / float(tp + fn + fp)
        map_total += m
    return map_total / len(thresholds)


def competition_metric(y_true_bboxes, y_pred_bboxes):
    """
    Mean MAP over all images.
     input:
        y_true_bboxes: List of gt bounding boxes for all images (the bounding box gts for each image are a sublist)
        y_pred_bboxes: List of pred bounding boxes for all images (the bounding box preds for each image are a sublist)
    output:
        mean_map: Mean MAP over all images.
    """
    return np.nanmean([map_iou(box_true, box_pred) for box_true, box_pred in zip(y_true_bboxes, y_pred_bboxes)])



if __name__ == '__main__':
    d = kaggle_pna('trainval', '2018')
    res = d.roidb
    from IPython import embed;
    embed()
