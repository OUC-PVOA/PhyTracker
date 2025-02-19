from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import numpy as np
import json
import os
import copy

from ..generic_dataset import GenericDataset


class PMOT2023(GenericDataset):
    default_resolution = [864, 640]
    num_categories = 21
    class_name = ['ceratium furca', 'ceratium fucus', 'ceratium trichoceros', 'chaetoceros curvisetus', 'cladocera',
                  'copepoda1', 'copepoda2', 'coscinodiscus', 'curve thalassiosira', 'guinardia delicatulad', 'helicotheca',
                  'lauderia cleve', 'skeletonema', 'thalassionema nitzschioides', 'thalassiosira nordenskioldi', 'tintinnid',
                  'sanguinea', 'Thalassiosira rotula', 'Protoperidinium', 'Eucampia zoodiacus', 'Guinardia striata']
    _valid_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                  14, 15, 16, 17, 18, 19, 20]
    cat_ids = {v: i for i, v in enumerate(_valid_ids)}
    print(cat_ids)
    num_joints = 21
    flip_idx = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10],
                [11, 12], [13, 14], [15, 16]]
    edges = [[0, 1], [0, 2], [1, 3], [2, 4],
             [4, 6], [3, 5], [5, 6],
             [5, 7], [7, 9], [6, 8], [8, 10],
             [6, 12], [5, 11], [11, 12],
             [12, 14], [14, 16], [11, 13], [13, 15]]
    max_objs = 256

    def __init__(self, opt, split):
        self.data_dir = os.path.join(opt.data_dir, 'PMOT2023')
        self.img_dir = os.path.join(self.data_dir, 'train')
        if split == 'val':
            self.annot_path = os.path.join(
                self.data_dir, 'annotations',
                'val.json').format(split)

        elif split == 'train':
            self.annot_path = os.path.join(
                self.data_dir, 'annotations',
                'train.json').format(split)

        else:
            if opt.task == 'exdet':
                self.annot_path = os.path.join(
                    self.data_dir, 'annotations',
                    'train.json').format(split)
            else:
                self.annot_path = os.path.join(
                    self.data_dir, 'annotations',
                    'train.json').format(split)

        self.images = None
        ann_path = self.annot_path
        img_dir = self.img_dir
        super(PMOT2023, self).__init__(opt, split, ann_path, img_dir)

        self.num_samples = len(self.images)

        print('Loaded {} {} samples'.format(split, self.num_samples))

    def _to_float(self, x):
        return float("{:.2f}".format(x))

    def convert_eval_format(self, all_bboxes):
        detections = []
        for image_id in all_bboxes:
            if type(all_bboxes[image_id]) != type({}):
                for j in range(len(all_bboxes[image_id])):
                    item = all_bboxes[image_id][j]
                    cat_id = item['class'] - 1
                    category_id = self._valid_ids[cat_id]
                    bbox = item['bbox']
                    bbox[2] -= bbox[0]
                    bbox[3] -= bbox[1]
                    bbox_out = list(map(self._to_float, bbox[0:4]))
                    detection = {
                        "image_id": int(image_id),
                        "category_id": int(category_id),
                        "bbox": bbox_out,
                        "score": float("{:.2f}".format(item['score']))
                    }
                    detections.append(detection)
        return detections

    def __len__(self):
        return self.num_samples

    def save_results(self, results, save_dir):
        json.dump(self.convert_eval_format(results),
                  open('{}/results_coco.json'.format(save_dir), 'w'))

    def run_eval(self, results, save_dir):
        self.save_results(results, save_dir)
        coco_dets = self.coco.loadRes('{}/results_coco.json'.format(save_dir))
        coco_eval = COCOeval(self.coco, coco_dets, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()