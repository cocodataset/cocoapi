from __future__ import division
from __future__ import print_function

import numpy as np
import copy

from core.detection_input import AnchorTarget2D


class PyramidAnchorTarget2DBase(AnchorTarget2D):
    """
    input: image_meta: tuple(h, w, scale)
           gt_bbox, ndarry(max_num_gt, 4)
    output: anchor_label, ndarray(num_anchor * h * w)
            anchor_bbox_target, ndarray(num_anchor * h * w, 4)
            anchor_bbox_weight, ndarray(num_anchor * h * w, 4)
    """

    def apply(self, input_record):
        p = self.p

        im_info = input_record["im_info"]
        gt_bbox = input_record["gt_bbox"]
        assert isinstance(gt_bbox, np.ndarray)
        assert gt_bbox.dtype == np.float32
        valid = np.where(gt_bbox[:, 0] != -1)[0]
        gt_bbox = gt_bbox[valid]

        if gt_bbox.shape[1] == 5:
            gt_bbox = gt_bbox[:, :4]

        valid_index, valid_anchor = self._gather_valid_anchor(im_info)
        cls_label, anchor_label = \
            self._assign_label_to_anchor(valid_anchor, gt_bbox,
                                         p.assign.neg_thr, p.assign.pos_thr, p.assign.min_pos_thr)
        self._sample_anchor(cls_label, p.sample.image_anchor, p.sample.pos_fraction)
        reg_target, reg_weight = self._cal_anchor_target(cls_label, valid_anchor, gt_bbox, anchor_label)
        cls_label, reg_target, reg_weight = \
            self._scatter_valid_anchor(valid_index, cls_label, reg_target, reg_weight)

        """
        cls_label: (all_anchor,)
        reg_target: (all_anchor, 4)
        reg_weight: (all_anchor, 4)
        """
        input_record["rpn_cls_label"] = cls_label
        input_record["rpn_reg_target"] = reg_target
        input_record["rpn_reg_weight"] = reg_weight

        return input_record["rpn_cls_label"], \
               input_record["rpn_reg_target"], \
               input_record["rpn_reg_weight"]


class PyramidAnchorTarget2D(PyramidAnchorTarget2DBase):
    """
    input: image_meta: tuple(h, w, scale)
           gt_bbox, ndarry(max_num_gt, 4)
    output: anchor_label, ndarray(num_anchor * h * w)
            anchor_bbox_target, ndarray(num_anchor * 4, h * w)
            anchor_bbox_weight, ndarray(num_anchor * 4, h * w)
    """

    def __init__(self, pAnchor):
        super(PyramidAnchorTarget2D, self).__init__(pAnchor)

        self.pyramid_levels = len(self.p.generate.stride)
        self.p_list = [copy.deepcopy(self.p) for _ in range(self.pyramid_levels)]

        pyramid_stride = self.p.generate.stride
        pyramid_short = self.p.generate.short
        pyramid_long = self.p.generate.long

        for i in range(self.pyramid_levels):
            self.p_list[i].generate.stride = pyramid_stride[i]
            self.p_list[i].generate.short = pyramid_short[i]
            self.p_list[i].generate.long = pyramid_long[i]

        # generate anchors for multi-leval feature map
        self.anchor_target_2d_list = [PyramidAnchorTarget2DBase(p) for p in self.p_list[::-1]]
        self.anchor_target_2d = PyramidAnchorTarget2DBase(self.p_list[0])

        self.anchor_target_2d.v_all_anchor = self.v_all_anchor
        self.anchor_target_2d.h_all_anchor = self.h_all_anchor

    @property
    def v_all_anchor(self):
        anchors_list = [anchor_target_2d.v_all_anchor for anchor_target_2d in self.anchor_target_2d_list]
        anchors = np.concatenate(anchors_list)
        return anchors

    @property
    def h_all_anchor(self):
        anchors_list = [anchor_target_2d.h_all_anchor for anchor_target_2d in self.anchor_target_2d_list]
        anchors = np.concatenate(anchors_list)
        return anchors

    def apply(self, input_record):

        anchor_size = [0] + [x.h_all_anchor.shape[0] for x in self.anchor_target_2d_list]
        anchor_size = np.cumsum(anchor_size)
        cls_label, reg_target, reg_weight = \
            self.anchor_target_2d.apply(input_record)

        im_info = input_record["im_info"]
        h, w = im_info[:2]

        cls_label_list = []
        reg_target_list = []
        reg_weight_list = []
        for i in range(self.pyramid_levels):
            p = self.anchor_target_2d_list[i].p

            cls_label_level = cls_label[anchor_size[i]:anchor_size[i + 1]]
            reg_target_level = reg_target[anchor_size[i]:anchor_size[i + 1]]
            reg_weight_level = reg_weight[anchor_size[i]:anchor_size[i + 1]]
            """
            label: (h * w * A) -> (A * h * w)
            bbox_target: (h * w * A, 4) -> (A * 4, h * w)
            bbox_weight: (h * w * A, 4) -> (A * 4, h * w)
            """
            if h >= w:
                fh, fw = p.generate.long, p.generate.short
            else:
                fh, fw = p.generate.short, p.generate.long
            cls_label_level = cls_label_level.reshape((fh, fw, -1)).transpose(2, 0, 1).reshape(-1)
            reg_target_level = reg_target_level.reshape((fh, fw, -1)).transpose(2, 0, 1)
            reg_weight_level = reg_weight_level.reshape((fh, fw, -1)).transpose(2, 0, 1)

            reg_target_level = reg_target_level.reshape(-1, fh * fw)
            reg_weight_level = reg_weight_level.reshape(-1, fh * fw)

            cls_label_list.append(cls_label_level)
            reg_target_list.append(reg_target_level)
            reg_weight_list.append(reg_weight_level)

        cls_label = np.concatenate(cls_label_list, axis=0)
        reg_target = np.concatenate(reg_target_list, axis=1)
        reg_weight = np.concatenate(reg_weight_list, axis=1)

        input_record["rpn_cls_label"] = cls_label
        input_record["rpn_reg_target"] = reg_target
        input_record["rpn_reg_weight"] = reg_weight

        return input_record["rpn_cls_label"], \
               input_record["rpn_reg_target"], \
               input_record["rpn_reg_weight"]
