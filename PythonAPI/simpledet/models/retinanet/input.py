from __future__ import division
from __future__ import print_function

import numpy as np
import copy

from operator_py.cython.bbox import bbox_overlaps_cython
from operator_py.bbox_transform import nonlinear_transform as bbox_transform
from core.detection_input import DetectionAugmentation, AnchorTarget2D


class Norm2DImage(DetectionAugmentation):
    """
    input: image, ndarray(h, w, rgb)
    output: image, ndarray(h, w, rgb)
    """

    def __init__(self, pNorm):
        super(Norm2DImage, self).__init__()
        self.p = pNorm  # type: NormParam

    def apply(self, input_record):
        p = self.p

        image = input_record["image"].astype(np.float32, copy=False)

        image -= p.mean
        image /= p.std

        input_record["image"] = image


class PyramidAnchorTarget2DBase(AnchorTarget2D):
    """
    input: image_meta: tuple(h, w, scale)
           gt_bbox, ndarry(max_num_gt, 4)
    output: anchor_label, ndarray(num_anchor * h * w)
            anchor_bbox_target, ndarray(num_anchor * h * w, 4)
            anchor_bbox_weight, ndarray(num_anchor * h * w, 4)
    """

    def _assign_label_to_anchor(self, valid_anchor, gt_bbox, neg_thr, pos_thr, min_pos_thr):
        num_anchor = valid_anchor.shape[0]
        cls_label = np.full(shape=(num_anchor,), fill_value=-1, dtype=np.float32)
        reg_target = np.zeros(shape=(num_anchor, 4), dtype=np.float32)
        reg_weight = np.zeros(shape=(num_anchor, 4), dtype=np.float32)

        if len(gt_bbox) > 0:
            # num_anchor x num_gt
            overlaps = bbox_overlaps_cython(valid_anchor.astype(np.float32, copy=False), gt_bbox.astype(np.float32, copy=False))
            max_overlaps = overlaps.max(axis=1)
            argmax_overlaps = overlaps.argmax(axis=1)
            gt_max_overlaps = overlaps.max(axis=0)
            # TODO: speed up this
            # TODO: fix potentially assigning wrong anchors as positive
            # A correct implementation is given as
            # gt_argmax_overlaps = np.where((overlaps.transpose() == gt_max_overlaps[:, None]) &
            #                               (overlaps.transpose() >= min_pos_thr))[1]
            gt_argmax_overlaps = np.where((overlaps == gt_max_overlaps) &
                                          (overlaps >= min_pos_thr))
            # anchor class
            cls_label[max_overlaps < neg_thr] = 0
            # fg label: for each gt, anchor with highest overlap
            cls_label[gt_argmax_overlaps[0]] = gt_bbox[gt_argmax_overlaps[1],4]
            # fg label: above threshold IoU
            cls_label[max_overlaps >= pos_thr] = gt_bbox[argmax_overlaps[max_overlaps >= pos_thr],4]

            # anchor regression
            reg_target[:] = bbox_transform(valid_anchor, gt_bbox[argmax_overlaps, :4])
            reg_weight[cls_label >= 1, :] = 1.0
        else:
            cls_label[:] = 0

        return cls_label, reg_target, reg_weight

    def apply(self, input_record):
        p = self.p

        im_info = input_record["im_info"]
        gt_bbox = input_record["gt_bbox"]
        assert isinstance(gt_bbox, np.ndarray)
        assert gt_bbox.dtype == np.float32
        valid = np.where(gt_bbox[:, 0] != -1)[0]
        gt_bbox = gt_bbox[valid]

        valid_index, valid_anchor = self._gather_valid_anchor(im_info)
        cls_label, reg_target, reg_weight = \
            self._assign_label_to_anchor(valid_anchor, gt_bbox,
                                         p.assign.neg_thr, p.assign.pos_thr, p.assign.min_pos_thr)
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

        self.anchor_target_2d_list = [PyramidAnchorTarget2DBase(p) for p in self.p_list]
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

            cls_label_level = cls_label[anchor_size[i]:anchor_size[i+1]]
            reg_target_level = reg_target[anchor_size[i]:anchor_size[i+1]]
            reg_weight_level = reg_weight[anchor_size[i]:anchor_size[i+1]]
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
