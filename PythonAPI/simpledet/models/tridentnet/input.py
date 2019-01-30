import numpy as np
import mxnet as mx

from core.detection_input import DetectionAugmentation, AnchorTarget2D
from operator_py.cython.bbox import bbox_overlaps_cython


class ScaleAwareRange(DetectionAugmentation):
    def __init__(self, pScaleRange):
        super(ScaleAwareRange, self).__init__()
        self.p = pScaleRange

    def apply(self, input_record):
        p = self.p

        im_info = input_record['im_info']

       # input_record["valid_ranges_on_origin"] = p.cal_on_origin
        input_record["valid_ranges"] = np.array(p.valid_ranges, dtype=np.float32).reshape(-1, 2)
        if p.cal_on_origin:
            input_record["valid_ranges"] *= im_info[2]
        # replace -1 with max_size
        inds = np.where(input_record["valid_ranges"][:, 1] < 0)[0]
        input_record["valid_ranges"][inds, 1] = max(im_info[0], im_info[1])


class TridentAnchorTarget2D(AnchorTarget2D):
    """
    input: image_meta: tuple(h, w, scale)
           gt_bbox, ndarry(max_num_gt, 4)
    output: anchor_label, ndarray(num_branch, num_anchor * 2, h, w)
            anchor_bbox_target, ndarray(num_branch, num_anchor * 4, h, w)
            anchor_bbox_weight, ndarray(num_branch, num_anchor * 4, h, w)
            valid_ranges, ndarray(num_branch, 2)
    """

    def __init__(self, pAnchor):
        super(TridentAnchorTarget2D, self).__init__(pAnchor)

    def _filter_anchor_by_scale_range(self, cls_label, valid_anchor, gt_bbox, valid_range, invalid_anchor_threshd):
        if len(gt_bbox) == 0:
            return
        gt_bbox_sizes = (gt_bbox[:, 2] - gt_bbox[:, 0] + 1.0) * (gt_bbox[:, 3] - gt_bbox[:, 1] + 1.0)
        invalid_gt_bbox_inds = np.where((gt_bbox_sizes < valid_range[0]**2) | (gt_bbox_sizes > valid_range[1]**2))[0]
        invalid_gt_bbox = gt_bbox[invalid_gt_bbox_inds]
        if len(invalid_gt_bbox) > 0:
            invalid_overlaps = bbox_overlaps_cython(
                valid_anchor.astype(np.float32, copy=False), invalid_gt_bbox.astype(np.float32, copy=False))
            invalid_argmax_overlaps = invalid_overlaps.argmax(axis=1)
            invalid_max_overlaps = invalid_overlaps[np.arange(len(valid_anchor)), invalid_argmax_overlaps]

            # ignore anchors overlapped with invalid gt boxes
            disable_inds = np.where((invalid_max_overlaps > invalid_anchor_threshd))[0]
            cls_label[disable_inds] = -1

    def apply(self, input_record):
        p = self.p

        im_info = input_record["im_info"]
        gt_bbox = input_record["gt_bbox"]
        valid_ranges = input_record["valid_ranges"]
        assert isinstance(gt_bbox, np.ndarray)
        assert gt_bbox.dtype == np.float32

        valid = np.where(gt_bbox[:, 0] != -1)[0]
        gt_bbox = gt_bbox[valid]

        if gt_bbox.shape[1] == 5:
            gt_bbox = gt_bbox[:, :4]

        h, w = im_info[:2]
        if h >= w:
            fh, fw = p.generate.long, p.generate.short
        else:
            fh, fw = p.generate.short, p.generate.long

        valid_index, valid_anchor = self._gather_valid_anchor(im_info)

        valid_cls_label, valid_anchor_label = \
            self._assign_label_to_anchor(valid_anchor, gt_bbox,
                                         p.assign.neg_thr, p.assign.pos_thr, p.assign.min_pos_thr)

        cls_labels, reg_targets, reg_weights = [], [], []
        for valid_range in valid_ranges:
            # cls_label, reg_target, reg_weight = valid_cls_label.copy(), valid_reg_target.copy(), valid_reg_weight.copy()
            # self._filter_anchor_by_scale_range(cls_label, reg_weight, valid_anchor, gt_bbox,
            #                                    valid_range, p.trident.invalid_anchor_threshd)
            #
            # self._sample_anchor(cls_label, reg_weight, p.sample.image_anchor, p.sample.pos_fraction)
            #
            # cls_label, reg_target, reg_weight = \
            #     self._scatter_valid_anchor(valid_index, cls_label, reg_target, reg_weight)
            # cls_label, anchor_label = \
            #     self._assign_label_to_anchor(valid_anchor, gt_bbox,
            #                                  p.assign.neg_thr, p.assign.pos_thr, p.assign.min_pos_thr)
            cls_label = valid_cls_label.copy()
            self._filter_anchor_by_scale_range(cls_label, valid_anchor, gt_bbox,
                                               valid_range, p.trident.invalid_anchor_threshd)
            self._sample_anchor(cls_label, p.sample.image_anchor, p.sample.pos_fraction)
            reg_target, reg_weight = self._cal_anchor_target(cls_label, valid_anchor, gt_bbox, valid_anchor_label)
            cls_label, reg_target, reg_weight = \
                self._scatter_valid_anchor(valid_index, cls_label, reg_target, reg_weight)

            cls_labels.append(cls_label.reshape((fh, fw, -1)).transpose(2, 0, 1).reshape(-1))
            reg_targets.append(reg_target.reshape((fh, fw, -1)).transpose(2, 0, 1))
            reg_weights.append(reg_weight.reshape((fh, fw, -1)).transpose(2, 0, 1))

        input_record["rpn_cls_label"] = np.stack(cls_labels)
        input_record["rpn_reg_target"] = np.stack(reg_targets)
        input_record["rpn_reg_weight"] = np.stack(reg_weights)

        return input_record["rpn_cls_label"], \
               input_record["rpn_reg_target"], \
               input_record["rpn_reg_weight"]

