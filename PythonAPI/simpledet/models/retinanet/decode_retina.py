"""
Decode boxes for RetinaNet
author: Chenxia Han
"""

import mxnet as mx
import numpy as np

from models.retinanet.input import PyramidAnchorTarget2D
from operator_py.bbox_transform import clip_boxes
from operator_py.bbox_transform import nonlinear_pred as decode_boxes

class AnchorTarget2DParam:
    def __init__(self, stride, scales, aspects):
        self.generate = self._generate()
        self.generate.stride = tuple(stride)
        self.generate.scales = tuple(scales)
        self.generate.aspects = tuple(aspects)

        # no use for generating base anchor
        self.generate.short = (None,) * len(stride)
        self.generate.long = (None,) * len(stride)

    class _generate:
        def __init__(self):
            self.short = None
            self.long = None
            self.stride = None

        scales = None
        aspects = None


class DecodeRetinaOperator(mx.operator.CustomOp):
    def __init__(self, stride, scales, ratios, per_level_top_n, thresh):
        super(DecodeRetinaOperator, self).__init__()
        self._stride = np.array(stride)
        self._pyramid_levels = len(self._stride)
        self._level_keys = ['stride%s'%s for s in self._stride]

        self._scales = np.array(scales)
        self._ratios = np.array(ratios)

        anchor_param = AnchorTarget2DParam(self._stride, self._scales, self._ratios)
        anchor_target_2d_list = PyramidAnchorTarget2D(anchor_param).anchor_target_2d_list
        anchor_list = [anchor_target_2d_list[i].base_anchor for i in range(self._pyramid_levels)]
        self._anchors_fpn = dict(zip(self._level_keys, anchor_list))

        self._per_level_top_n = per_level_top_n
        self._thresh = thresh

    def forward(self, is_train, req, in_data, out_data, aux):
        per_level_top_n = self._per_level_top_n
        pyramid_levels = self._pyramid_levels

        cls_prob_dict = dict(zip(self._level_keys, in_data[0:pyramid_levels]))
        bbox_pred_dict = dict(zip(self._level_keys, in_data[pyramid_levels:2*pyramid_levels]))

        A = len(self._scales) * len(self._ratios)
        num_class = in_data[0].shape[1] // A + 1

        im_info = in_data[-1].asnumpy()[0, :]
        batch_size = in_data[0].shape[0]
        if batch_size > 1:
            raise ValueError("Multiple images each device is not implemented")

        proposals_list = []
        scores_list = []
        pred_label_list = []
        for s in self._stride:
            cls_prob = cls_prob_dict['stride%s'%s].asnumpy()
            box_pred = bbox_pred_dict['stride%s'%s].asnumpy()

            cell_anchors = self._anchors_fpn['stride%s'%s]

            cls_prob = cls_prob.reshape((
                cls_prob.shape[0], A, cls_prob.shape[1] // A,
                cls_prob.shape[2], cls_prob.shape[3]))
            box_pred = box_pred.reshape((
                box_pred.shape[0], A, 4, box_pred.shape[2], box_pred.shape[3]))

            cls_prob_ravel = cls_prob.ravel()

            # In some cases [especially for very small img sizes], it's possible that
            # candidate_ind is empty if we impose threshold 0.05 at all levels. This
            # will lead to errors since no detections are found for this image. Hence,
            # for lvl 7 which has small spatial resolution, we take the threshold 0.0
            thresh = self._thresh if s != np.max(self._stride) else 0.0
            keep = np.where(cls_prob_ravel > thresh)[0]
            if (len(keep) == 0):
                continue

            pre_nms_top_n = min(per_level_top_n, len(keep))
            inds = np.argpartition(
                cls_prob_ravel[keep], -pre_nms_top_n)[-pre_nms_top_n:]
            inds = keep[inds]

            inds_5d = np.array(np.unravel_index(inds, cls_prob.shape)).transpose()
            classes = inds_5d[:, 2]
            anchor_ids, y, x = inds_5d[:, 1], inds_5d[:, 3], inds_5d[:, 4]
            scores = cls_prob[:, anchor_ids, classes, y, x]

            boxes = np.column_stack((x, y, x, y)).astype(dtype=np.float32)
            boxes *= s
            boxes += cell_anchors[anchor_ids, :]

            box_deltas = box_pred[0, anchor_ids, :, y, x]

            pred_boxes = (
                decode_boxes(boxes, box_deltas)
            )

            # pred_boxes: (pre_nms_top_n, 4)
            # scores: (1, pre_nms_top_n)
            # classes: (pre_nms_top_n,)
            proposals_list.append(pred_boxes)
            scores_list.append(scores.squeeze(axis=0))
            pred_label_list.append(classes)

        # merge fpn level
        proposals = np.concatenate(proposals_list, axis=0)
        scores = np.concatenate(scores_list, axis=0)
        pred_label = np.concatenate(pred_label_list, axis=0)

        # add background
        pred_label += 1

        # clip boxes
        proposals = clip_boxes(proposals, im_info[:2])

        keep_num = proposals.shape[0]

        boxes_ = np.zeros((1, per_level_top_n * pyramid_levels, 4))
        scores_ = np.zeros((1, per_level_top_n * pyramid_levels, num_class))

        inds = range(keep_num)
        boxes_[:,inds,:] = proposals
        scores_[:,inds,pred_label[inds]] = scores

        self.assign(out_data[0], req[0], boxes_)
        self.assign(out_data[1], req[1], scores_)


    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        raise NotImplementedError


@mx.operator.register("decode_retina")
class DecodeRetinaProp(mx.operator.CustomOpProp):
    def __init__(self, stride, scales, ratios, per_level_top_n, thresh):
        super(DecodeRetinaProp, self).__init__(need_top_grad=False)
        self._stride = eval(stride)
        self._scales = eval(scales)
        self._ratios = eval(ratios)
        self._pyramid_levels = len(self._stride)
        self._per_level_top_n = int(per_level_top_n)
        self._thresh = float(thresh)

    def list_arguments(self):
        args_list = []
        for s in self._stride:
            args_list.append('cls_logit_stride%s'%s)
        for s in self._stride:
            args_list.append('bbox_delta_stride%s'%s)
        args_list.append('im_info')

        return args_list

    def list_outputs(self):
        return ['boxes', 'scores']

    def infer_shape(self, in_shape):
        A = len(self._scales) * len(self._ratios)
        num_class = in_shape[0][1] // A + 1
        output_shape = (1, self._pyramid_levels*self._per_level_top_n, 4)
        score_shape = (1, self._pyramid_levels*self._per_level_top_n, num_class)

        return in_shape, [output_shape, score_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return DecodeRetinaOperator(self._stride, self._scales, self._ratios,
                                   self._per_level_top_n, self._thresh)

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []
