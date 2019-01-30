from __future__ import print_function

import numpy as np
import mxnet as mx
import mxnext as X

from symbol.builder import RpnHead, Backbone
from models.tridentnet.resnet_v2 import TridentResNetV2Builder


class TridentFasterRcnn(object):
    def __init__(self):
        super(TridentFasterRcnn, self).__init__()

    @staticmethod
    def get_train_symbol(backbone, neck, rpn_head, roi_extractor, bbox_head, num_branch, scaleaware):
        gt_bbox = X.var("gt_bbox")
        im_info = X.var("im_info")
        if scaleaware:
            valid_ranges = X.var("valid_ranges")
        rpn_cls_label = X.var("rpn_cls_label")
        rpn_reg_target = X.var("rpn_reg_target")
        rpn_reg_weight = X.var("rpn_reg_weight")

        im_info = TridentResNetV2Builder.stack_branch_symbols([im_info] * num_branch)
        gt_bbox = TridentResNetV2Builder.stack_branch_symbols([gt_bbox] * num_branch)
        if scaleaware:
            valid_ranges = X.reshape(valid_ranges, (-3, -2))
        rpn_cls_label = X.reshape(rpn_cls_label, (-3, -2))
        rpn_reg_target = X.reshape(rpn_reg_target, (-3, -2))
        rpn_reg_weight = X.reshape(rpn_reg_weight, (-3, -2))

        rpn_feat = backbone.get_rpn_feature()
        rcnn_feat = backbone.get_rcnn_feature()
        rpn_feat = neck.get_rpn_feature(rpn_feat)
        rcnn_feat = neck.get_rcnn_feature(rcnn_feat)

        rpn_loss = rpn_head.get_loss(rpn_feat, rpn_cls_label, rpn_reg_target, rpn_reg_weight)
        if scaleaware:
            proposal, bbox_cls, bbox_target, bbox_weight = rpn_head.get_sampled_proposal_with_filter(rpn_feat, gt_bbox, im_info, valid_ranges)
        else:
            proposal, bbox_cls, bbox_target, bbox_weight = rpn_head.get_sampled_proposal(rpn_feat, gt_bbox, im_info)
        roi_feat = roi_extractor.get_roi_feature(rcnn_feat, proposal)
        bbox_loss = bbox_head.get_loss(roi_feat, bbox_cls, bbox_target, bbox_weight)

        return X.group(rpn_loss + bbox_loss)

    @staticmethod
    def get_test_symbol(backbone, neck, rpn_head, roi_extractor, bbox_head, num_branch):
        im_info = X.var("im_info")
        im_id = X.var("im_id")
        rec_id = X.var("rec_id")

        im_info_branches = TridentResNetV2Builder.stack_branch_symbols([im_info] * num_branch)

        rpn_feat = backbone.get_rpn_feature()
        rcnn_feat = backbone.get_rcnn_feature()
        rpn_feat = neck.get_rpn_feature(rpn_feat)
        rcnn_feat = neck.get_rcnn_feature(rcnn_feat)

        proposal = rpn_head.get_all_proposal(rpn_feat, im_info_branches)
        roi_feat = roi_extractor.get_roi_feature(rcnn_feat, proposal)
        cls_score, bbox_xyxy = bbox_head.get_prediction(roi_feat, im_info_branches, proposal)

        cls_score = X.reshape(cls_score, (-3, -2))
        bbox_xyxy = X.reshape(bbox_xyxy, (-3, -2))

        return X.group([rec_id, im_id, im_info, cls_score, bbox_xyxy])


class TridentRpnHead(RpnHead):
    def __init__(self, pRpn):
        super(TridentRpnHead, self).__init__(pRpn)
        self.p = pRpn  # type: RPNParam

    def get_all_proposal_with_filter(self, conv_feat, im_info, valid_ranges):
        if self._proposal is not None:
            return self._proposal

        p = self.p
        rpn_stride = p.anchor_generate.stride
        anchor_scale = p.anchor_generate.scale
        anchor_ratio = p.anchor_generate.ratio
        pre_nms_top_n = p.proposal.pre_nms_top_n
        post_nms_top_n = p.proposal.post_nms_top_n
        nms_thr = p.proposal.nms_thr
        min_bbox_side = p.proposal.min_bbox_side

        cls_logit, bbox_delta = self.get_output(conv_feat)

        # TODO: remove this reshape hell
        cls_logit_reshape = X.reshape(
            cls_logit,
            shape=(0, -4, 2, -1, 0, 0),  # (N,C,H,W) -> (N,2,C/2,H,W)
            name="rpn_cls_logit_reshape_"
        )
        cls_score = X.softmax(
            cls_logit_reshape,
            axis=1,
            name='rpn_cls_score'
        )
        cls_logit_reshape = X.reshape(
            cls_score,
            shape=(0, -3, 0, 0),
            name='rpn_cls_score_reshape'
        )
        proposal = mx.sym.contrib.Proposal_v2(
            cls_prob=cls_logit_reshape,
            bbox_pred=bbox_delta,
            im_info=im_info,
            valid_ranges=valid_ranges,
            name='proposal',
            feature_stride=rpn_stride,
            scales=tuple(anchor_scale),
            ratios=tuple(anchor_ratio),
            rpn_pre_nms_top_n=pre_nms_top_n,
            rpn_post_nms_top_n=post_nms_top_n,
            threshold=nms_thr,
            rpn_min_size=min_bbox_side,
            iou_loss=False,
            filter_scales=True,
        )

        self._proposal = proposal

        return proposal

    def get_sampled_proposal_with_filter(self, conv_feat, gt_bbox, im_info, valid_ranges):
        p = self.p

        batch_image = p.batch_image

        proposal_wo_gt = p.subsample_proposal.proposal_wo_gt
        image_roi = p.subsample_proposal.image_roi
        fg_fraction = p.subsample_proposal.fg_fraction
        fg_thr = p.subsample_proposal.fg_thr
        bg_thr_hi = p.subsample_proposal.bg_thr_hi
        bg_thr_lo = p.subsample_proposal.bg_thr_lo

        num_reg_class = p.bbox_target.num_reg_class
        class_agnostic = p.bbox_target.class_agnostic
        bbox_target_weight = p.bbox_target.weight
        bbox_target_mean = p.bbox_target.mean
        bbox_target_std = p.bbox_target.std

        proposal = self.get_all_proposal_with_filter(conv_feat, im_info, valid_ranges)

        (bbox, label, bbox_target, bbox_weight) = mx.sym.ProposalTarget_v2(
            rois=proposal,
            gt_boxes=gt_bbox,
            valid_ranges=valid_ranges,
            num_classes=num_reg_class,
            class_agnostic=class_agnostic,
            batch_images=batch_image,
            proposal_without_gt=proposal_wo_gt,
            image_rois=image_roi,
            fg_fraction=fg_fraction,
            fg_thresh=fg_thr,
            bg_thresh_hi=bg_thr_hi,
            bg_thresh_lo=bg_thr_lo,
            bbox_weight=bbox_target_weight,
            bbox_mean=bbox_target_mean,
            bbox_std=bbox_target_std,
            filter_scales=True,
            name="subsample_proposal"
        )

        label = X.reshape(label, (-3, -2))
        bbox_target = X.reshape(bbox_target, (-3, -2))
        bbox_weight = X.reshape(bbox_weight, (-3, -2))

        return bbox, label, bbox_target, bbox_weight


class TridentMXNetResNetV2(Backbone):
    def __init__(self, pBackbone):
        super(TridentMXNetResNetV2, self).__init__(pBackbone)
        p = pBackbone
        b = TridentResNetV2Builder()
        self.symbol = b.get_backbone("mxnet", p.depth, "c4", p.normalizer, p.fp16,
                                     p.num_branch, p.branch_dilates, p.branch_ids,
                                     p.branch_bn_shared, p.branch_conv_shared, p.branch_deform)

    def get_rpn_feature(self):
        return self.symbol

    def get_rcnn_feature(self):
        return self.symbol


def process_branch_outputs(outputs, num_branch, valid_ranges_input=None, valid_ranges_on_origin=True):
    processed_outputs = []
    for j, output_ in enumerate(outputs):
        output = output_.copy()
        box = output['bbox_xyxy']
        cls = output['cls_score']
        valid_ranges = valid_ranges_input.copy()

        box_size = (box[:, 2] - box[:, 0] + 1.0) * (box[:, 3] - box[:, 1] + 1.0)
        if not valid_ranges_on_origin:
            valid_ranges = valid_ranges / output['im_info'][2]  # scale back to origin image size

        assert len(valid_ranges) == num_branch

        box_size = box_size.reshape(num_branch, -1)
        box = box.reshape(num_branch, -1, box.shape[1])
        cls = cls.reshape(num_branch, -1, cls.shape[1])

        valid_box, valid_cls = [], []
        for i in range(num_branch):
            range_low = valid_ranges[i][0]
            range_high = valid_ranges[i][1] if valid_ranges[i][1] >= 0 else max(output['im_info'][0], output['im_info'][1])

            valid_ind = np.where((box_size[i, :] >= range_low ** 2) & (box_size[i, :] <= range_high ** 2))[0]
            valid_box.append(box[i, valid_ind, :])
            valid_cls.append(cls[i, valid_ind, :])
        output['bbox_xyxy'] = np.concatenate(valid_box)
        output['cls_score'] = np.concatenate(valid_cls)

        processed_outputs.append(output)

    return processed_outputs

