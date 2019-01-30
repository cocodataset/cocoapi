from __future__ import print_function

import mxnet as mx
import mxnext as X

from symbol.builder import Backbone, BboxHead
from models.FPN import assign_layer_fpn, get_topk_proposal


class FPNBbox2fcHead(BboxHead):
    def __init__(self, pBbox):
        super(FPNBbox2fcHead, self).__init__(pBbox)

    def _get_bbox_head_logit(self, conv_feat):
        if self._head_feat is not None:
            return self._head_feat

        xavier_init = mx.init.Xavier(factor_type="in", rnd_type="uniform", magnitude=3)

        flatten = X.flatten(conv_feat, name="bbox_feat_flatten")
        fc1 = X.fc(flatten, filter=1024, name="bbox_fc1", init=xavier_init)
        fc1 = X.relu(fc1)
        fc2 = X.fc(fc1, filter=1024, name="bbox_fc2", init=xavier_init)
        fc2 = X.relu(fc2)

        self._head_feat = fc2

        return self._head_feat


class FPNRpnHead(object):
    def __init__(self, pRpn):
        self.p = pRpn  # type: RPNParam

        self.cls_logit_dict         = None
        self.bbox_delta_dict        = None
        self._proposal              = None
        self._proposal_scores       = None

    def get_output(self, conv_fpn_feat):
        if self.cls_logit_dict is not None and self.bbox_delta_dict is not None:
            return self.cls_logit_dict, self.bbox_delta_dict

        p = self.p
        num_base_anchor = len(p.anchor_generate.ratio) * len(p.anchor_generate.scale)
        conv_channel = p.head.conv_channel

        # FPN RPN share weight
        normal001 = mx.init.Normal(sigma=0.01)
        rpn_conv_weight = X.var('rpn_conv_weight', init=normal001)
        rpn_conv_bias = X.var('rpn_conv_bias', init=X.zero_init())
        rpn_conv_cls_weight = X.var('rpn_conv_cls_weight', init=normal001)
        rpn_conv_cls_bias = X.var('rpn_conv_cls_bias', init=X.zero_init())
        rpn_conv_bbox_weight = X.var('rpn_conv_bbox_weight', init=normal001)
        rpn_conv_bbox_bias = X.var('rpn_conv_bbox_bias', init=X.zero_init())

        cls_logit_dict = {}
        bbox_delta_dict = {}

        for stride in p.anchor_generate.stride:
            rpn_conv = X.conv(
                conv_fpn_feat['stride%s' % stride],
                kernel=3,
                filter=conv_channel,
                name="rpn_conv_3x3_%s" % stride,
                no_bias=False,
                weight=rpn_conv_weight,
                bias=rpn_conv_bias
            )
            rpn_relu = X.relu(rpn_conv, name='rpn_relu_%s' % stride)
            if p.fp16:
                rpn_relu = X.to_fp32(rpn_conv, name="rpn_relu_%s_fp32" % stride)
            cls_logit = X.conv(
                rpn_relu,
                filter=2 * num_base_anchor,
                name="rpn_cls_score_stride%s" % stride,
                no_bias=False,
                weight=rpn_conv_cls_weight,
                bias=rpn_conv_cls_bias
            )

            bbox_delta = X.conv(
                rpn_relu,
                filter=4 * num_base_anchor,
                name="rpn_bbox_pred_stride%s" % stride,
                no_bias=False,
                weight=rpn_conv_bbox_weight,
                bias=rpn_conv_bbox_bias
            )

            cls_logit_dict[stride]  = cls_logit
            bbox_delta_dict[stride] = bbox_delta

        self.cls_logit_dict = cls_logit_dict
        self.bbox_delta_dict = bbox_delta_dict

        return self.cls_logit_dict, self.bbox_delta_dict

    def get_anchor_target(self, conv_fpn_feat):
        raise NotImplementedError

    def get_loss(self, conv_fpn_feat, cls_label, bbox_target, bbox_weight):
        p = self.p
        batch_image = p.batch_image
        image_anchor = p.anchor_generate.image_anchor
        rpn_stride = p.anchor_generate.stride

        cls_logit_dict, bbox_delta_dict = self.get_output(conv_fpn_feat)

        scale_loss_shift = 128.0 if p.fp16 else 1.0

        rpn_cls_logit_list = []
        rpn_bbox_delta_list = []

        for stride in rpn_stride:
            rpn_cls_logit = cls_logit_dict[stride]
            rpn_bbox_delta = bbox_delta_dict[stride]
            rpn_cls_logit_reshape = X.reshape(data=rpn_cls_logit,
                                              shape=(0, 2, -1),
                                              name="rpn_cls_score_reshape_stride%s" % stride)
            rpn_bbox_delta_reshape = X.reshape(data=rpn_bbox_delta,
                                              shape=(0, 0, -1),
                                              name="rpn_bbox_pred_reshape_stride%s" % stride)
            rpn_bbox_delta_list.append(rpn_bbox_delta_reshape)
            rpn_cls_logit_list.append(rpn_cls_logit_reshape)

        # concat output of each level
        rpn_bbox_delta_concat = X.concat(rpn_bbox_delta_list, axis=2, name="rpn_bbox_pred_concat")
        rpn_cls_logit_concat = X.concat(rpn_cls_logit_list, axis=2, name="rpn_cls_score_concat")

        cls_loss = X.softmax_output(
            data=rpn_cls_logit_concat,
            label=cls_label,
            multi_output=True,
            normalization='valid',
            use_ignore=True,
            ignore_label=-1,
            grad_scale=1.0 * scale_loss_shift,
            name="rpn_cls_loss"
        )

        # regression loss
        reg_loss = X.smooth_l1(
            (rpn_bbox_delta_concat - bbox_target),
            scalar=3.0,
            name='rpn_reg_l1'
        )
        reg_loss = bbox_weight * reg_loss
        reg_loss = X.loss(
            reg_loss,
            grad_scale=1.0 / (batch_image * image_anchor) * scale_loss_shift,
            name='rpn_reg_loss'
        )
        return cls_loss, reg_loss

    def get_all_proposal(self, conv_fpn_feat, im_info):
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
        num_anchors = len(p.anchor_generate.ratio) * len(p.anchor_generate.scale)

        cls_logit_dict, bbox_delta_dict = self.get_output(conv_fpn_feat)

        # rpn rois for multi level feature
        proposal_list = []
        proposal_scores_list = []
        for stride in rpn_stride:
            rpn_cls_logit = cls_logit_dict[stride]
            rpn_bbox_delta = bbox_delta_dict[stride]
            # ROI Proposal
            rpn_cls_logit_reshape = X.reshape(
                data=rpn_cls_logit,
                shape=(0, 2, -1, 0),
                name="rpn_cls_logit_reshape_stride%s" % stride)
            rpn_cls_score = mx.symbol.SoftmaxActivation(
                data=rpn_cls_logit_reshape,
                mode="channel",
                name="rpn_cls_score_stride%s" % stride)
            rpn_cls_score_reshape = X.reshape(
                data=rpn_cls_score,
                shape=(0, 2 * num_anchors, -1, 0),
                name="rpn_cls_score_reshape_stride%s" % stride)
            rpn_proposal, rpn_proposal_scores = mx.sym.contrib.Proposal_v3(
                cls_prob=rpn_cls_score_reshape,
                bbox_pred=rpn_bbox_delta,
                im_info=im_info,
                rpn_pre_nms_top_n=pre_nms_top_n,
                rpn_post_nms_top_n=post_nms_top_n,
                feature_stride=stride,
                output_score=True,
                scales=tuple(anchor_scale),
                ratios=tuple(anchor_ratio),
                rpn_min_size=min_bbox_side,
                threshold=nms_thr,
                iou_loss=False)
            proposal_list.append(rpn_proposal)
            proposal_scores_list.append(rpn_proposal_scores)

        # concat output rois of each level
        proposal_concat = X.concat(proposal_list, axis=1, name="proposal_concat")
        proposal_scores_concat = X.concat(proposal_scores_list, axis=1, name="proposal_scores_concat")

        proposal = mx.symbol.Custom(rois=proposal_concat, rois_scores=proposal_scores_concat,
                                    op_type='get_top_proposal', rpn_post_nms_top_n=post_nms_top_n)

        self._proposal = proposal

        return proposal

    def get_sampled_proposal(self, conv_fpn_feat, gt_bbox, im_info):
        p = self.p

        batch_image = p.batch_image

        proposal_wo_gt = p.subsample_proposal.proposal_wo_gt
        image_roi = p.subsample_proposal.image_roi
        fg_fraction = p.subsample_proposal.fg_fraction
        fg_thr = p.subsample_proposal.fg_thr
        bg_thr_hi = p.subsample_proposal.bg_thr_hi
        bg_thr_lo = p.subsample_proposal.bg_thr_lo
        post_nms_top_n = p.proposal.post_nms_top_n

        num_reg_class = p.bbox_target.num_reg_class
        class_agnostic = p.bbox_target.class_agnostic
        bbox_target_weight = p.bbox_target.weight
        bbox_target_mean = p.bbox_target.mean
        bbox_target_std = p.bbox_target.std

        proposal = self.get_all_proposal(conv_fpn_feat, im_info)

        (bbox, label, bbox_target, bbox_weight) = X.proposal_target(
            rois=proposal,
            gt_boxes=gt_bbox,
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
            name="subsample_proposal"
        )

        label = X.reshape(label, (-3, -2))
        bbox_target = X.reshape(bbox_target, (-3, -2))
        bbox_weight = X.reshape(bbox_weight, (-3, -2))

        return bbox, label, bbox_target, bbox_weight


class MSRAResNet50V1FPN(Backbone):
    def __init__(self, pBackbone):
        super(MSRAResNet50V1FPN, self).__init__(pBackbone)
        from mxnext.backbone.resnet_v1 import Builder
        b = Builder()
        self.symbol = b.get_backbone("msra", 50, "fpn", pBackbone.normalizer, pBackbone.fp16)

    def get_rpn_feature(self):
        return self.symbol

    def get_rcnn_feature(self):
        return self.symbol


class MSRAResNet101V1FPN(Backbone):
    def __init__(self, pBackbone):
        super(MSRAResNet101V1FPN, self).__init__(pBackbone)
        from mxnext.backbone.resnet_v1 import Builder
        b = Builder()
        self.symbol = b.get_backbone("msra", 101, "fpn", pBackbone.normalizer, pBackbone.fp16)

    def get_rpn_feature(self):
        return self.symbol

    def get_rcnn_feature(self):
        return self.symbol


class Neck(object):
    def __init__(self):
        pass

    def get_rpn_feature(self, rpn_feat):
        return rpn_feat

    def get_rcnn_feature(self, rcnn_feat):
        return rcnn_feat


class FPNConvTopDown(Neck):
    def __init__(self, pNeck):
        super(FPNConvTopDown, self).__init__()
        self.fpn_feat = None
        self.p = pNeck

    def fpn_conv_down(self, data):
        if self.fpn_feat:
            return self.fpn_feat

        c2, c3, c4, c5 = data

        if self.p.fp16:
            c2 = X.to_fp32(c2, name="c2_to_fp32")
            c3 = X.to_fp32(c3, name="c3_to_fp32")
            c4 = X.to_fp32(c4, name="c4_to_fp32")
            c5 = X.to_fp32(c5, name="c5_to_fp32")

        xavier_init = mx.init.Xavier(factor_type="in", rnd_type="uniform", magnitude=3)

        # P5
        p5 = X.conv(data=c5,
                    filter=256,
                    no_bias=False,
                    weight=X.var(name="P5_lateral_weight", init=xavier_init),
                    bias=X.var(name="P5_lateral_bias", init=X.zero_init()),
                    name="P5_lateral")
        p5_conv = X.conv(data=p5,
                         kernel=3,
                         filter=256,
                         no_bias=False,
                         weight=X.var(name="P5_conv_weight", init=xavier_init),
                         bias=X.var(name="P5_conv_bias", init=X.zero_init()),
                         name="P5_conv")

        # P4
        p5_up = mx.sym.UpSampling(p5, scale=2, sample_type="nearest", name="P5_upsampling", num_args=1)
        p4_la = X.conv(data=c4,
                       filter=256,
                       no_bias=False,
                       weight=X.var(name="P4_lateral_weight", init=xavier_init),
                        bias=X.var(name="P4_lateral_bias", init=X.zero_init()),
                       name="P4_lateral")
        p5_clip = mx.sym.Crop(*[p5_up, p4_la], name="P4_clip")
        p4 = mx.sym.ElementWiseSum(*[p5_clip, p4_la], name="P4_sum")

        p4_conv = X.conv(data=p4,
                         kernel=3,
                         filter=256,
                         no_bias=False,
                         weight=X.var(name="P4_conv_weight", init=xavier_init),
                         bias=X.var(name="P4_conv_bias", init=X.zero_init()),
                         name="P4_conv")

        # P3
        p4_up = mx.sym.UpSampling(p4, scale=2, sample_type="nearest", name="P4_upsampling", num_args=1)
        p3_la = X.conv(data=c3,
                       filter=256,
                       no_bias=False,
                       weight=X.var(name="P3_lateral_weight", init=xavier_init),
                       bias=X.var(name="P3_lateral_bias", init=X.zero_init()),
                       name="P3_lateral")
        p4_clip = mx.sym.Crop(*[p4_up, p3_la], name="P3_clip")
        p3 = mx.sym.ElementWiseSum(*[p4_clip, p3_la], name="P3_sum")

        p3_conv = X.conv(data=p3,
                         kernel=3,
                         filter=256,
                         no_bias=False,
                         weight=X.var(name="P3_conv_weight", init=xavier_init),
                         bias=X.var(name="P3_conv_bias", init=X.zero_init()),
                         name="P3_conv")

        # P2
        p3_up = mx.sym.UpSampling(p3, scale=2, sample_type="nearest", name="P3_upsampling", num_args=1)
        p2_la = X.conv(data=c2,
                       filter=256,
                       no_bias=False,
                       weight=X.var(name="P2_lateral_weight", init=xavier_init),
                       bias=X.var(name="P2_lateral_bias", init=X.zero_init()),
                       name="P2_lateral")
        p3_clip = mx.sym.Crop(*[p3_up, p2_la], name="P2_clip")
        p2 = mx.sym.ElementWiseSum(*[p3_clip, p2_la], name="P2_sum")

        p2_conv = X.conv(data=p2,
                         kernel=3,
                         filter=256,
                         no_bias=False,
                         weight=X.var(name="P2_conv_weight", init=xavier_init),
                         bias=X.var(name="P2_conv_bias", init=X.zero_init()),
                         name="P2_conv")

        # P6
        p6 = X.pool(p5_conv, name="P6_subsampling", kernel=1, stride=2, pad=0, pool_type='max')
        if self.p.fp16:
            p6 = X.to_fp16(p6, name="p6_to_fp16")
            p5_conv = X.to_fp16(p5_conv, name="p5_conv_to_fp16")
            p4_conv = X.to_fp16(p4_conv, name="p4_conv_to_fp16")
            p3_conv = X.to_fp16(p3_conv, name="p3_conv_to_fp16")
            p2_conv = X.to_fp16(p2_conv, name="p2_conv_to_fp16")

        conv_fpn_feat = dict()
        conv_fpn_feat.update({"stride64": p6, "stride32": p5_conv, "stride16": p4_conv, "stride8": p3_conv, "stride4": p2_conv})

        self.fpn_feat = conv_fpn_feat
        return self.fpn_feat

    def get_rpn_feature(self, rpn_feat):
        return self.fpn_conv_down(rpn_feat)

    def get_rcnn_feature(self, rcnn_feat):
        return self.fpn_conv_down(rcnn_feat)


class FPNRoiExtractor(object):
    def __init__(self, pRoi):
        self.p = pRoi  # type: RoiParam

    def get_roi_feature(self, rcnn_feat, proposal):
        pass


class FPNRoiAlign(FPNRoiExtractor):
    def __init__(self, pRoi):
        super(FPNRoiAlign, self).__init__(pRoi)

    def get_roi_feature_test(self, conv_fpn_feat, proposals):
        return self.get_roi_feature(conv_fpn_feat, proposals)

    def get_roi_feature(self, conv_fpn_feat, proposal):
        p = self.p
        rcnn_stride = p.stride

        group = mx.symbol.Custom(rois=proposal, op_type='assign_layer_fpn')
        proposal_fpn = dict()
        proposal_fpn["stride4"] = group[1]
        proposal_fpn["stride8"] = group[2]
        proposal_fpn["stride16"] = group[3]
        proposal_fpn["stride32"] = group[4]

        if p.fp16:
            for stride in rcnn_stride:
                conv_fpn_feat["stride%s" % stride] = X.to_fp32(conv_fpn_feat["stride%s" % stride], name="fpn_stride%s_to_fp32")

        fpn_roi_feats = list()
        for stride in rcnn_stride:
            feat_lvl = conv_fpn_feat["stride%s" % stride]
            proposal_lvl = proposal_fpn["stride%s" % stride]
            roi_feat = X.roi_align(
                feat_lvl,
                rois=proposal_lvl,
                out_size=p.out_size,
                stride=stride,
                name="roi_align"
            )
            roi_feat = X.reshape(data=roi_feat, shape=(-3, -2), name='roi_feat_reshape')
            fpn_roi_feats.append(roi_feat)
        roi_feat = X.add_n(*fpn_roi_feats)

        if p.fp16:
            roi_feat = X.to_fp16(roi_feat, name="roi_feat_to_fp16")

        return roi_feat
