from __future__ import print_function

import mxnext as X


class FasterRcnn(object):
    def __init__(self):
        pass

    @staticmethod
    def get_train_symbol(backbone, neck, rpn_head, roi_extractor, bbox_head):
        gt_bbox = X.var("gt_bbox")
        im_info = X.var("im_info")
        rpn_cls_label = X.var("rpn_cls_label")
        rpn_reg_target = X.var("rpn_reg_target")
        rpn_reg_weight = X.var("rpn_reg_weight")

        rpn_feat = backbone.get_rpn_feature()
        rcnn_feat = backbone.get_rcnn_feature()
        rpn_feat = neck.get_rpn_feature(rpn_feat)
        rcnn_feat = neck.get_rcnn_feature(rcnn_feat)

        rpn_loss = rpn_head.get_loss(rpn_feat, rpn_cls_label, rpn_reg_target, rpn_reg_weight)
        proposal, bbox_cls, bbox_target, bbox_weight = rpn_head.get_sampled_proposal(rpn_feat, gt_bbox, im_info)
        roi_feat = roi_extractor.get_roi_feature(rcnn_feat, proposal)
        bbox_loss = bbox_head.get_loss(roi_feat, bbox_cls, bbox_target, bbox_weight)

        return X.group(rpn_loss + bbox_loss)

    @staticmethod
    def get_test_symbol(backbone, neck, rpn_head, roi_extractor, bbox_head):
        im_info = X.var("im_info")
        im_id = X.var("im_id")
        rec_id = X.var("rec_id")

        rpn_feat = backbone.get_rpn_feature()
        rcnn_feat = backbone.get_rcnn_feature()
        rpn_feat = neck.get_rpn_feature(rpn_feat)
        rcnn_feat = neck.get_rcnn_feature(rcnn_feat)

        proposal = rpn_head.get_all_proposal(rpn_feat, im_info)
        roi_feat = roi_extractor.get_roi_feature_test(rcnn_feat, proposal)
        cls_score, bbox_xyxy = bbox_head.get_prediction(roi_feat, im_info, proposal)

        return X.group([rec_id, im_id, im_info, cls_score, bbox_xyxy])


class RpnHead(object):
    def __init__(self, pRpn):
        self.p = pRpn  # type: RPNParam

        self._cls_logit             = None
        self._bbox_delta            = None
        self._proposal              = None

    def get_output(self, conv_feat):
        if self._cls_logit is not None and self._bbox_delta is not None:
            return self._cls_logit, self._bbox_delta

        p = self.p
        num_base_anchor = len(p.anchor_generate.ratio) * len(p.anchor_generate.scale)
        conv_channel = p.head.conv_channel

        conv = X.convrelu(
            conv_feat,
            kernel=3,
            filter=conv_channel,
            name="rpn_conv_3x3",
            no_bias=False,
            init=X.gauss(0.01)
        )

        if p.fp16:
            conv = X.to_fp32(conv, name="rpn_conv_3x3_fp32")

        cls_logit = X.conv(
            conv,
            filter=2 * num_base_anchor,
            name="rpn_cls_logit",
            no_bias=False,
            init=X.gauss(0.01)
        )

        bbox_delta = X.conv(
            conv,
            filter=4 * num_base_anchor,
            name="rpn_bbox_delta",
            no_bias=False,
            init=X.gauss(0.01)
        )

        self._cls_logit = cls_logit
        self._bbox_delta = bbox_delta

        return self._cls_logit, self._bbox_delta

    def get_anchor_target(self, conv_feat):
        raise NotImplementedError

    def get_loss(self, conv_feat, cls_label, bbox_target, bbox_weight):
        p = self.p
        batch_image = p.batch_image
        image_anchor = p.anchor_generate.image_anchor

        cls_logit, bbox_delta = self.get_output(conv_feat)

        scale_loss_shift = 128.0 if p.fp16 else 1.0

        # classification loss
        cls_logit_reshape = X.reshape(
            cls_logit,
            shape=(0, -4, 2, -1, 0, 0),  # (N,C,H,W) -> (N,2,C/2,H,W)
            name="rpn_cls_logit_reshape"
        )
        cls_loss = X.softmax_output(
            data=cls_logit_reshape,
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
            (bbox_delta - bbox_target),
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

    def get_all_proposal(self, conv_feat, im_info):
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

        # TODO: ask all to add is_train filed in RPNParam
        proposal = X.proposal(
            cls_prob=cls_logit_reshape,
            bbox_pred=bbox_delta,
            im_info=im_info,
            name='proposal',
            feature_stride=rpn_stride,
            scales=tuple(anchor_scale),
            ratios=tuple(anchor_ratio),
            rpn_pre_nms_top_n=pre_nms_top_n,
            rpn_post_nms_top_n=post_nms_top_n,
            threshold=nms_thr,
            rpn_min_size=min_bbox_side,
            iou_loss=False
        )

        self._proposal = proposal

        return proposal

    def get_sampled_proposal(self, conv_feat, gt_bbox, im_info):
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

        proposal = self.get_all_proposal(conv_feat, im_info)

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


class BboxHead(object):
    def __init__(self, pBbox):
        self.p = pBbox  # type: BboxParam

        self._head_feat = None

    def _get_bbox_head_logit(self, conv_feat):
        raise NotImplemented

    def get_output(self, conv_feat):
        p = self.p
        num_class = p.num_class
        num_reg_class = 2 if p.regress_target.class_agnostic else num_class

        head_feat = self._get_bbox_head_logit(conv_feat)

        if p.fp16:
            head_feat = X.to_fp32(head_feat, name="bbox_head_to_fp32")

        cls_logit = X.fc(
            head_feat,
            filter=num_class,
            name='bbox_cls_logit',
            init=X.gauss(0.01)
        )
        bbox_delta = X.fc(
            head_feat,
            filter=4 * num_reg_class,
            name='bbox_reg_delta',
            init=X.gauss(0.001)
        )

        return cls_logit, bbox_delta

    def get_prediction(self, conv_feat, im_info, proposal):
        p = self.p
        bbox_mean = p.regress_target.mean
        bbox_std = p.regress_target.std
        batch_image = p.batch_image
        num_class = p.num_class
        class_agnostic = p.regress_target.class_agnostic
        num_reg_class = 2 if class_agnostic else num_class

        cls_logit, bbox_delta = self.get_output(conv_feat)

        bbox_delta = X.reshape(
            bbox_delta,
            shape=(batch_image, -1, 4 * num_reg_class),
            name='bbox_delta_reshape'
        )

        bbox_xyxy = X.decode_bbox(
            rois=proposal,
            bbox_pred=bbox_delta,
            im_info=im_info,
            name='decode_bbox',
            bbox_mean=bbox_mean,
            bbox_std=bbox_std,
            class_agnostic=class_agnostic
        )
        cls_score = X.softmax(
            cls_logit,
            axis=-1,
            name='bbox_cls_score'
        )
        cls_score = X.reshape(
            cls_score,
            shape=(batch_image, -1, num_class),
            name='bbox_cls_score_reshape'
        )
        return cls_score, bbox_xyxy

    def get_loss(self, conv_feat, cls_label, bbox_target, bbox_weight):
        p = self.p
        batch_roi = p.image_roi * p.batch_image
        batch_image = p.batch_image

        cls_logit, bbox_delta = self.get_output(conv_feat)

        scale_loss_shift = 128.0 if p.fp16 else 1.0

        # classification loss
        cls_loss = X.softmax_output(
            data=cls_logit,
            label=cls_label,
            normalization='batch',
            grad_scale=1.0 * scale_loss_shift,
            name='bbox_cls_loss'
        )

        # bounding box regression
        reg_loss = X.smooth_l1(
            bbox_delta - bbox_target,
            scalar=1.0,
            name='bbox_reg_l1'
        )
        reg_loss = bbox_weight * reg_loss
        reg_loss = X.loss(
            reg_loss,
            grad_scale=1.0 / batch_roi * scale_loss_shift,
            name='bbox_reg_loss',
        )

        # append label
        cls_label = X.reshape(
            cls_label,
            shape=(batch_image, -1),
            name='bbox_label_reshape'
        )
        cls_label = X.block_grad(cls_label, name='bbox_label_blockgrad')

        # output
        return cls_loss, reg_loss, cls_label


class Bbox2fcHead(BboxHead):
    def __init__(self, pBbox):
        super(Bbox2fcHead, self).__init__(pBbox)

    def _get_bbox_head_logit(self, conv_feat):
        if self._head_feat is not None:
            return self._head_feat

        flatten = X.flatten(conv_feat, name="bbox_feat_flatten")
        reshape = X.reshape(flatten, (0, 0, 1, 1), name="bbox_feat_reshape")
        fc1 = X.convrelu(reshape, filter=1024, name="bbox_fc1")
        fc2 = X.convrelu(fc1, filter=1024, name="bbox_fc2")

        self._head_feat = fc2

        return self._head_feat


class BboxC5Head(BboxHead):
    def __init__(self, pBbox):
        super(BboxC5Head, self).__init__(pBbox)

    def _get_bbox_head_logit(self, conv_feat):
        if self._head_feat is not None:
            return self._head_feat

        from mxnext.backbone.resnet_v2 import Builder

        unit = Builder.resnet_stage(
            conv_feat,
            name="stage4",
            num_block=3,
            filter=2048,
            stride=1,
            dilate=1,
            norm_type=self.p.normalizer,
            norm_mom=0.9,
            ndev=8
        )
        bn1 = X.fixbn(unit, name='bn1')
        relu1 = X.relu(bn1, name='relu1')
        relu1 = X.to_fp32(relu1, name='c5_to_fp32')
        pool1 = X.pool(relu1, global_pool=True, name='pool1')

        self._head_feat = pool1

        return self._head_feat


class BboxResNeXtC5Head(BboxHead):
    def __init__(self, pBbox):
        super(BboxResNeXtC5Head, self).__init__(pBbox)

    def _get_bbox_head_logit(self, conv_feat):
        if self._head_feat is not None:
            return self._head_feat

        from mxnext.backbone.resnext import Builder

        unit = Builder.resnext_stage(
            conv_feat,
            name="stage4",
            num_block=3,
            filter=2048,
            stride=1,
            dilate=1,
            num_group=self.p.num_group,
            norm_type=self.p.normalizer,
            norm_mom=0.9,
            ndev=8
        )
        pool1 = X.pool(unit, global_pool=True, name='pool1')

        self._head_feat = pool1

        return self._head_feat


class BboxC5V1Head(BboxHead):
    def __init__(self, pBbox):
        super(BboxC5V1Head, self).__init__(pBbox)

    def _get_bbox_head_logit(self, conv_feat):
        if self._head_feat is not None:
            return self._head_feat

        from mxnext.backbone.resnet_v1 import Builder

        unit = Builder.resnet_stage(
            conv_feat,
            name="stage4",
            num_block=3,
            filter=2048,
            stride=1,
            dilate=1,
            norm_type=self.p.normalizer,
            norm_mom=0.9,
            ndev=8
        )
        unit = X.to_fp32(unit, name='c5_to_fp32')
        pool1 = X.pool(unit, global_pool=True, name='pool1')

        self._head_feat = pool1

        return self._head_feat


class Backbone(object):
    def __init__(self, pBackbone):
        self.pBackbone = pBackbone

    def get_rpn_feature(self):
        raise NotImplementedError

    def get_rcnn_feature(self):
        raise NotImplementedError


class MXNetResNet50V2(Backbone):
    def __init__(self, pBackbone):
        super(MXNetResNet50V2, self).__init__(pBackbone)
        from mxnext.backbone.resnet_v2 import Builder
        b = Builder()
        self.symbol = b.get_backbone("mxnet", 50, "c4", pBackbone.normalizer, pBackbone.fp16)

    def get_rpn_feature(self):
        return self.symbol

    def get_rcnn_feature(self):
        return self.symbol


class MXNetResNeXt50(Backbone):
    def __init__(self, pBackbone):
        super(MXNetResNeXt50, self).__init__(pBackbone)
        from mxnext.backbone.resnext import Builder
        b = Builder()
        self.symbol = b.get_backbone("mxnet", 50, "c4", pBackbone.normalizer, pBackbone.num_group, pBackbone.fp16)

    def get_rpn_feature(self):
        return self.symbol

    def get_rcnn_feature(self):
        return self.symbol


class MXNetResNet101V2(Backbone):
    def __init__(self, pBackbone):
        super(MXNetResNet101V2, self).__init__(pBackbone)
        from mxnext.backbone.resnet_v2 import Builder
        b = Builder()
        self.symbol = b.get_backbone("mxnet", 101, "c4", pBackbone.normalizer, pBackbone.fp16)

    def get_rpn_feature(self):
        return self.symbol

    def get_rcnn_feature(self):
        return self.symbol


class ResNet50V1(Backbone):
    def __init__(self, pBackbone):
        super(ResNet50V1, self).__init__(pBackbone)
        from mxnext.backbone.resnet_v1 import Builder
        b = Builder()
        self.symbol = b.get_backbone("msra", 50, "c4", pBackbone.normalizer, pBackbone.fp16)

    def get_rpn_feature(self):
        return self.symbol

    def get_rcnn_feature(self):
        return self.symbol


class ResNet101V1(Backbone):
    def __init__(self, pBackbone):
        super(ResNet101V1, self).__init__(pBackbone)
        from mxnext.backbone.resnet_v1 import Builder
        b = Builder()
        self.symbol = b.get_backbone("msra", 101, "c4", pBackbone.normalizer, pBackbone.fp16)

    def get_rpn_feature(self):
        return self.symbol

    def get_rcnn_feature(self):
        return self.symbol


class MXNetResNet50V2C4C5(Backbone):
    def __init__(self, pBackbone):
        super(MXNetResNet50V2C4C5, self).__init__(pBackbone)
        from mxnext.backbone.resnet_v2 import Builder
        b = Builder()
        self.c4, self.c5 = b.get_backbone("mxnet", 50, "c4c5", pBackbone.normalizer, pBackbone.fp16)

    def get_rpn_feature(self):
        return self.c4

    def get_rcnn_feature(self):
        return self.c5


class MXNetResNet101V2C4C5(Backbone):
    def __init__(self, pBackbone):
        super(MXNetResNet101V2C4C5, self).__init__(pBackbone)
        from mxnext.backbone.resnet_v2 import Builder
        b = Builder()
        self.c4, self.c5 = b.get_backbone("mxnet", 101, "c4c5", pBackbone.normalizer, pBackbone.fp16)

    def get_rpn_feature(self):
        return self.c4

    def get_rcnn_feature(self):
        return self.c5


class Neck(object):
    def __init__(self, pNeck):
        self.pNeck = pNeck

    def get_rpn_feature(self, rpn_feat):
        return rpn_feat

    def get_rcnn_feature(self, rcnn_feat):
        return rcnn_feat


class RoiExtractor(object):
    def __init__(self, pRoi):
        self.p = pRoi  # type: RoiParam

    def get_roi_feature(self, rcnn_feat, proposal):
        raise NotImplementedError

    def get_roi_feature_test(self, rcnn_feat, proposal):
        raise NotImplementedError


class RoiAlign(RoiExtractor):
    def __init__(self, pRoi):
        super(RoiAlign, self).__init__(pRoi)

    def get_roi_feature(self, rcnn_feat, proposal):
        p = self.p

        if p.fp16:
            rcnn_feat = X.to_fp32(rcnn_feat, "rcnn_feat_to_fp32")

        roi_feat = X.roi_align(
            rcnn_feat,
            rois=proposal,
            out_size=p.out_size,
            stride=p.stride,
            name="roi_align"
        )

        if p.fp16:
            roi_feat = X.to_fp16(roi_feat, "roi_feat_to_fp16")

        roi_feat = X.reshape(roi_feat, (-3, -2))

        return roi_feat

    def get_roi_feature_test(self, rcnn_feat, proposal):
        return self.get_roi_feature(rcnn_feat, proposal)
