from __future__ import print_function

import mxnext as X
import mxnet as mx

from models.FPN.builder import FPNRpnHead, FPNRoiExtractor
from models.FPN import assign_layer_fpn, get_topk_proposal

from models.maskrcnn import bbox_post_processing


class MaskFasterRcnn(object):
    def __init__(self):
        pass

    @staticmethod
    def get_train_symbol(backbone, neck, rpn_head, roi_extractor, mask_roi_extractor, bbox_head, mask_head):
        gt_bbox = X.var("gt_bbox")
        gt_poly = X.var("gt_poly")
        im_info = X.var("im_info")
        rpn_cls_label = X.var("rpn_cls_label")
        rpn_reg_target = X.var("rpn_reg_target")
        rpn_reg_weight = X.var("rpn_reg_weight")

        rpn_feat = backbone.get_rpn_feature()
        rcnn_feat = backbone.get_rcnn_feature()
        rpn_feat = neck.get_rpn_feature(rpn_feat)
        rcnn_feat = neck.get_rcnn_feature(rcnn_feat)

        rpn_loss = rpn_head.get_loss(rpn_feat, rpn_cls_label, rpn_reg_target, rpn_reg_weight)
        proposal, bbox_cls, bbox_target, bbox_weight, mask_proposal, mask_target = \
            rpn_head.get_sampled_proposal(rpn_feat, gt_bbox, gt_poly, im_info)
        roi_feat = roi_extractor.get_roi_feature(rcnn_feat, proposal)
        mask_roi_feat = mask_roi_extractor.get_roi_feature(rcnn_feat, mask_proposal)

        bbox_loss = bbox_head.get_loss(roi_feat, bbox_cls, bbox_target, bbox_weight)
        mask_loss = mask_head.get_loss(mask_roi_feat, mask_target)
        return X.group(rpn_loss + bbox_loss + mask_loss)

    @staticmethod
    def get_test_symbol(backbone, neck, rpn_head, roi_extractor, mask_roi_extractor, bbox_head, mask_head, bbox_post_processor):
        im_info = X.var("im_info")
        im_id = X.var("im_id")
        rec_id = X.var("rec_id")

        rpn_feat = backbone.get_rpn_feature()
        rcnn_feat = backbone.get_rcnn_feature()
        rpn_feat = neck.get_rpn_feature(rpn_feat)
        rcnn_feat = neck.get_rcnn_feature(rcnn_feat)

        proposal = rpn_head.get_all_proposal(rpn_feat, im_info)
        roi_feat = roi_extractor.get_roi_feature(rcnn_feat, proposal)
        cls_score, bbox_xyxy = bbox_head.get_prediction(roi_feat, im_info, proposal)

        post_cls_score, post_bbox_xyxy, post_cls = bbox_post_processor.get_post_processing(cls_score, bbox_xyxy)

        mask_roi_feat = mask_roi_extractor.get_roi_feature(rcnn_feat, post_bbox_xyxy)
        mask = mask_head.get_prediction(mask_roi_feat)

        return X.group([rec_id, im_id, im_info, post_cls_score, post_bbox_xyxy, post_cls, mask])


class BboxPostProcessor(object):
    def __init__(self, pTest):
        super(BboxPostProcessor, self).__init__()
        self.p = pTest

    def get_post_processing(self, cls_score, bbox_xyxy):
        p = self.p
        max_det_per_image = p.max_det_per_image
        min_det_score = p.min_det_score
        nms_type = p.nms.type
        nms_thr = p.nms.thr

        post_cls_score, post_bbox_xyxy, post_cls = mx.sym.Custom(
            cls_score=cls_score,
            bbox_xyxy=bbox_xyxy,
            max_det_per_image = max_det_per_image,
            min_det_score = min_det_score,
            nms_type = nms_type,
            nms_thr = nms_thr,
            op_type='BboxPostProcessing')
        return post_cls_score, post_bbox_xyxy, post_cls


class MaskFPNRpnHead(FPNRpnHead):
    def __init__(self, pRpn, pMask):
        super(MaskFPNRpnHead, self).__init__(pRpn)
        self.pMask = pMask

    def get_sampled_proposal(self, conv_fpn_feat, gt_bbox, gt_poly, im_info):
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

        mask_size = self.pMask.resolution

        proposal = self.get_all_proposal(conv_fpn_feat, im_info)

        (bbox, label, bbox_target, bbox_weight, match_gt_iou, mask_target) = mx.sym.ProposalMaskTarget(
            rois=proposal,
            gt_boxes=gt_bbox,
            gt_polys=gt_poly,
            mask_size=mask_size,
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
            output_iou=True,
            name="subsample_proposal"
        )

        label = X.reshape(label, (-3, -2))
        bbox_target = X.reshape(bbox_target, (-3, -2))
        bbox_weight = X.reshape(bbox_weight, (-3, -2))
        mask_target = X.reshape(mask_target, (-3, -2))

        num_fg_rois_per_img = int(image_roi * fg_fraction)
        mask_proposal = mx.sym.slice_axis(
            bbox,
            axis=1,
            begin=0,
            end=num_fg_rois_per_img)

        return bbox, label, bbox_target, bbox_weight, mask_proposal, mask_target


class MaskFasterRcnnHead(object):
    def __init__(self, pBbox, pMask, pMaskRoi):
        self.pBbox = pBbox
        self.pMask = pMask
        self.pMaskRoi = pMaskRoi

        self._head_feat = None

    def _get_mask_head_logit(self, conv_feat):
        raise NotImplemented

    def get_output(self, conv_feat):
        pBbox = self.pBbox
        num_class = pBbox.num_class

        head_feat = self._get_mask_head_logit(conv_feat)

        msra_init = mx.init.Xavier(rnd_type="gaussian", factor_type="out", magnitude=2)

        if self.pMask:
            head_feat = X.to_fp32(head_feat, name="mask_head_to_fp32")

        mask_fcn_logit = X.conv(
            head_feat,
            filter=num_class,
            name="mask_fcn_logit",
            no_bias=False,
            init=msra_init
        )

        return mask_fcn_logit

    def get_prediction(self, conv_feat):
        """
        input: conv_feat, (1 * num_box, channel, pool_size, pool_size)
        """
        mask_fcn_logit = self.get_output(conv_feat)
        mask_prob = mx.symbol.Activation(
            data=mask_fcn_logit,
            act_type='sigmoid',
            name="mask_prob")
        return mask_prob

    def get_loss(self, conv_feat, mask_target):
        pBbox = self.pBbox
        pMask = self.pMask
        batch_image = pBbox.batch_image

        mask_fcn_logit = self.get_output(conv_feat)

        scale_loss_shift = 128.0 if pMask.fp16 else 1.0

        mask_fcn_logit = X.reshape(
            mask_fcn_logit,
            shape=(1, -1),
            name="mask_fcn_logit_reshape"
        )
        mask_target = X.reshape(
            mask_target,
            shape=(1, -1),
            name="mask_target_reshape"
        )
        mask_loss = mx.sym.contrib.SigmoidCrossEntropy(
            mask_fcn_logit,
            mask_target,
            grad_scale=1.0 * scale_loss_shift,
            name="mask_loss"
        )
        return (mask_loss,)


class MaskFasterRcnn4ConvHead(MaskFasterRcnnHead):
    def __init__(self, pBbox, pMask, pMaskRoi):
        super(MaskFasterRcnn4ConvHead, self).__init__(pBbox, pMask, pMaskRoi)

    def _get_mask_head_logit(self, conv_feat):
        if self._head_feat is not None:
            return self._head_feat

        up_stride = int(self.pMask.resolution // self.pMaskRoi.out_size)
        dim_reduced = self.pMask.dim_reduced

        msra_init = mx.init.Xavier(rnd_type="gaussian", factor_type="out", magnitude=2)

        current = conv_feat
        for i in range(4):
            current = X.convrelu(
                current,
                name="mask_fcn_conv{}".format(i),
                filter=dim_reduced,
                kernel=3,
                no_bias=False,
                init=msra_init
            )

        mask_up = current
        for i in range(up_stride // 2):
            weight = X.var(
                name="mask_up{}_weight".format(i),
                init=msra_init,
                lr_mult=1,
                wd_mult=1)
            mask_up = mx.sym.Deconvolution(
                mask_up,
                kernel=(2, 2),
                stride=(2, 2),
                num_filter=dim_reduced,
                no_bias=False,
                weight=weight,
                name="mask_up{}".format(i)
                )
            mask_up = X.relu(
                mask_up,
                name="mask_up{}_relu".format(i))

        mask_up = X.to_fp32(mask_up, name='mask_up_to_fp32')
        self._head_feat = mask_up

        return self._head_feat

