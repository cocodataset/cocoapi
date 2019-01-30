"""
Bbox Target Operator
select foreground and background proposal and encode them as training target.
"""

import mxnet as mx
import numpy as np
import numpy.random as npr
from ast import literal_eval
from .detectron_bbox_utils import bbox_overlaps, bbox_transform_inv


def _sample_proposal(proposals, gt_bboxes, image_rois, fg_fraction, fg_thresh, bg_thresh_hi,
                     bg_thresh_lo, inv_stds, num_reg_class):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    proposal_to_gt_overlaps = bbox_overlaps(
        proposals.astype(np.float32, copy=False),
        gt_bboxes.astype(np.float32, copy=False)
    )
    proposal_assigned_gt_index = proposal_to_gt_overlaps.argmax(axis=1)
    proposal_assigned_class = gt_bboxes[:, 4][proposal_assigned_gt_index]
    proposal_max_overlap_w_gt = proposal_to_gt_overlaps.max(axis=1)

    rois_per_image = image_rois
    fg_rois_per_image = int(np.round(fg_fraction * rois_per_image))

    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = np.where(proposal_max_overlap_w_gt >= fg_thresh)[0]
    # Guard against the case when an image has fewer than fg_rois_per_image
    # foreground RoIs
    fg_rois_per_this_image = np.minimum(fg_rois_per_image, fg_inds.size)
    # Sample foreground regions without replacement
    if fg_inds.size > 0:
        fg_inds = npr.choice(
            fg_inds, size=fg_rois_per_this_image, replace=False
        )

    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where(
        (proposal_max_overlap_w_gt < bg_thresh_hi) &
        (proposal_max_overlap_w_gt >= bg_thresh_lo)
    )[0]
    # Compute number of background RoIs to take from this image (guarding
    # against there being fewer than desired)
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = np.minimum(bg_rois_per_this_image, bg_inds.size)
    # Sample background regions without replacement
    if bg_inds.size > 0:
        bg_inds = npr.choice(
            bg_inds, size=bg_rois_per_this_image, replace=False
        )

    # The indices that we're selecting (both fg and bg)
    keep_inds = np.append(fg_inds, bg_inds)
    # Label is the class each RoI has max overlap with
    sampled_labels = proposal_assigned_class[keep_inds]
    sampled_labels[fg_rois_per_this_image:] = 0  # Label bg RoIs with class 0
    sampled_proposals = proposals[keep_inds]
    sampled_gt_bboxes = gt_bboxes[proposal_assigned_gt_index[keep_inds]]

    bbox_targets = bbox_transform_inv(sampled_proposals, sampled_gt_bboxes, inv_stds)
    bbox_class = sampled_labels[:, None]
    if num_reg_class == 2:
        bbox_class = np.array(bbox_class > 0, dtype=bbox_targets.dtype)
    bbox_targets_with_class = np.concatenate([bbox_class, bbox_targets], axis=1)
    bbox_targets, bbox_weights = _expand_bbox_targets(bbox_targets_with_class, num_reg_class)

    return sampled_proposals, sampled_labels, bbox_targets, bbox_weights


def _expand_bbox_targets(bbox_target_data, num_bbox_reg_classes):
    """Bounding-box regression targets are stored in a compact form in the
    roidb.
    This function expands those targets into the 4-of-4*K representation used
    by the network (i.e. only one class has non-zero targets). The loss weights
    are similarly expanded.
    Returns:
        bbox_target_data (ndarray): N x 4K blob of regression targets
        bbox_inside_weights (ndarray): N x 4K blob of loss weights
    """
    clss = bbox_target_data[:, 0]
    bbox_targets = np.zeros((clss.size, 4 * num_bbox_reg_classes))
    bbox_weights = np.zeros(bbox_targets.shape)
    inds = np.where(clss > 0)[0]
    for ind in inds:
        cls = int(clss[ind])
        start = 4 * cls
        end = start + 4
        bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
        bbox_weights[ind, start:end] = (1.0, 1.0, 1.0, 1.0)
    return bbox_targets, bbox_weights


class BboxTargetOperator(mx.operator.CustomOp):
    def __init__(self, num_classes, add_gt_to_proposal, image_rois, fg_fraction,
                 fg_thresh, bg_thresh_hi, bg_thresh_lo, bbox_target_std):
        super(BboxTargetOperator, self).__init__()
        self._num_classes = num_classes
        self._add_gt_to_proposal = add_gt_to_proposal
        self._image_rois = image_rois
        self._fg_fraction = fg_fraction
        self._fg_thresh = fg_thresh
        self._bg_thresh_hi = bg_thresh_hi
        self._bg_thresh_lo = bg_thresh_lo
        self._bbox_target_std = bbox_target_std

    def forward(self, is_train, req, in_data, out_data, aux):
        proposals = in_data[0].asnumpy()  # N x K x 4
        gt_bboxes = in_data[1].asnumpy()  # N x M x 5

        batch_image = proposals.shape[0]
        image_rois = self._image_rois
        fg_fraction = self._fg_fraction
        fg_thresh = self._fg_thresh
        bg_thresh_hi = self._bg_thresh_hi
        bg_thresh_lo = self._bg_thresh_lo
        inv_stds = list(1.0 / std for std in self._bbox_target_std)
        num_reg_class = self._num_classes

        keep_proposals = []
        keep_gt_bboxes = []

        # clean up gt_bbox
        for im_gt_bbox in gt_bboxes:
            valid = np.where(im_gt_bbox[:, 4] != -1)[0]  # class == -1 indicates padding
            keep_gt_bboxes.append(im_gt_bbox[valid])

        # clean up proposal
        for im_proposal in proposals:
            valid = np.where(im_proposal[:, -1] != 0)[0]  # y2 == 0 indicates padding
            keep_proposals.append(im_proposal[valid])

        if self._add_gt_to_proposal:
            for i in range(batch_image):
                im_proposal, im_gt_bbox = keep_proposals[i], keep_gt_bboxes[i]
                keep_proposals[i] = np.append(im_proposal, im_gt_bbox[:, :4], axis=0)

        sampled_proposal, bbox_class, bbox_target, bbox_target_weight = [], [], [], []
        for i in range(batch_image):
            output = _sample_proposal(
                keep_proposals[i],
                keep_gt_bboxes[i],
                image_rois,
                fg_fraction,
                fg_thresh,
                bg_thresh_hi,
                bg_thresh_lo,
                inv_stds,
                num_reg_class
            )
            sampled_proposal_i, bbox_class_i, bbox_target_i, bbox_target_weight_i = output
            sampled_proposal.append(sampled_proposal_i)
            bbox_class.append(bbox_class_i)
            bbox_target.append(bbox_target_i)
            bbox_target_weight.append(bbox_target_weight_i)

        sampled_proposal = np.array(sampled_proposal, dtype=np.float32)
        bbox_class = np.array(bbox_class, dtype=np.float32)
        bbox_target = np.array(bbox_target, dtype=np.float32)
        bbox_target_weight = np.array(bbox_target_weight, dtype=np.float32)

        for i, val in enumerate([sampled_proposal, bbox_class, bbox_target, bbox_target_weight]):
            self.assign(out_data[i], req[i], val)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], 0)
        self.assign(in_grad[1], req[1], 0)


@mx.operator.register('bbox_target')
class BboxTargetProp(mx.operator.CustomOpProp):
    def __init__(self, num_class, add_gt_to_proposal, image_rois, fg_fraction, fg_thresh,
                 bg_thresh_hi, bg_thresh_lo, bbox_target_std):
        super(BboxTargetProp, self).__init__(need_top_grad=False)
        self._num_class = int(num_class)
        self._add_gt_to_proposal = literal_eval(add_gt_to_proposal)
        self._image_rois = int(image_rois)
        self._fg_fraction = float(fg_fraction)
        self._fg_thresh = float(fg_thresh)
        self._bg_thresh_hi = float(bg_thresh_hi)
        self._bg_thresh_lo = float(bg_thresh_lo)
        self._bbox_target_std = literal_eval(bbox_target_std)

    def list_arguments(self):
        return ['proposal', 'gt_bbox']

    def list_outputs(self):
        return ['sampled_proposal', 'bbox_cls', 'bbox_target', 'bbox_target_weight']

    def infer_shape(self, in_shape):
        rpn_rois_shape = in_shape[0]
        gt_boxes_shape = in_shape[1]

        batch_image = rpn_rois_shape[0]

        sampled_proposal_shape = (batch_image, self._image_rois, 4)
        bbox_cls_shape = (batch_image, self._image_rois, )
        bbox_target_shape = (batch_image, self._image_rois, self._num_class * 4)
        bbox_weight_shape = (batch_image, self._image_rois, self._num_class * 4)

        return [rpn_rois_shape, gt_boxes_shape], \
               [sampled_proposal_shape, bbox_cls_shape, bbox_target_shape, bbox_weight_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return BboxTargetOperator(
            self._num_class,
            self._add_gt_to_proposal,
            self._image_rois,
            self._fg_fraction,
            self._fg_thresh,
            self._bg_thresh_hi,
            self._bg_thresh_lo,
            self._bbox_target_std
        )

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []
