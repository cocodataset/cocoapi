import numpy as np
import cv2

from pycocotools import mask as mask_util


def expand_boxes(boxes, scale):
    """Expand an array of boxes by a given scale."""
    w_half = (boxes[:, 2] - boxes[:, 0]) * .5
    h_half = (boxes[:, 3] - boxes[:, 1]) * .5
    x_c = (boxes[:, 2] + boxes[:, 0]) * .5
    y_c = (boxes[:, 3] + boxes[:, 1]) * .5

    w_half *= scale
    h_half *= scale

    boxes_exp = np.zeros(boxes.shape)
    boxes_exp[:, 0] = x_c - w_half
    boxes_exp[:, 2] = x_c + w_half
    boxes_exp[:, 1] = y_c - h_half
    boxes_exp[:, 3] = y_c + h_half

    return boxes_exp


def segm_results(bbox_xyxy, cls, masks, im_h, im_w):
    # Modify from Detectron
    # To work around an issue with cv2.resize (it seems to automatically pad
    # with repeated border values), we manually zero-pad the masks by 1 pixel
    # prior to resizing back to the original image resolution. This prevents
    # "top hat" artifacts. We therefore need to expand the reference boxes by an
    # appropriate factor.
    segms = []
    M = masks.shape[-1]
    scale = (M + 2.0) / M
    ref_boxes = expand_boxes(bbox_xyxy, scale)
    ref_boxes = ref_boxes.astype(np.int32)
    padded_mask = np.zeros((M + 2, M + 2), dtype=np.float32)

    for ref_box_i, mask_i, cls_i in zip(ref_boxes, masks, cls):
        padded_mask[1:-1, 1:-1] = mask_i[cls_i, :, :]

        w = ref_box_i[2] - ref_box_i[0] + 1
        h = ref_box_i[3] - ref_box_i[1] + 1
        w = np.maximum(w, 1)
        h = np.maximum(h, 1)

        mask = cv2.resize(padded_mask, (w, h))
        mask = np.array(mask > 0.5, dtype=np.uint8)
        im_mask = np.zeros((im_h, im_w), dtype=np.uint8)

        x_0 = max(ref_box_i[0], 0)
        x_1 = min(ref_box_i[2] + 1, im_w)
        y_0 = max(ref_box_i[1], 0)
        y_1 = min(ref_box_i[3] + 1, im_h)

        im_mask[y_0:y_1, x_0:x_1] = mask[
            (y_0 - ref_box_i[1]):(y_1 - ref_box_i[1]),
            (x_0 - ref_box_i[0]):(x_1 - ref_box_i[0])
        ]

        # Get RLE encoding used by the COCO evaluation API
        rle = mask_util.encode(
            np.array(im_mask[:, :, np.newaxis], order='F')
        )[0]
        segms.append(rle)
    segms = np.array(segms)
    return segms