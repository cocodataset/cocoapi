from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
import copy

from core.detection_input import DetectionAugmentation, AnchorTarget2D


class Resize2DImageBboxMask(DetectionAugmentation):
    """
    input: image, ndarray(h, w, rgb)
           gt_bbox, ndarry(n, 5)
           gt_poly, [[(p, 2)]]
    output: image, ndarray(h', w', rgb)
            im_info, tuple(h', w', scale)
            gt_bbox, ndarray(n, 5)
            gt_poly, [[ndarray, ndarray, ...]]
    """

    def __init__(self, pResize):
        super(Resize2DImageBboxMask, self).__init__()
        self.p = pResize  # type: ResizeParam

    def apply(self, input_record):
        p = self.p

        image = input_record["image"]
        gt_bbox = input_record["gt_bbox"]
        gt_poly = input_record["gt_poly"]

        short = min(image.shape[:2])
        long = max(image.shape[:2])
        scale = min(p.short / short, p.long / long)

        input_record["image"] = cv2.resize(image, None, None, scale, scale,
                                           interpolation=cv2.INTER_LINEAR)
        # make sure gt boxes do not overflow
        gt_bbox[:, :4] = gt_bbox[:, :4] * scale
        if image.shape[0] < image.shape[1]:
            gt_bbox[:, [0, 2]] = np.clip(gt_bbox[:, [0, 2]], 0, p.long)
            gt_bbox[:, [1, 3]] = np.clip(gt_bbox[:, [1, 3]], 0, p.short)
        else:
            gt_bbox[:, [0, 2]] = np.clip(gt_bbox[:, [0, 2]], 0, p.short)
            gt_bbox[:, [1, 3]] = np.clip(gt_bbox[:, [1, 3]], 0, p.long)
        input_record["gt_bbox"] = gt_bbox

        # exactly as opencv
        h, w = image.shape[:2]
        input_record["im_info"] = (round(h * scale), round(w * scale), scale)

        # resize poly
        for i, segms in enumerate(gt_poly):
            input_record["gt_poly"][i] = [segm_j * scale for segm_j in segms]


class Flip2DImageBboxMask(DetectionAugmentation):
    """
    input: image, ndarray(h, w, rgb)
           gt_bbox, ndarry(n, 5)
           gt_poly, [[ndarray, ndarray, ...]]
    output: image, ndarray(h, w, rgb)
            gt_bbox, ndarray(n, 5)
            gt_poly, [[ndarray, ndarray, ...]]
    """

    def __init__(self):
        super(Flip2DImageBboxMask, self).__init__()

    def apply(self, input_record):
        def _flip_poly(poly, width):
            flipped_poly = poly.copy()
            flipped_poly[0::2] = width - poly[0::2] - 1
            return flipped_poly

        if input_record["flipped"]:
            image = input_record["image"]
            gt_bbox = input_record["gt_bbox"]
            gt_poly = input_record["gt_poly"]

            input_record["image"] = image[:, ::-1]
            flipped_bbox = gt_bbox.copy()
            h, w = image.shape[:2]
            flipped_bbox[:, 0] = (w - 1) - gt_bbox[:, 2]
            flipped_bbox[:, 2] = (w - 1) - gt_bbox[:, 0]
            input_record["gt_bbox"] = flipped_bbox

            # flip poly
            for i, segms in enumerate(gt_poly):
                input_record["gt_poly"][i] = [_flip_poly(segm_j, w) for segm_j in segms]


class Pad2DImageBboxMask(DetectionAugmentation):
    """
    input: image, ndarray(h, w, rgb)
           gt_bbox, ndarry(n, 5)
           gt_poly, [[ndarray, ndarray, ...]]
    output: image, ndarray(h, w, rgb)
            gt_bbox, ndarray(max_num_gt, 5)
            gt_poly, [[ndarray, ndarray, ...]]
    """

    def __init__(self, pPad):
        super(Pad2DImageBboxMask, self).__init__()
        self.p = pPad  # type: PadParam

    def apply(self, input_record):
        p = self.p

        image = input_record["image"]
        gt_bbox = input_record["gt_bbox"]
        gt_poly = input_record["gt_poly"]

        h, w = image.shape[:2]
        shape = (p.long, p.short, 3) if h >= w \
            else (p.short, p.long, 3)

        padded_image = np.zeros(shape, dtype=np.float32)
        padded_image[:h, :w] = image
        padded_gt_bbox = np.full(shape=(p.max_num_gt, 5), fill_value=-1, dtype=np.float32)
        padded_gt_bbox[:len(gt_bbox)] = gt_bbox

        padded_gt_poly = np.full(shape=(p.max_num_gt, p.max_len_gt_poly), fill_value=-1, dtype=np.float32)
        padded_gt_poly[:len(gt_bbox)] = gt_poly

        input_record["image"] = padded_image
        input_record["gt_bbox"] = padded_gt_bbox
        input_record["gt_poly"] = padded_gt_poly


class PreprocessGtPoly(DetectionAugmentation):
    # TODO: remove this function and set gt_poly in cache to ndarray
    """
    input: gt_poly
    output: gt_poly
    """

    def __init__(self):
        super(PreprocessGtPoly, self).__init__()

    def apply(self, input_record):
        ins_poly = input_record["gt_poly"]
        gt_poly = [None] * len(ins_poly)
        for i, ins_poly_i in enumerate(ins_poly):
            segms = [None] * len(ins_poly_i)
            for j, segm_j in enumerate(ins_poly_i):
                segms[j] = np.array(segm_j, dtype=np.float32)
            gt_poly[i] = segms
        input_record["gt_poly"] = gt_poly


class EncodeGtPoly(DetectionAugmentation):
    """
    input: gt_class, gt_poly
    output: gt_poly
    """

    def __init__(self, pPad):
        super(EncodeGtPoly, self).__init__()
        self.p = pPad

    def apply(self, input_record):
        gt_class = input_record["gt_class"]
        gt_poly = input_record["gt_poly"] # [[ndarray, ndarray, ...]]

        num_instance = len(gt_class)
        encoded_gt_poly = np.full((num_instance, self.p.max_len_gt_poly), -1, dtype=np.float32)

        for i, (class_id, segms) in enumerate(zip(gt_class, gt_poly)):
            # encoded_gt_poly_i: [class_id, num_segms, len_segm1, len_segm2, segm1, segm2]
            encoded_gt_poly[i][0] = class_id
            num_segms = len(segms)
            encoded_gt_poly[i][1] = num_segms
            segms_len = [len(segm_j) for segm_j in segms]
            encoded = np.hstack([np.array(segms_len), np.hstack(segms)])
            encoded_gt_poly[i][2:2+len(encoded)] = encoded

        input_record["gt_poly"] = encoded_gt_poly


if __name__ == "__main__":
    import six.moves.cPickle as pkl
    import time

    import pycocotools.mask as mask_util

    from rcnn.core.detection_input import ReadRoiRecord, \
        ConvertImageFromHwcToChw, Flip2DImageBbox, Pad2DImageBbox, \
        RenameRecord, AnchorTarget2D, AnchorLoader

    from models.maskrcnn.input import PreprocessGtPoly, EncodeGtPoly, \
        Resize2DImageBboxMask, Flip2DImageBboxMask, Pad2DImageBboxMask

    def vis_mask(img, mask, col, alpha=0.4, show_border=True, border_thick=1):
        """Visualizes a single binary mask."""

        img = img.astype(np.float32)
        idx = np.nonzero(mask)

        img[idx[0], idx[1], :] *= 1.0 - alpha
        img[idx[0], idx[1], :] += alpha * col

        if show_border:
            _, contours, _ = cv2.findContours(
                mask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(img, contours, -1, _WHITE, border_thick, cv2.LINE_AA)

        return img.astype(np.uint8)

    class ResizeParam:
        short = 800
        long = 1200


    class PadParam:
        short = 800
        long = 1200
        max_num_gt = 100
        max_len_gt_poly = 2500


    class AnchorTarget2DParam:
        class generate:
            short = 800 // 16
            long = 1200 // 16
            stride = 16
            scales = (2, 4, 8, 16, 32)
            aspects = (0.5, 1.0, 2.0)

        class assign:
            allowed_border = 0
            pos_thr = 0.7
            neg_thr = 0.3
            min_pos_thr = 0.0

        class sample:
            image_anchor = 256
            pos_fraction = 0.5

    class RenameParam:
        mapping = dict(image="data")

    transform = [
        ReadRoiRecord(None),
        PreprocessGtPoly(),
        Resize2DImageBboxMask(ResizeParam),
        Flip2DImageBboxMask(),
        EncodeGtPoly(PadParam),
        Pad2DImageBboxMask(PadParam),
        ConvertImageFromHwcToChw(),
        AnchorTarget2D(AnchorTarget2DParam),
        RenameRecord(RenameParam.mapping)
    ]

    DEBUG = True

    with open("data/cache/coco_valminusminival2014.roidb", "rb") as fin:
        roidb = pkl.load(fin)
        roidb = [rec for rec in roidb if rec["gt_bbox"].shape[0] > 0]
        roidb = [roidb[i] for i in np.random.choice(len(roidb), 20, replace=False)]

        print(roidb[0])
        flipped_roidb = []
        for rec in roidb:
            new_rec = rec.copy()
            new_rec["flipped"] = True
            flipped_roidb.append(new_rec)
        roidb = roidb + flipped_roidb

        loader = AnchorLoader(roidb=roidb,
                              transform=transform,
                              data_name=["data", "im_info", "gt_bbox", "gt_poly"],
                              label_name=["rpn_cls_label", "rpn_reg_target", "rpn_reg_weight"],
                              batch_size=2,
                              shuffle=False,
                              num_thread=4,
                              kv=None)


        tic = time.time()
        while True:
            try:
                data_batch = loader.next()
                if DEBUG:
                    print(data_batch.provide_data)
                    print(data_batch.provide_label)
                    print(data_batch.data[0].shape)
                    print(data_batch.label[1].shape)
                    print(data_batch.label[2].shape)
                    data = data_batch.data[0]
                    gt_bbox = data_batch.data[2]
                    gt_poly = data_batch.data[3]
                    for i, (im, bbox, poly) in enumerate(zip(data, gt_bbox, gt_poly)):
                        im = im.transpose((1, 2, 0))[:, :, ::-1].asnumpy()
                        im = np.uint8(im)
                        valid_instance = np.where(bbox[:, -1] != -1)[0]
                        bbox = bbox[valid_instance].asnumpy()
                        poly = poly[valid_instance].asnumpy()
                        for j, (bbox_j, poly_j) in enumerate(zip(bbox, poly)):
                            x1, y1, x2, y2 = bbox_j[:4].astype(int)
                            cv2.rectangle(im, (x1, y1), (x2, y2), (255, 0, 0), 2)
                            class_index = poly_j[0]
                            assert int(class_index) == int(bbox_j[-1])
                            num_segms = poly_j[1]
                            len_segms = poly_j[2:2+int(num_segms)]
                            cur_start = 2 + int(num_segms)
                            segms = []
                            for len_segm in len_segms:
                                segm = poly_j[cur_start:cur_start+int(len_segm)]
                                segm = segm.tolist()
                                segms.append(segm)
                                cur_start = cur_start + int(len_segm)
                            rle = mask_util.frPyObjects(segms, im.shape[0], im.shape[1])
                            mask = mask_util.decode(rle)
                            mask = np.sum(mask, axis=2)
                            mask = np.array(mask > 0, dtype=np.float32)
                            im = vis_mask(im, mask, np.array([18, 127, 15]), alpha=0.4, show_border=False, border_thick=1)
                        cv2.imshow("im", im)
                        cv2.waitKey(0)
            except StopIteration:
                toc = time.time()
                print("{} samples/s".format(len(roidb) / (toc - tic)))
                break