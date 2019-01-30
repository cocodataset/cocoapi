import numpy as np
from pycocotools import mask as mask_util

from models.maskrcnn.utils import segm_results

def process_output(all_outputs, roidb):
    for output_record in all_outputs:
        rec_id = int(output_record['rec_id'])
        bbox_xyxy = output_record['bbox_xyxy']
        cls_score = output_record['cls_score']
        cls = output_record['cls']
        mask = output_record['mask']

        im_h = roidb[rec_id]["h"]
        im_w = roidb[rec_id]["w"]
        segm = segm_results(bbox_xyxy, cls, mask, im_h, im_w)
        output_record['segm'] = segm
        del output_record['mask']
    return all_outputs
