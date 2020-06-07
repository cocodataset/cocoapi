import logging
from collections import defaultdict
from copy import deepcopy

import pycocotools.mask as mask_utils

from .lvis import LVIS


class LVISResults(LVIS):
    def __init__(self, lvis_gt, results, max_dets=300):
        """Constructor for LVIS results.
        Args:
            lvis_gt (LVIS class instance, or str containing path of
            annotation file)
            results (str containing path of result file or a list of dicts)
            max_dets (int):  max number of detections per image. The official
            value of max_dets for LVIS is 300.
        """
        if isinstance(lvis_gt, LVIS):
            self.dataset = deepcopy(lvis_gt.dataset)
        elif isinstance(lvis_gt, str):
            self.dataset = self._load_json(lvis_gt)
        else:
            raise TypeError('Unsupported type {} of lvis_gt.'.format(lvis_gt))

        self.logger = logging.getLogger(__name__)
        self.logger.info('Loading and preparing results.')

        if isinstance(results, str):
            result_anns = self._load_json(results)
        else:
            # this path way is provided to avoid saving and loading result
            # during training.
            self.logger.warn(
                'Assuming user provided the results in correct format.')
            result_anns = results

        assert isinstance(result_anns, list), 'results is not a list.'

        if max_dets >= 0:
            result_anns = self.limit_dets_per_image(result_anns, max_dets)

        if 'bbox' in result_anns[0]:
            for id, ann in enumerate(result_anns):
                x1, y1, w, h = ann['bbox']
                x2 = x1 + w
                y2 = y1 + h

                if 'segmentation' not in ann:
                    ann['segmentation'] = [[x1, y1, x1, y2, x2, y2, x2, y1]]

                ann['area'] = w * h
                ann['id'] = id + 1

        elif 'segmentation' in result_anns[0]:
            for id, ann in enumerate(result_anns):
                # Only support compressed RLE format as segmentation results
                ann['area'] = mask_utils.area(ann['segmentation'])

                if 'bbox' not in ann:
                    ann['bbox'] = mask_utils.toBbox(ann['segmentation'])

                ann['id'] = id + 1

        self.dataset['annotations'] = result_anns
        self._create_index()

        img_ids_in_result = [ann['image_id'] for ann in result_anns]

        assert set(img_ids_in_result) == (
            set(img_ids_in_result) & set(self.get_img_ids())
        ), 'Results do not correspond to current LVIS set.'

    def limit_dets_per_image(self, anns, max_dets):
        img_ann = defaultdict(list)
        for ann in anns:
            img_ann[ann['image_id']].append(ann)

        for img_id, _anns in img_ann.items():
            if len(_anns) <= max_dets:
                continue
            _anns = sorted(_anns, key=lambda ann: ann['score'], reverse=True)
            img_ann[img_id] = _anns[:max_dets]

        return [ann for anns in img_ann.values() for ann in anns]

    def get_top_results(self, img_id, score_thrs):
        ann_ids = self.get_ann_ids(img_ids=[img_id])
        anns = self.load_anns(ann_ids)
        return list(filter(lambda ann: ann['score'] > score_thrs, anns))
