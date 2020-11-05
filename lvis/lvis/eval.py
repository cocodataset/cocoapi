import datetime
import logging
from collections import OrderedDict, defaultdict

import numpy as np

import pycocotools.mask as mask_utils

from .lvis import LVIS
from .results import LVISResults


class LVISEval:
    def __init__(self, lvis_gt, lvis_dt, iou_type='segm'):
        """Constructor for LVISEval.
        Args:
            lvis_gt (LVIS class instance, or str containing path of annotation
                file)
            lvis_dt (LVISResult class instance, or str containing path of
                result file, or list of dict)
            iou_type (str): segm or bbox evaluation
        """
        self.logger = logging.getLogger(__name__)

        if iou_type not in ['bbox', 'segm']:
            raise ValueError('iou_type: {} is not supported.'.format(iou_type))

        if isinstance(lvis_gt, LVIS):
            self.lvis_gt = lvis_gt
        elif isinstance(lvis_gt, str):
            self.lvis_gt = LVIS(lvis_gt)
        else:
            raise TypeError('Unsupported type {} of lvis_gt.'.format(lvis_gt))

        if isinstance(lvis_dt, LVISResults):
            self.lvis_dt = lvis_dt
        elif isinstance(lvis_dt, (str, list)):
            self.lvis_dt = LVISResults(self.lvis_gt, lvis_dt)
        else:
            raise TypeError('Unsupported type {} of lvis_dt.'.format(lvis_dt))

        # per-image per-category evaluation results
        self.eval_imgs = defaultdict(list)
        self.eval = {}  # accumulated evaluation results
        self._gts = defaultdict(list)  # gt for evaluation
        self._dts = defaultdict(list)  # dt for evaluation
        self.params = Params(iou_type=iou_type)  # parameters
        self.results = OrderedDict()
        self.ious = {}  # ious between all gts and dts

        self.params.img_ids = sorted(self.lvis_gt.get_img_ids())
        self.params.cat_ids = sorted(self.lvis_gt.get_cat_ids())

    def _to_mask(self, anns, lvis):
        for ann in anns:
            rle = lvis.ann_to_rle(ann)
            ann['segmentation'] = rle

    def _prepare(self):
        """Prepare self._gts and self._dts for evaluation based on params."""

        cat_ids = self.params.cat_ids if self.params.cat_ids else None

        gts = self.lvis_gt.load_anns(
            self.lvis_gt.get_ann_ids(img_ids=self.params.img_ids,
                                     cat_ids=cat_ids))
        dts = self.lvis_dt.load_anns(
            self.lvis_dt.get_ann_ids(img_ids=self.params.img_ids,
                                     cat_ids=cat_ids))
        # convert ground truth to mask if iou_type == 'segm'
        if self.params.iou_type == 'segm':
            self._to_mask(gts, self.lvis_gt)
            self._to_mask(dts, self.lvis_dt)

        # set ignore flag
        for gt in gts:
            if 'ignore' not in gt:
                gt['ignore'] = 0

        for gt in gts:
            self._gts[gt['image_id'], gt['category_id']].append(gt)

        # For federated dataset evaluation we will filter out all dt for an
        # image which belong to categories not present in gt and not present in
        # the negative list for an image. In other words detector is not
        # penalized for categories about which we don't have gt information
        # about their presence or absence in an image.
        img_data = self.lvis_gt.load_imgs(ids=self.params.img_ids)
        # per image map of categories not present in image
        img_nl = {d['id']: d['neg_category_ids'] for d in img_data}
        # per image list of categories present in image
        img_pl = defaultdict(set)
        for ann in gts:
            img_pl[ann['image_id']].add(ann['category_id'])
        # per image map of categoires which have missing gt. For these
        # categories we don't penalize the detector for flase positives.
        self.img_nel = {
            d['id']: d['not_exhaustive_category_ids']
            for d in img_data
        }

        for dt in dts:
            img_id, cat_id = dt['image_id'], dt['category_id']
            if cat_id not in img_nl[img_id] and cat_id not in img_pl[img_id]:
                continue
            self._dts[img_id, cat_id].append(dt)

        self.freq_groups = self._prepare_freq_group()

    def _prepare_freq_group(self):
        freq_groups = [[] for _ in self.params.img_count_lbl]
        cat_data = self.lvis_gt.load_cats(self.params.cat_ids)
        for idx, _cat_data in enumerate(cat_data):
            frequency = _cat_data['frequency']
            freq_groups[self.params.img_count_lbl.index(frequency)].append(idx)
        return freq_groups

    def evaluate(self):
        """
        Run per image evaluation on given images and store results
        (a list of dict) in self.eval_imgs.
        """
        self.logger.info('Running per image evaluation.')
        self.logger.info('Evaluate annotation type *{}*'.format(
            self.params.iou_type))

        self.params.img_ids = list(np.unique(self.params.img_ids))

        if self.params.use_cats:
            cat_ids = self.params.cat_ids
        else:
            cat_ids = [-1]

        self._prepare()

        self.ious = {(img_id, cat_id): self.compute_iou(img_id, cat_id)
                     for img_id in self.params.img_ids for cat_id in cat_ids}

        # loop through images, area range, max detection number
        self.eval_imgs = [
            self.evaluate_img(img_id, cat_id, area_rng) for cat_id in cat_ids
            for area_rng in self.params.area_rng
            for img_id in self.params.img_ids
        ]

    def _get_gt_dt(self, img_id, cat_id):
        """Create gt, dt which are list of anns/dets. If use_cats is true
        only anns/dets corresponding to tuple (img_id, cat_id) will be
        used. Else, all anns/dets in image are used and cat_id is not used.
        """
        if self.params.use_cats:
            gt = self._gts[img_id, cat_id]
            dt = self._dts[img_id, cat_id]
        else:
            gt = [
                _ann for _cat_id in self.params.cat_ids
                for _ann in self._gts[img_id, cat_id]
            ]
            dt = [
                _ann for _cat_id in self.params.cat_ids
                for _ann in self._dts[img_id, cat_id]
            ]
        return gt, dt

    def compute_iou(self, img_id, cat_id):
        gt, dt = self._get_gt_dt(img_id, cat_id)

        if len(gt) == 0 and len(dt) == 0:
            return []

        # Sort detections in decreasing order of score.
        idx = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in idx]

        iscrowd = [int(False)] * len(gt)

        if self.params.iou_type == 'segm':
            ann_type = 'segmentation'
        elif self.params.iou_type == 'bbox':
            ann_type = 'bbox'
        else:
            raise ValueError('Unknown iou_type for iou computation.')
        gt = [g[ann_type] for g in gt]
        dt = [d[ann_type] for d in dt]

        # compute iou between each dt and gt region
        # will return array of shape len(dt), len(gt)
        ious = mask_utils.iou(dt, gt, iscrowd)
        return ious

    def evaluate_img(self, img_id, cat_id, area_rng):
        """Perform evaluation for single category and image."""
        gt, dt = self._get_gt_dt(img_id, cat_id)

        if len(gt) == 0 and len(dt) == 0:
            return None

        # Add another filed _ignore to only consider anns based on area range.
        for g in gt:
            if g['ignore'] or (g['area'] < area_rng[0]
                               or g['area'] > area_rng[1]):
                g['_ignore'] = 1
            else:
                g['_ignore'] = 0

        # Sort gt ignore last
        gt_idx = np.argsort([g['_ignore'] for g in gt], kind='mergesort')
        gt = [gt[i] for i in gt_idx]

        # Sort dt highest score first
        dt_idx = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in dt_idx]

        # load computed ious
        ious = (self.ious[img_id, cat_id][:, gt_idx]
                if len(self.ious[img_id, cat_id]) > 0 else self.ious[img_id,
                                                                     cat_id])

        num_thrs = len(self.params.iou_thrs)
        num_gt = len(gt)
        num_dt = len(dt)

        # Array to store the "id" of the matched dt/gt
        gt_m = np.zeros((num_thrs, num_gt))
        dt_m = np.zeros((num_thrs, num_dt))

        gt_ig = np.array([g['_ignore'] for g in gt])
        dt_ig = np.zeros((num_thrs, num_dt))

        for iou_thr_idx, iou_thr in enumerate(self.params.iou_thrs):
            if len(ious) == 0:
                break

            for dt_idx, _dt in enumerate(dt):
                iou = min([iou_thr, 1 - 1e-10])
                # information about best match so far (m=-1 -> unmatched)
                # store the gt_idx which matched for _dt
                m = -1
                for gt_idx, _ in enumerate(gt):
                    # if this gt already matched continue
                    if gt_m[iou_thr_idx, gt_idx] > 0:
                        continue
                    # if _dt matched to reg gt, and on ignore gt, stop
                    if m > -1 and gt_ig[m] == 0 and gt_ig[gt_idx] == 1:
                        break
                    # continue to next gt unless better match made
                    if ious[dt_idx, gt_idx] < iou:
                        continue
                    # if match successful and best so far, store appropriately
                    iou = ious[dt_idx, gt_idx]
                    m = gt_idx

                # No match found for _dt, go to next _dt
                if m == -1:
                    continue

                # if gt to ignore for some reason update dt_ig.
                # Should not be used in evaluation.
                dt_ig[iou_thr_idx, dt_idx] = gt_ig[m]
                # _dt match found, update gt_m, and dt_m with "id"
                dt_m[iou_thr_idx, dt_idx] = gt[m]['id']
                gt_m[iou_thr_idx, m] = _dt['id']

        # For LVIS we will ignore any unmatched detection if that category was
        # not exhaustively annotated in gt.
        dt_ig_mask = [
            d['area'] < area_rng[0] or d['area'] > area_rng[1]
            or d['category_id'] in self.img_nel[d['image_id']] for d in dt
        ]
        dt_ig_mask = np.array(dt_ig_mask).reshape((1, num_dt))  # 1 X num_dt
        dt_ig_mask = np.repeat(dt_ig_mask, num_thrs, 0)  # num_thrs X num_dt
        # Based on dt_ig_mask ignore any unmatched detection by updating dt_ig
        dt_ig = np.logical_or(dt_ig, np.logical_and(dt_m == 0, dt_ig_mask))
        # store results for given image and category
        return {
            'image_id': img_id,
            'category_id': cat_id,
            'area_rng': area_rng,
            'dt_ids': [d['id'] for d in dt],
            'gt_ids': [g['id'] for g in gt],
            'dt_matches': dt_m,
            'gt_matches': gt_m,
            'dt_scores': [d['score'] for d in dt],
            'gt_ignore': gt_ig,
            'dt_ignore': dt_ig,
        }

    def accumulate(self):
        """Accumulate per image evaluation results and store the result in
        self.eval.
        """
        self.logger.info('Accumulating evaluation results.')

        if not self.eval_imgs:
            self.logger.warn('Please run evaluate first.')

        if self.params.use_cats:
            cat_ids = self.params.cat_ids
        else:
            cat_ids = [-1]

        num_thrs = len(self.params.iou_thrs)
        num_recalls = len(self.params.rec_thrs)
        num_cats = len(cat_ids)
        num_area_rngs = len(self.params.area_rng)
        num_imgs = len(self.params.img_ids)

        # -1 for absent categories
        precision = -np.ones((num_thrs, num_recalls, num_cats, num_area_rngs))
        recall = -np.ones((num_thrs, num_cats, num_area_rngs))

        # Initialize dt_pointers
        dt_pointers = {}
        for cat_idx in range(num_cats):
            dt_pointers[cat_idx] = {}
            for area_idx in range(num_area_rngs):
                dt_pointers[cat_idx][area_idx] = {}

        # Per category evaluation
        for cat_idx in range(num_cats):
            Nk = cat_idx * num_area_rngs * num_imgs
            for area_idx in range(num_area_rngs):
                Na = area_idx * num_imgs
                E = [
                    self.eval_imgs[Nk + Na + img_idx]
                    for img_idx in range(num_imgs)
                ]
                # Remove elements which are None
                E = [e for e in E if e is not None]
                if len(E) == 0:
                    continue

                # Append all scores: shape (N,)
                dt_scores = np.concatenate([e['dt_scores'] for e in E], axis=0)
                dt_ids = np.concatenate([e['dt_ids'] for e in E], axis=0)

                dt_idx = np.argsort(-dt_scores, kind='mergesort')
                dt_scores = dt_scores[dt_idx]
                dt_ids = dt_ids[dt_idx]

                dt_m = np.concatenate([e['dt_matches'] for e in E],
                                      axis=1)[:, dt_idx]
                dt_ig = np.concatenate([e['dt_ignore'] for e in E],
                                       axis=1)[:, dt_idx]

                gt_ig = np.concatenate([e['gt_ignore'] for e in E])
                # num gt anns to consider
                num_gt = np.count_nonzero(gt_ig == 0)

                if num_gt == 0:
                    continue

                tps = np.logical_and(dt_m, np.logical_not(dt_ig))
                fps = np.logical_and(np.logical_not(dt_m),
                                     np.logical_not(dt_ig))

                tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float)
                fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float)

                dt_pointers[cat_idx][area_idx] = {
                    'dt_ids': dt_ids,
                    'tps': tps,
                    'fps': fps,
                }

                for iou_thr_idx, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                    tp = np.array(tp)
                    fp = np.array(fp)
                    num_tp = len(tp)
                    rc = tp / num_gt
                    if num_tp:
                        recall[iou_thr_idx, cat_idx, area_idx] = rc[-1]
                    else:
                        recall[iou_thr_idx, cat_idx, area_idx] = 0

                    # np.spacing(1) ~= eps
                    pr = tp / (fp + tp + np.spacing(1))
                    pr = pr.tolist()

                    # Replace each precision value with the maximum precision
                    # value to the right of that recall level. This ensures
                    # that the  calculated AP value will be less suspectable
                    # to small variations in the ranking.
                    for i in range(num_tp - 1, 0, -1):
                        if pr[i] > pr[i - 1]:
                            pr[i - 1] = pr[i]

                    rec_thrs_insert_idx = np.searchsorted(rc,
                                                          self.params.rec_thrs,
                                                          side='left')

                    pr_at_recall = [0.0] * num_recalls

                    try:
                        for _idx, pr_idx in enumerate(rec_thrs_insert_idx):
                            pr_at_recall[_idx] = pr[pr_idx]
                    except:  # noqa: E722
                        pass
                    precision[iou_thr_idx, :, cat_idx,
                              area_idx] = np.array(pr_at_recall)

        self.eval = {
            'params': self.params,
            'counts': [num_thrs, num_recalls, num_cats, num_area_rngs],
            'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'precision': precision,
            'recall': recall,
            'dt_pointers': dt_pointers,
        }

    def _summarize(self,
                   summary_type,
                   iou_thr=None,
                   area_rng='all',
                   freq_group_idx=None):
        aidx = [
            idx for idx, _area_rng in enumerate(self.params.area_rng_lbl)
            if _area_rng == area_rng
        ]

        if summary_type == 'ap':
            s = self.eval['precision']
            if iou_thr is not None:
                tidx = np.where(iou_thr == self.params.iou_thrs)[0]
                s = s[tidx]
            if freq_group_idx is not None:
                s = s[:, :, self.freq_groups[freq_group_idx], aidx]
            else:
                s = s[:, :, :, aidx]
        else:
            s = self.eval['recall']
            if iou_thr is not None:
                tidx = np.where(iou_thr == self.params.iou_thrs)[0]
                s = s[tidx]
            s = s[:, :, aidx]

        if len(s[s > -1]) == 0:
            mean_s = -1
        else:
            mean_s = np.mean(s[s > -1])
        return mean_s

    def summarize(self):
        """Compute and display summary metrics for evaluation results."""
        if not self.eval:
            raise RuntimeError('Please run accumulate() first.')

        max_dets = self.params.max_dets

        self.results['AP'] = self._summarize('ap')
        self.results['AP50'] = self._summarize('ap', iou_thr=0.50)
        self.results['AP75'] = self._summarize('ap', iou_thr=0.75)
        self.results['APs'] = self._summarize('ap', area_rng='small')
        self.results['APm'] = self._summarize('ap', area_rng='medium')
        self.results['APl'] = self._summarize('ap', area_rng='large')
        self.results['APr'] = self._summarize('ap', freq_group_idx=0)
        self.results['APc'] = self._summarize('ap', freq_group_idx=1)
        self.results['APf'] = self._summarize('ap', freq_group_idx=2)

        key = 'AR@{}'.format(max_dets)
        self.results[key] = self._summarize('ar')

        for area_rng in ['small', 'medium', 'large']:
            key = 'AR{}@{}'.format(area_rng[0], max_dets)
            self.results[key] = self._summarize('ar', area_rng=area_rng)

    def run(self):
        """Wrapper function which calculates the results."""
        self.evaluate()
        self.accumulate()
        self.summarize()

    def print_results(self):
        template = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} catIds={:>3s}] = {:0.3f}'  # noqa: E501

        for key, value in self.results.items():
            max_dets = self.params.max_dets
            if 'AP' in key:
                title = 'Average Precision'
                _type = '(AP)'
            else:
                title = 'Average Recall'
                _type = '(AR)'

            if len(key) > 2 and key[2].isdigit():
                iou_thr = (float(key[2:]) / 100)
                iou = '{:0.2f}'.format(iou_thr)
            else:
                iou = '{:0.2f}:{:0.2f}'.format(self.params.iou_thrs[0],
                                               self.params.iou_thrs[-1])

            if len(key) > 2 and key[2] in ['r', 'c', 'f']:
                cat_group_name = key[2]
            else:
                cat_group_name = 'all'

            if len(key) > 2 and key[2] in ['s', 'm', 'l']:
                area_rng = key[2]
            else:
                area_rng = 'all'

            print(
                template.format(title, _type, iou, area_rng, max_dets,
                                cat_group_name, value))

    def get_results(self):
        if not self.results:
            self.logger.warn('results is empty. Call run().')
        return self.results


class Params:
    def __init__(self, iou_type):
        """Params for LVIS evaluation API."""
        self.img_ids = []
        self.cat_ids = []
        # np.arange causes trouble.  the data point on arange is slightly
        # larger than the true value
        self.iou_thrs = np.linspace(0.5,
                                    0.95,
                                    int(np.round((0.95 - 0.5) / 0.05)) + 1,
                                    endpoint=True)
        self.rec_thrs = np.linspace(0.0,
                                    1.00,
                                    int(np.round((1.00 - 0.0) / 0.01)) + 1,
                                    endpoint=True)
        self.max_dets = 300
        self.area_rng = [
            [0**2, 1e5**2],
            [0**2, 32**2],
            [32**2, 96**2],
            [96**2, 1e5**2],
        ]
        self.area_rng_lbl = ['all', 'small', 'medium', 'large']
        self.use_cats = 1
        # We bin categories in three bins based how many images of the training
        # set the category is present in.
        # r: Rare    :  < 10
        # c: Common  : >= 10 and < 100
        # f: Frequent: >= 100
        self.img_count_lbl = ['r', 'c', 'f']
        self.iou_type = iou_type
