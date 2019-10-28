__author__ = 'tsungyi'

import numpy as np
import datetime
import time
from collections import defaultdict
from . import mask as maskUtils
import copy

class VideoEval:
    # Interface for evaluating video tracking / instance segmentation.
    #
    # The usage for VideoEval is as follows:
    #  cocoGt=..., cocoDt=...       # load dataset and results
    #  E = VideoEval(cocoGt,cocoDt); # initialize VideoEval object
    #  E.params.recThrs = ...;      # set parameters as desired
    #  E.evaluate();                # run per video evaluation
    #  E.accumulate();              # accumulate per video results
    #  E.summarize();               # display summary metrics of results
    # For example usage see evalDemo.m and http://mscoco.org/.
    #
    # The evaluation parameters are as follows (defaults in brackets):
    #  vidIds     - [all] N vid ids to use for evaluation
    #  catIds     - [all] K cat ids to use for evaluation
    #  iouThrs    - [.5:.05:.95] T=10 IoU thresholds for evaluation
    #  recThrs    - [0:.01:1] R=101 recall thresholds for evaluation
    #  areaRng    - [...] A=4 object area ranges for evaluation
    #  maxDets    - [1 10 100] M=3 thresholds on max detections per video
    #  iouType    - ['segm'] set iouType to 'segm', 'bbox' or 'keypoints'
    #  iouType replaced the now DEPRECATED useSegm parameter.
    #  useCats    - [1] if true use category labels for evaluation
    # Note: if useCats=0 category labels are ignored as in proposal scoring.
    # Note: multiple areaRngs [Ax2] and maxDets [Mx1] can be specified.
    #
    # evaluate(): evaluates detections on every video and every category and
    # concats the results into the "evalVids" with fields:
    #  dtIds      - [1xD] id for each of the D detections (dt)
    #  gtIds      - [1xG] id for each of the G ground truths (gt)
    #  dtMatches  - [TxD] matching gt id at each IoU or 0
    #  gtMatches  - [TxG] matching dt id at each IoU or 0
    #  dtScores   - [1xD] confidence of each dt
    #  gtIgnore   - [1xG] ignore flag for each gt
    #  dtIgnore   - [TxD] ignore flag for each dt at each IoU
    #
    # accumulate(): accumulates the per-video, per-category evaluation
    # results in "evalVids" into the dictionary "eval" with fields:
    #  params     - parameters used for evaluation
    #  date       - date evaluation was performed
    #  counts     - [T,R,K,A,M] parameter dimensions (see above)
    #  precision  - [TxRxKxAxM] precision for every evaluation setting
    #  recall     - [TxKxAxM] max recall for every evaluation setting
    # Note: precision and recall==-1 for settings with no gt objects.
    #
    # See also coco, mask, pycocoDemo, pycocoEvalDemo
    #
    # Microsoft COCO Toolbox.      version 2.0
    # Data, paper, and tutorials available at:  http://mscoco.org/
    # Code written by Piotr Dollar and Tsung-Yi Lin, 2015.
    # Licensed under the Simplified BSD License [see coco/license.txt]
    def __init__(self, cocoGt=None, cocoDt=None, iouType='segm'):
        '''
        Initialize CocoEval using coco APIs for gt and dt
        :param cocoGt: coco object with ground truth annotations
        :param cocoDt: coco object with detection results
        :return: None
        '''
        if not iouType:
            print('iouType not specified. use default iouType segm')
        self.cocoGt   = cocoGt              # ground truth COCO API
        self.cocoDt   = cocoDt              # detections COCO API
        self.evalVids = defaultdict(list)   # per-video per-category evaluation results [KxAxV] elements
        self.eval     = {}                  # accumulated evaluation results
        self._gts = defaultdict(list)       # gt for evaluation
        self._dts = defaultdict(list)       # dt for evaluation
        self.params = Params(iouType=iouType) # parameters
        self._paramsEval = {}               # parameters for evaluation
        self.stats = []                     # result summarization
        self.ious = {}                      # ious between all gts and dts
        if not cocoGt is None:
            self.params.vidIds = sorted(cocoGt.getVideoIds())
            self.params.catIds = sorted(cocoGt.getCatIds())


    def _prepare(self, with_occlusion=False):
        '''
        Prepare ._gts and ._dts for evaluation based on params
        :return: None
        '''
        def _toMask(anns, coco):
            # modify ann['segmentation'] by reference
            for ann in anns:
                rle = coco.annToRLE(ann)
                ann['segmentation'] = rle
        p = self.params
        # gt and dt tracks
        gts = []
        dts = []
        for vidId in p.vidIds:
            if p.useCats:
                gt=self.cocoGt.loadAnns(self.cocoGt.getAnnIds(vidIds=[vidId], catIds=p.catIds))
                dt=self.cocoDt.loadAnns(self.cocoDt.getAnnIds(vidIds=[vidId], catIds=p.catIds))
            else:
                gt=self.cocoGt.loadAnns(self.cocoGt.getAnnIds(vidIds=[vidId]))
                dt=self.cocoDt.loadAnns(self.cocoDt.getAnnIds(vidIds=[vidId]))
            imgIds = self.cocoGt.getImgIdsFromVideoId(vidId)
            gts += self.anns2Tracks(gt, imgIds, vidId, with_occlusion=with_occlusion)
            dts += self.anns2Tracks(dt, imgIds, vidId, with_occlusion=with_occlusion)

        # convert ground truth to mask if iouType == 'segm'
        # TODO: add instance segmentation
        if p.iouType == 'segm':
            _toMask(gts, self.cocoGt)
            _toMask(dts, self.cocoDt)
        # set ignore flag
        for gt in gts:
            gt['ignore'] = gt['ignore'] if 'ignore' in gt else 0
            gt['ignore'] = 'iscrowd' in gt and gt['iscrowd']
            if p.iouType == 'keypoints':
                gt['ignore'] = (gt['num_keypoints'] == 0) or gt['ignore']
        self._gts = defaultdict(list)       # gt for evaluation
        self._dts = defaultdict(list)       # dt for evaluation
        for gt in gts:
            self._gts[gt['video_id'], gt['category_id']].append(gt)
        for dt in dts:
            self._dts[dt['video_id'], dt['category_id']].append(dt)
        self.evalVids = defaultdict(list)   # per-video per-category evaluation results
        self.eval     = {}                  # accumulated evaluation results

    def evaluate(self, with_occlusion=False, vidIds=None):
        '''
        Run per video evaluation on given videos and store results (a list of dict) in self.evalVids
        :return: None
        '''
        tic = time.time()
        print('Running per video evaluation...')
        p = self.params
        # add backward compatibility if useSegm is specified in params
        if not p.useSegm is None:
            p.iouType = 'segm' if p.useSegm == 1 else 'bbox'
            print('useSegm (deprecated) is not None. Running {} evaluation'.format(p.iouType))
        print('Evaluate annotation type *{}*'.format(p.iouType))
        p.vidIds = list(np.unique(p.vidIds))
        # override vidIds
        if vidIds is not None:
            p.vidIds = vidIds
        if p.useCats:
            p.catIds = list(np.unique(p.catIds))
        p.maxDets = sorted(p.maxDets)
        self.params=p

        self._prepare(with_occlusion=with_occlusion)
        # loop through videos, area range, max detection number
        catIds = p.catIds if p.useCats else [-1]

        if p.iouType == 'segm' or p.iouType == 'bbox':
            computeIoU = self.computeIoU
        elif p.iouType == 'keypoints':
            computeIoU = self.computeOks
        print('Computing IoU...')
        self.ious = {(vidId, catId): computeIoU(vidId, catId) \
                        for vidId in p.vidIds
                        for catId in catIds}

        evaluateVid = self.evaluateVid
        maxDet = p.maxDets[-1]
        print('Matching...')
        self.evalVids = [evaluateVid(vidId, catId, areaRng, tempRng, maxDet)
                 for catId in catIds
                 for areaRng in p.areaRng
                 for tempRng in p.tempRng
                 for vidId in p.vidIds
             ]
        self._paramsEval = copy.deepcopy(self.params)
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format(toc-tic))

    def avgOverlap(self, dts, gts):
        dts = dts['track']
        gts = gts['track']
        # average overlap
        num_frames = 0
        cum_iou = 0
        for (d, g) in zip(dts, gts):
            if d is not None and g is not None:
                x1, y1 = max(d[0], g[0]), max(d[1], g[1])
                x2 = min(d[2] + d[0] - 1, g[2] + g[0] - 1)
                y2 = min(d[3] + d[1] - 1, g[3] + g[1] - 1)
                i = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
                u = g[2] * g[3] + d[2] * d[3] - i
                cum_iou += i / float(u)
            if (d is not None) or (g is not None):
                num_frames += 1
        return cum_iou / num_frames if num_frames > 0 else 0

    def avgOverlapAndDetOverlap(self, dts, gts):
        dts = dts['track']
        gts = gts['track']
        # average overlap
        num_frames = 0
        det_frames = 0
        cum_iou = 0
        for (d, g) in zip(dts, gts):
            if d is not None and g is not None:
                x1, y1 = max(d[0], g[0]), max(d[1], g[1])
                x2 = min(d[2] + d[0] - 1, g[2] + g[0] - 1)
                y2 = min(d[3] + d[1] - 1, g[3] + g[1] - 1)
                i = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
                u = g[2] * g[3] + d[2] * d[3] - i
                cum_iou += i / float(u)
            if (d is not None) or (g is not None):
                num_frames += 1
            if d is not None:
                det_frames += 1
        return cum_iou / num_frames if num_frames > 0 else 0, cum_iou / det_frames if det_frames > 0 else 0

    # convert annotations to tracks for a single video
    def anns2Tracks(self, anns, img_ids, vid_id, with_occlusion=True):
        if len(anns) == 0:
            return []
        p = self.params
        # get all IDs
        ids = np.unique([f['instance_id'] for f in anns])
        max_id = np.max(ids)
        tracks = {i: {
                        'id': i,
                        'video_id': vid_id,
                        'track': [None] * len(img_ids),
                        'avgArea': 0,
                        'length': 0
        } for i in ids}

        score_counters = defaultdict(int)
        remap_instance_id = defaultdict(int)
        crowd_count = 0
        for ann in anns:
            img_ind = img_ids.index(ann['image_id'])
            ins_id = ann['instance_id']
            ins_id = remap_instance_id[ins_id] if ins_id \
                        in remap_instance_id.keys() else ins_id
            # reappear from occlusion
            if not with_occlusion:
                if ins_id in tracks.keys() and \
                        tracks[ins_id]['length'] > 0 and \
                        tracks[ins_id]['track'][img_ind - 1] is None:
                    max_id += 1
                    remap_instance_id[ann['instance_id']] = max_id
                    tracks[max_id] = {
                                    'id': max_id,
                                    'video_id': vid_id,
                                    'track': [None] * len(img_ids),
                                    'avgArea': 0,
                                    'length': 0
                    }
                    ins_id = max_id

            ann_by_track = tracks[ins_id]
            if ('iscrowd' in ann) and not ('iscrowd' in ann_by_track):
                ann_by_track['iscrowd'] = ann['iscrowd']
            if 'category_id' not in ann_by_track:
                ann_by_track['category_id'] = ann['category_id']
            # bbox
            ann_by_track['track'][img_ind] = ann['bbox']
            ann_by_track['avgArea'] += ann['area']
            ann_by_track['length'] += 1
            # score
            if 'score' in ann.keys():
                if 'score' not in ann_by_track.keys():
                    ann_by_track['score'] = 0
                ann_by_track['score'] += ann['score']
                score_counters[ins_id] += 1

        for i in tracks.keys():
            tracks[i]['avgArea'] /= tracks[i]['length']

        # the score of the track is the average of that of each label
        if len(score_counters) > 0:
            for i in score_counters.keys():
                tracks[i]['score'] /= score_counters[i]
        return tracks.values()

    def computeIoU(self, vidId, catId):
        p = self.params
        if p.useCats:
            gt = self._gts[vidId,catId]
            dt = self._dts[vidId,catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[vidId,cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[vidId,cId]]
        if len(gt) == 0 and len(dt) ==0:
            return []
        inds = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in inds]
        if len(dt) > p.maxDets[-1]:
            dt=dt[0:p.maxDets[-1]]

        # compute iou between each dt and gt region
        iscrowd = [int(o['iscrowd']) for o in gt]
        ious = np.zeros([len(dt), len(gt)])
        det_ious = np.zeros([len(dt), len(gt)])

        for i, j in np.ndindex(ious.shape):
            ious[i, j], det_ious[i, j] = self.avgOverlapAndDetOverlap(dt[i], gt[j])
        return ious, det_ious

    def computeOks(self, vidId, catId):
        p = self.params
        # dimention here should be Nxm
        gts = self._gts[vidId, catId]
        dts = self._dts[vidId, catId]
        inds = np.argsort([-d['score'] for d in dts], kind='mergesort')
        dts = [dts[i] for i in inds]
        if len(dts) > p.maxDets[-1]:
            dts = dts[0:p.maxDets[-1]]
        # if len(gts) == 0 and len(dts) == 0:
        if len(gts) == 0 or len(dts) == 0:
            return []
        ious = np.zeros((len(dts), len(gts)))
        sigmas = p.kpt_oks_sigmas
        vars = (sigmas * 2)**2
        k = len(sigmas)
        # compute oks between each detection and ground truth object
        for j, gt in enumerate(gts):
            # create bounds for ignore regions(double the gt bbox)
            g = np.array(gt['keypoints'])
            xg = g[0::3]; yg = g[1::3]; vg = g[2::3]
            k1 = np.count_nonzero(vg > 0)
            bb = gt['bbox']
            x0 = bb[0] - bb[2]; x1 = bb[0] + bb[2] * 2
            y0 = bb[1] - bb[3]; y1 = bb[1] + bb[3] * 2
            for i, dt in enumerate(dts):
                d = np.array(dt['keypoints'])
                xd = d[0::3]; yd = d[1::3]
                if k1>0:
                    # measure the per-keypoint distance if keypoints visible
                    dx = xd - xg
                    dy = yd - yg
                else:
                    # measure minimum distance to keypoints in (x0,y0) & (x1,y1)
                    z = np.zeros((k))
                    dx = np.max((z, x0-xd),axis=0)+np.max((z, xd-x1),axis=0)
                    dy = np.max((z, y0-yd),axis=0)+np.max((z, yd-y1),axis=0)
                e = (dx**2 + dy**2) / vars / (gt['avgArea']+np.spacing(1)) / 2
                if k1 > 0:
                    e=e[vg > 0]
                ious[i, j] = np.sum(np.exp(-e)) / e.shape[0]
        return ious

    def evaluateVid(self, vidId, catId, aRng, tempRng, maxDet):
        '''
        perform evaluation for single category and video
        :return: dict (single video results)
        '''
        p = self.params
        if p.useCats:
            gt = self._gts[vidId,catId]
            dt = self._dts[vidId,catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[vidId,cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[vidId,cId]]
        if len(gt) == 0 and len(dt) ==0:
            return None

        for g in gt:
            if g['ignore'] or (g['avgArea']<aRng[0] or g['avgArea']>aRng[1]) or \
                              (g['length']<tempRng[0] or g['length']>tempRng[1]):
                g['_ignore'] = 1
            else:
                g['_ignore'] = 0

        # sort dt highest score first, sort gt ignore last
        gtind = np.argsort([g['_ignore'] for g in gt], kind='mergesort')
        gt = [gt[i] for i in gtind]
        dtind = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in dtind[0:maxDet]]
        iscrowd = [int(o['iscrowd']) for o in gt]
        # load computed ious
        ious, det_ious = self.ious[vidId, catId][0][:, gtind], self.ious[vidId, catId][1][:, gtind] if len(self.ious[vidId, catId]) > 0 else self.ious[vidId, catId]
        T = len(p.iouThrs)
        G = len(gt)
        D = len(dt)
        gtm  = np.zeros((T,G))
        dtm  = np.zeros((T,D))
        gtIg = np.array([g['_ignore'] for g in gt])
        dtIg = np.zeros((T,D))
        if not len(ious)==0:
            for tind, t in enumerate(p.iouThrs):
                counter = 0
                for dind, d in enumerate(dt):
                    # information about best match so far (m=-1 -> unmatched)
                    iou = min([t,1-1e-10])
                    m   = -1
                    matched_previously_matched = False
                    for gind, g in enumerate(gt):
                        # if this gt already matched, and not a crowd, continue
                        if det_ious[dind,gind] >= iou:
                            matched_previously_matched = True
                        if gtm[tind,gind]>0 and not iscrowd[gind]:
                            # set this prediction to ignored if det IoU > iou
                            continue
                        # if dt matched to reg gt, and on ignore gt, stop
                        if m>-1 and gtIg[m]==0 and gtIg[gind]==1:
                            break
                        # continue to next gt unless better match made
                        if ious[dind,gind] < iou:
                            continue
                        # if match successful and best so far, store appropriately
                        iou=ious[dind,gind]
                        m=gind
                    # if match made store id of match for both dt and gt
                    if m ==-1:
                        if matched_previously_matched:
                            dtIg[tind, dind] = 1
                        continue
                    dtIg[tind,dind] = gtIg[m]
                    dtm[tind,dind]  = gt[m]['id']
                    gtm[tind,m]     = d['id']
                    counter += 1
        # set unmatched detections outside of area and temporal range to ignore
        a = np.array([d['avgArea']<aRng[0] or d['avgArea']>aRng[1] or d['length']<tempRng[0] or d['length']>tempRng[1] for d in dt]).reshape((1, len(dt)))
        dtIg = np.logical_or(dtIg, np.logical_and(dtm==0, np.repeat(a,T,0)))
        # store results for given video and category
        return {
                'video_id':     vidId,
                'category_id':  catId,
                'aRng':         aRng,
                'tempRng':         tempRng,
                'maxDet':       maxDet,
                'dtIds':        [d['id'] for d in dt],
                'gtIds':        [g['id'] for g in gt],
                'dtMatches':    dtm,
                'gtMatches':    gtm,
                'dtScores':     [d['score'] for d in dt],
                'gtIgnore':     gtIg,
                'dtIgnore':     dtIg,
            }

    def accumulate(self, p = None):
        '''
        Accumulate per video evaluation results and store the result in self.eval
        :param p: input params for evaluation
        :return: None
        '''
        print('Accumulating evaluation results...')
        tic = time.time()
        if not self.evalVids:
            print('Please run evaluate() first')
        # allows input customized parameters
        if p is None:
            p = self.params
        p.catIds = p.catIds if p.useCats == 1 else [-1]
        T           = len(p.iouThrs)
        R           = len(p.recThrs)
        K           = len(p.catIds) if p.useCats else 1
        A           = len(p.areaRng)
        TEMP        = len(p.tempRng)
        M           = len(p.maxDets)
        precision   = -np.ones((T,R,K,A,TEMP,M)) # -1 for the precision of absent categories
        recall      = -np.ones((T,K,A,TEMP,M))
        scores      = -np.ones((T,R,K,A,TEMP,M))

        # create dictionary for future indexing
        _pe = self._paramsEval
        catIds = _pe.catIds if _pe.useCats else [-1]
        setK = set(catIds)
        setA = set(map(tuple, _pe.areaRng))
        setTEMP = set(map(tuple, _pe.tempRng))
        setM = set(_pe.maxDets)
        setI = set(_pe.vidIds)
        # get inds to evaluate
        k_list = [n for n, k in enumerate(p.catIds)  if k in setK]
        m_list = [m for n, m in enumerate(p.maxDets) if m in setM]
        a_list = [n for n, a in enumerate(map(lambda x: tuple(x), p.areaRng)) if a in setA]
        temp_list = [n for n, a in enumerate(map(lambda x: tuple(x), p.tempRng)) if a in setTEMP]
        i_list = [n for n, i in enumerate(p.vidIds)  if i in setI]
        I0 = len(_pe.vidIds)
        A0 = len(_pe.areaRng)
        TEMP0 = len(_pe.tempRng)
        # retrieve E at each category, area range, temporal range, and max number of detections
        for k, k0 in enumerate(k_list):
            Nk = k0*A0*TEMP0*I0
            for a, a0 in enumerate(a_list):
                Na = a0*TEMP0*I0
                for temp, temp0 in enumerate(temp_list):
                    Ntemp = temp0*I0
                    for m, maxDet in enumerate(m_list):
                        E = [self.evalVids[Nk + Na + Ntemp + i] for i in i_list]
                        E = [e for e in E if not e is None]
                        if len(E) == 0:
                            continue
                        dtScores = np.concatenate([e['dtScores'][0:maxDet] for e in E])
                        # different sorting method generates slightly different results.
                        # mergesort is used to be consistent as Matlab implementation.
                        inds = np.argsort(-dtScores, kind='mergesort')
                        dtScoresSorted = dtScores[inds]

                        dtm  = np.concatenate([e['dtMatches'][:,0:maxDet] for e in E], axis=1)[:,inds]
                        dtIg = np.concatenate([e['dtIgnore'][:,0:maxDet]  for e in E], axis=1)[:,inds]
                        gtIg = np.concatenate([e['gtIgnore'] for e in E])
                        npig = np.count_nonzero(gtIg==0 )
                        if npig == 0:
                            continue
                        tps = np.logical_and(               dtm,  np.logical_not(dtIg) )
                        fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg) )
                        tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float)
                        fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float)
                        for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                            tp = np.array(tp)
                            fp = np.array(fp)
                            nd = len(tp)
                            rc = tp / npig
                            pr = tp / (fp+tp+np.spacing(1))
                            q  = np.zeros((R,))
                            ss = np.zeros((R,))

                            if nd:
                                recall[t,k,a,temp,m] = rc[-1]
                            else:
                                recall[t,k,a,temp,m] = 0

                            # numpy is slow without cython optimization for accessing elements
                            # use python array gets significant speed improvement
                            pr = pr.tolist(); q = q.tolist()

                            for i in range(nd-1, 0, -1):
                                if pr[i] > pr[i-1]:
                                    pr[i-1] = pr[i]

                            inds = np.searchsorted(rc, p.recThrs, side='left')
                            try:
                                for ri, pi in enumerate(inds):
                                    q[ri] = pr[pi]
                                    ss[ri] = dtScoresSorted[pi]
                            except:
                                pass
                            precision[t,:,k,a,temp,m] = np.array(q)
                            scores[t,:,k,a,temp,m] = np.array(ss)
        self.eval = {
            'params': p,
            'counts': [T, R, K, A, TEMP, M],
            'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'precision': precision,
            'recall':   recall,
            'scores': scores,
        }
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format( toc-tic))

    def summarize(self, by_cat=True):
        '''
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        '''
        def _summarize( ap=1, iouThr=None, catId=None, areaRng='all', tempRng='all', maxDets=1000 ):
            p = self.params
            iStr = ' {:<18} {} @[ IoU={:<9} | catId={:>3s} | area={:>6s} | length={:>6s} | maxDets={:>3d} ] = {:0.3f}'
            titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
            typeStr = '(AP)' if ap==1 else '(AR)'
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(iouThr)

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            tempind = [i for i, tRng in enumerate(p.tempRngLbl) if tRng == tempRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            cind = [i for i, cat in enumerate(p.catIds) if cat == catId] if catId else range(len(p.catIds))
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval['precision']
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:,:,cind,aind,tempind,mind]
            else:
                # dimension of recall: [TxKxAxM]
                s = self.eval['recall']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:,cind,aind,tempind,mind]
            if len(s[s>-1])==0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s>-1])
            catStr = 'all' if catId is None else str(catId)
            print(iStr.format(titleStr, typeStr, iouStr, catStr, areaRng, tempRng, maxDets, mean_s))
            return mean_s
        def _summarizeDets(catId=None):
            stats = np.zeros((18,))
            stats[0] = _summarize(1, catId=catId)
            stats[1] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2], catId=catId)
            stats[2] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[2], catId=catId)
            stats[3] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[2], catId=catId)
            stats[4] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[2], catId=catId)
            stats[5] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[2], catId=catId)
            stats[6] = _summarize(1, tempRng='short', maxDets=self.params.maxDets[2], catId=catId)
            stats[7] = _summarize(1, tempRng='medium', maxDets=self.params.maxDets[2], catId=catId)
            stats[8] = _summarize(1, tempRng='long', maxDets=self.params.maxDets[2], catId=catId)
            stats[9] = _summarize(0, maxDets=self.params.maxDets[0], catId=catId)
            stats[10] = _summarize(0, maxDets=self.params.maxDets[1], catId=catId)
            stats[11] = _summarize(0, maxDets=self.params.maxDets[2], catId=catId)
            stats[12] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[2], catId=catId)
            stats[13] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[2], catId=catId)
            stats[14] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[2], catId=catId)
            stats[15] = _summarize(0, tempRng='short', maxDets=self.params.maxDets[2], catId=catId)
            stats[16] = _summarize(0, tempRng='medium', maxDets=self.params.maxDets[2], catId=catId)
            stats[17] = _summarize(0, tempRng='long', maxDets=self.params.maxDets[2], catId=catId)
            return stats
        def _summarizeKps(catId=None):
            stats = np.zeros((10,))
            stats[0] = _summarize(1, maxDets=20)
            stats[1] = _summarize(1, maxDets=20, iouThr=.5)
            stats[2] = _summarize(1, maxDets=20, iouThr=.75)
            stats[3] = _summarize(1, maxDets=20, areaRng='medium')
            stats[4] = _summarize(1, maxDets=20, areaRng='large')
            stats[5] = _summarize(0, maxDets=20)
            stats[6] = _summarize(0, maxDets=20, iouThr=.5)
            stats[7] = _summarize(0, maxDets=20, iouThr=.75)
            stats[8] = _summarize(0, maxDets=20, areaRng='medium')
            stats[9] = _summarize(0, maxDets=20, areaRng='large')
            return stats
        if not self.eval:
            raise Exception('Please run accumulate() first')
        iouType = self.params.iouType
        if iouType == 'segm' or iouType == 'bbox':
            summarize = _summarizeDets
        elif iouType == 'keypoints':
            summarize = _summarizeKps
        if by_cat:
            for cat in [None] + self.params.catIds:
                self.stats = summarize(catId=cat)

    def __str__(self):
        self.summarize()

class Params:
    '''
    Params for coco evaluation api
    '''
    def setDetParams(self):
        self.vidIds = []
        self.catIds = []
        # np.arange causes trouble.  the data point on arange is slightly larger than the true value
        # self.iouThrs = np.linspace(.1, 0.95, np.round((0.95 - .1) / .05) + 1, endpoint=True)
        self.iouThrs = np.array([0.5, 0.55, 0.6, 0.65, 0.7, 0.75])
        self.recThrs = np.linspace(.0, 1.00, np.round((1.00 - .0) / .01) + 1, endpoint=True)
        self.maxDets = [10, 100, 1000]
        self.areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, 32 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
        self.areaRngLbl = ['all', 'small', 'medium', 'large']
        self.tempRng = [[0, 1e5], [0, 10], [10, 25], [25, 1e5]]
        self.tempRngLbl = ['all', 'short', 'medium', 'long']
        self.useCats = 1

    def setKpParams(self):
        self.vidIds = []
        self.catIds = []
        # np.arange causes trouble.  the data point on arange is slightly larger than the true value
        self.iouThrs = np.linspace(.5, 0.95, np.round((0.95 - .5) / .05) + 1, endpoint=True)
        self.recThrs = np.linspace(.0, 1.00, np.round((1.00 - .0) / .01) + 1, endpoint=True)
        self.maxDets = [20]
        self.areaRng = [[0 ** 2, 1e5 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
        self.areaRngLbl = ['all', 'medium', 'large']
        self.useCats = 1
        self.kpt_oks_sigmas = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62,.62, 1.07, 1.07, .87, .87, .89, .89])/10.0

    def __init__(self, iouType='segm'):
        if iouType == 'segm' or iouType == 'bbox':
            self.setDetParams()
        elif iouType == 'keypoints':
            self.setKpParams()
        else:
            raise Exception('iouType not supported')
        self.iouType = iouType
        # useSegm is deprecated
        self.useSegm = None
