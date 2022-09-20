import copy
import datetime
import time
import warnings
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from . import mask as maskUtils

from ocr_metric import eval_ocr_metric


warnings.filterwarnings(action='ignore', message='Mean of empty slice')


class COCOeval:
    # Interface for evaluating detection on the Microsoft COCO dataset.
    #
    # The usage for CocoEval is as follows:
    #  cocoGt=..., cocoDt=...       # load dataset and results
    #  E = CocoEval(cocoGt,cocoDt); # initialize CocoEval object
    #  E.params.recThrs = ...;      # set parameters as desired
    #  E.evaluate();                # run per image evaluation
    #  E.accumulate();              # accumulate per image results
    #  E.summarize();               # display summary metrics of results
    # For example usage see evalDemo.m and http://mscoco.org/.
    #
    # The evaluation parameters are as follows (defaults in brackets):
    #  imgIds     - [all] N img ids to use for evaluation
    #  catIds     - [all] K cat ids to use for evaluation
    #  iouThrs    - [.5:.05:.95] T=10 IoU thresholds for evaluation
    #  recThrs    - [0:.01:1] R=101 recall thresholds for evaluation
    #  areaRng    - [...] A=4 object area ranges for evaluation
    #  maxDets    - [1 10 100] M=3 thresholds on max detections per image
    #  iouType    - ['segm'] set iouType to 'segm', 'bbox' or 'keypoints'
    #  iouType replaced the now DEPRECATED useSegm parameter.
    #  useCats    - [1] if true use category labels for evaluation
    # Note: if useCats=0 category labels are ignored as in proposal scoring.
    # Note: multiple areaRngs [Ax2] and maxDets [Mx1] can be specified.
    #
    # evaluate(): evaluates detections on every image and every category and
    # concats the results into the 'evalImgs' with fields:
    #  dtIds      - [1xD] id for each of the D detections (dt)
    #  gtIds      - [1xG] id for each of the G ground truths (gt)
    #  dtMatches  - [TxD] matching gt id at each IoU or 0
    #  gtMatches  - [TxG] matching dt id at each IoU or 0
    #  dtScores   - [1xD] confidence of each dt
    #  gtIgnore   - [1xG] ignore flag for each gt
    #  dtIgnore   - [TxD] ignore flag for each dt at each IoU
    #
    # accumulate(): accumulates the per-image, per-category evaluation
    # results in 'evalImgs' into the dictionary 'eval' with fields:
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
        self.evalImgs = defaultdict(list)   # per-image per-category evaluation results [KxAxI] elements
        self.eval     = {}                  # accumulated evaluation results
        self.evalFBeta = {}                 # accumulated F-beta evaluation results
        self._gts = defaultdict(list)       # gt for evaluation
        self._dts = defaultdict(list)       # dt for evaluation
        self.params = Params(iouType=iouType) # parameters
        self._paramsEval = {}               # parameters for evaluation
        self.stats = []                     # result summarization (COCO-style)
        self.fstats = []                    # result summarization
        self.ious = {}                      # ious between all gts and dts
        if not cocoGt is None:
            self.params.imgIds = sorted(cocoGt.getImgIds())
            self.params.catIds = sorted(cocoGt.getCatIds())
            self.params.catNms = cocoGt.getCatNms()
            self.params.catIdsToCatNms = dict(zip(self.params.catIds, self.params.catNms))

    def _prepare(self):
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
        if p.useCats:
            gts=self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
            dts=self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
        else:
            gts=self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds))
            dts=self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds))

        # convert ground truth to mask if iouType == 'segm'
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
            self._gts[gt['image_id'], gt['category_id']].append(gt)
        for dt in dts:
            self._dts[dt['image_id'], dt['category_id']].append(dt)
        self.evalImgs = defaultdict(list)   # per-image per-category evaluation results
        self.eval     = {}                  # accumulated evaluation results

    def evaluate(self):
        '''
        Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
        :return: None
        '''
        tic = time.time()
        print('Running per image evaluation...')
        p = self.params
        # add backward compatibility if useSegm is specified in params
        if not p.useSegm is None:
            p.iouType = 'segm' if p.useSegm == 1 else 'bbox'
            print('useSegm (deprecated) is not None. Running {} evaluation'.format(p.iouType))
        print('Evaluate annotation type *{}*'.format(p.iouType))
        p.imgIds = list(np.unique(p.imgIds))
        if p.useCats:
            p.catIds = list(np.unique(p.catIds))
        p.maxDets = sorted(p.maxDets)
        self.params=p

        self._prepare()
        # loop through images, area range, max detection number
        catIds = p.catIds if p.useCats else [-1]

        if p.iouType == 'segm' or p.iouType == 'bbox':
            computeIoU = self.computeIoU
        elif p.iouType == 'keypoints':
            computeIoU = self.computeOks
        self.ious = {(imgId, catId): computeIoU(imgId, catId) \
                        for imgId in p.imgIds
                        for catId in catIds}

        evaluateImg = self.evaluateImg
        maxDet = p.maxDets[-1]
        self.evalImgs = [evaluateImg(imgId, catId, areaRng, maxDet)
                 for catId in catIds
                 for areaRng in p.areaRng
                 for imgId in p.imgIds
             ]
        self._paramsEval = copy.deepcopy(self.params)
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format(toc-tic))

    def computeIoU(self, imgId, catId):
        p = self.params
        if p.useCats:
            gt = self._gts[imgId,catId]
            dt = self._dts[imgId,catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId,cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId,cId]]
        if len(gt) == 0 and len(dt) ==0:
            return []
        inds = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in inds]
        if len(dt) > p.maxDets[-1]:
            dt=dt[0:p.maxDets[-1]]

        if p.iouType == 'segm':
            g = [g['segmentation'] for g in gt]
            d = [d['segmentation'] for d in dt]
        elif p.iouType == 'bbox':
            g = [g['bbox'] for g in gt]
            d = [d['bbox'] for d in dt]
        else:
            raise Exception('unknown iouType for iou computation')

        # compute iou between each dt and gt region
        iscrowd = [int(o['iscrowd']) for o in gt]
        ious = maskUtils.iou(d,g,iscrowd)
        return ious

    def computeOks(self, imgId, catId):
        p = self.params
        # dimention here should be Nxm
        gts = self._gts[imgId, catId]
        dts = self._dts[imgId, catId]
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
                e = (dx**2 + dy**2) / vars / (gt['area']+np.spacing(1)) / 2
                if k1 > 0:
                    e=e[vg > 0]
                ious[i, j] = np.sum(np.exp(-e)) / e.shape[0]
        return ious

    def evaluateImg(self, imgId, catId, aRng, maxDet):
        '''
        perform evaluation for single category and image
        :return: dict (single image results)
        '''
        p = self.params
        if p.useCats:
            gt = self._gts[imgId,catId]
            dt = self._dts[imgId,catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId,cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId,cId]]
        if len(gt) == 0 and len(dt) ==0:
            return None

        for g in gt:
            if g['ignore'] or (g['area']<aRng[0] or g['area']>aRng[1]):
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
        ious = self.ious[imgId, catId][:, gtind] if len(self.ious[imgId, catId]) > 0 else self.ious[imgId, catId]

        T = len(p.iouThrs)
        G = len(gt)
        D = len(dt)
        gtm  = np.zeros((T,G))
        dtm  = np.zeros((T,D))
        gtIg = np.array([g['_ignore'] for g in gt])
        dtIg = np.zeros((T,D))
        if not len(ious)==0:
            for tind, t in enumerate(p.iouThrs):
                for dind, d in enumerate(dt):
                    # information about best match so far (m=-1 -> unmatched)
                    iou = min([t,1-1e-10])
                    m   = -1
                    for gind, g in enumerate(gt):
                        # if this gt already matched, and not a crowd, continue
                        if gtm[tind,gind]>0 and not iscrowd[gind]:
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
                        continue
                    dtIg[tind,dind] = gtIg[m]
                    dtm[tind,dind]  = gt[m]['id']
                    gtm[tind,m]     = d['id']
        # set unmatched detections outside of area range to ignore
        a = np.array([d['area']<aRng[0] or d['area']>aRng[1] for d in dt]).reshape((1, len(dt)))
        dtIg = np.logical_or(dtIg, np.logical_and(dtm==0, np.repeat(a,T,0)))
        # store results for given image and category
        
        '''
        for dtm, the ids in each line represents the gt_id that each dt matched with (ids start from 1; 0 = match with nothing)
        Each line should represent the scale of the detection based on cocoapi

        So, how to compare?
        1. Walk through dt and line at the same time
        2. Match each dt with the gt[id-1] for each id in line
        3. ???
        4. Profit! 
        '''
        try:
            ### BEGIN ADDITION ###
            gt = sorted(gt, key=lambda item: item['id'])
            gtIds = [g['id'] for g in gt]
            dtm = np.where(dtIg == True, 0, dtm).astype(int)
            dtmWord = []
            gtmWord = []
            for line in dtm:
                temp1, temp2 = [], []
                for id, x in zip(line, dt):
                    # print(int(id)-1, gt[int(id)-1]['attributes']['Text'], x['attributes']['Text'], line, [(g['id'], g['attributes']['Text']) for g in gt])
                    try:
                        # temp1.append(gt[int(id)-1]['attributes']['Text'] if int(id) > 0 else None)
                        temp1.append(gt[gtIds.index(id)]['attributes']['Text'] if int(id) > 0 else None)
                    except Exception:
                        temp1.append(None)
                    temp2.append(x['attributes']['Text'])
                    #     print('gtdt', gt[list(gtind).index(id)]['attributes']['Text'], x['attributes']['Text'])
                    # print(id, x, gtIds)
                # print(line, temp1, temp2)
                dtmWord.append(temp1)
                gtmWord.append(temp2)
            ### END ADDITION ###
            return {
                    'image_id':     imgId,
                    'category_id':  catId,
                    'aRng':         aRng,
                    'maxDet':       maxDet,
                    'dtIds':        [d['id'] for d in dt],
                    'gtIds':        gtIds,
                    'dtMatches':    dtm,
                    'gtMatches':    gtm,
                    ### BEGIN ADDITION ###
                    'dtMatchWord':  dtmWord, 
                    'gtMatchWord':  gtmWord,
                    ### BEGIN ADDITION ###
                    'dtScores':     [d['score'] for d in dt],
                    'gtIgnore':     gtIg,
                    'dtIgnore':     dtIg,
                }
        except Exception as e:
            print('WARNING: {}, defaulting output'.format(e))
            return {
                    'image_id':     imgId,
                    'category_id':  catId,
                    'aRng':         aRng,
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
        Accumulate per image evaluation results and store the result in self.eval
        :param p: input params for evaluation
        :return: None
        '''
        print('Accumulating evaluation results...')
        tic = time.time()
        if not self.evalImgs:
            print('Please run evaluate() first')
        # allows input customized parameters
        if p is None:
            p = self.params
        p.catIds = p.catIds if p.useCats == 1 else [-1]
        T           = len(p.iouThrs)
        R           = len(p.recThrs)
        K           = len(p.catIds) if p.useCats else 1
        A           = len(p.areaRng)
        M           = len(p.maxDets)
        precision   = -np.ones((T,R,K,A,M)) # -1 for the precision of absent categories
        recall      = -np.ones((T,K,A,M))
        scores      = -np.ones((T,R,K,A,M))

        # create dictionary for future indexing
        _pe = self._paramsEval
        catIds = _pe.catIds if _pe.useCats else [-1]
        setK = set(catIds)
        setA = set(map(tuple, _pe.areaRng))
        setM = set(_pe.maxDets)
        setI = set(_pe.imgIds)
        # get inds to evaluate
        k_list = [n for n, k in enumerate(p.catIds)  if k in setK]
        m_list = [m for n, m in enumerate(p.maxDets) if m in setM]
        a_list = [n for n, a in enumerate(map(lambda x: tuple(x), p.areaRng)) if a in setA]
        i_list = [n for n, i in enumerate(p.imgIds)  if i in setI]
        I0 = len(_pe.imgIds)
        A0 = len(_pe.areaRng)
        # retrieve E at each category, area range, and max number of detections
        for k, k0 in enumerate(k_list):
            Nk = k0*A0*I0
            for a, a0 in enumerate(a_list):
                Na = a0*I0
                for m, maxDet in enumerate(m_list):
                    E = [self.evalImgs[Nk + Na + i] for i in i_list]
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
                    tps = np.logical_and(dtm, np.logical_not(dtIg) )
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
                            recall[t,k,a,m] = rc[-1]
                        else:
                            recall[t,k,a,m] = 0

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
                        precision[t,:,k,a,m] = np.array(q)
                        scores[t,:,k,a,m] = np.array(ss)
        self.eval = {
            'params': p,
            'counts': [T, R, K, A, M],
            'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'precision': precision,
            'recall':   recall,
            'scores': scores,
        }
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format( toc-tic))

    def accumulateText(self, p = None):
        '''
        Accumulate per image evaluation results and store the result in self.eval
        :param p: input params for evaluation
        :return: None
        '''
        print('Accumulating evaluation results...')
        tic = time.time()
        if not self.evalImgs:
            print('Please run evaluate() first')
        # allows input customized parameters
        if p is None:
            p = self.params
        p.catIds = p.catIds if p.useCats == 1 else [-1]
        T           = len(p.iouThrs)
        R           = len(p.recThrs)
        K           = len(p.catIds) if p.useCats else 1
        A           = len(p.areaRng)
        M           = len(p.maxDets)
        precision   = -np.ones((T,R,K,A,M)) # -1 for the precision of absent categories
        recall      = -np.ones((T,K,A,M))
        scores      = -np.ones((T,R,K,A,M))
        '''
        Notes:
        text_scores size can be of size T, x, K, A, M, 
        where x is the absolute maximum # of detections possible, or maybe just set it to 1?
        IT'S NOT M, because M is the maximum matches, which can be less than absolute maximum # of detections

        So this test case should have shape of (10,4,1,4,3)
        '''
        text_scores = np.zeros((T,6,K,A,M))

        # create dictionary for future indexing
        _pe = self._paramsEval
        catIds = _pe.catIds if _pe.useCats else [-1]
        setK = set(catIds)
        setA = set(map(tuple, _pe.areaRng))
        setM = set(_pe.maxDets)
        setI = set(_pe.imgIds)
        # get inds to evaluate
        k_list = [n for n, k in enumerate(p.catIds)  if k in setK]
        m_list = [m for n, m in enumerate(p.maxDets) if m in setM]
        a_list = [n for n, a in enumerate(map(lambda x: tuple(x), p.areaRng)) if a in setA]
        i_list = [n for n, i in enumerate(p.imgIds)  if i in setI]
        I0 = len(_pe.imgIds)
        A0 = len(_pe.areaRng)
        # retrieve E at each category, area range, and max number of detections
        for k, k0 in enumerate(k_list):
            Nk = k0*A0*I0
            for a, a0 in enumerate(a_list):
                Na = a0*I0
                for m, maxDet in enumerate(m_list):
                    E = [self.evalImgs[Nk + Na + i] for i in i_list]
                    E = [e for e in E if not e is None]
                    if len(E) == 0:
                        continue
                    dtScores = np.concatenate([e['dtScores'][0:maxDet] for e in E])

                    # different sorting method generates slightly different results.
                    # mergesort is used to be consistent as Matlab implementation.
                    inds = np.argsort(-dtScores, kind='mergesort')
                    dtScoresSorted = dtScores[inds]

                    dtm  = np.concatenate([e['dtMatches'][:,0:maxDet] for e in E], axis=1) 
                    dtw  = [x[:len(dtm[0])] for x in [e['dtMatchWord'] for e in E][0]]
                    gtw  = [x[:len(dtm[0])] for x in [e['gtMatchWord'] for e in E][0]]
                    # print('data?', dtm, dtw, gtw)

                    dtIg = np.concatenate([e['dtIgnore'][:,0:maxDet]  for e in E], axis=1)[:,inds]
                    gtIg = np.concatenate([e['gtIgnore'] for e in E])
                    npig = np.count_nonzero(gtIg==0 )
                    if npig == 0:
                        continue
                    tps = np.logical_and(               dtm,  np.logical_not(dtIg) )
                    fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg) )
                    # print('TPS',tps)
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
                            recall[t,k,a,m] = rc[-1]
                        else:
                            recall[t,k,a,m] = 0

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
                        precision[t,:,k,a,m] = np.array(q)
                        scores[t,:,k,a,m] = np.array(ss)
                        ocr_score = np.array([value for key, value in eval_ocr_metric(dtw[t], gtw[t]).items()])
                        text_scores[t,:,k,a,m] = ocr_score
        self.eval = {
            'params': p,
            'counts': [T, R, K, A, M],
            'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'precision': precision,
            'recall':   recall,
            'scores': scores,
            'ocr_scores': text_scores
        }
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format( toc-tic))

    def summarize(self):
        '''
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        '''
        def _summarize( ap=1, iouThr=None, areaRng='all', maxDets=100 ):
            p = self.params
            iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
            titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
            typeStr = '(AP)' if ap==1 else '(AR)'
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(iouThr)

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval['precision']
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:,:,:,aind,mind]
            else:
                # dimension of recall: [TxKxAxM]
                s = self.eval['recall']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:,:,aind,mind]
            if len(s[s>-1])==0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s>-1])
            print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
            return mean_s

        def _summarizeRecog( iouThr=None, areaRng='all', maxDets=100 ):
            p = self.params
            iStr = ' {:<10} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {}'
            titleStr = 'OCR Results'
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(iouThr)

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            text_result = 'N/A'
            ### BEGIN ADDITION ###
            ocr_result = self.eval['ocr_scores']
            ### END ADDITION ###
            # dimension of precision: [TxRxKxAxM]
            # IoU
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                ocr_result = ocr_result[t]
            ocr_result = ocr_result[:,:,:,aind,mind]
            ### BEGIN ADDITION ###
            # print('OCR Result', np.concatenate(np.mean([np.concatenate(x) for x in ocr_result], axis=0)), len(ocr_result), len(ocr_result[0]))
            keys = ['word_acc', 'word_acc_ignore_case', 'word_acc_ignore_case_symbol', 'char_recall', 'char_precision', '1-N.E.D']
            values = np.concatenate(np.mean([np.concatenate(x) for x in ocr_result], axis=0))
            
            text_result = {key : round(value, 4) for key, value in zip(keys, values)}
            ### END ADDITION ###
            print(iStr.format(titleStr, iouStr, areaRng, maxDets, text_result)) ### Added text_result ###
            return text_result['1-N.E.D']

        def _summarizeDets():
            stats = np.zeros((12,))
            stats[0] = _summarize(1)
            stats[1] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2])
            stats[2] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[2])
            stats[3] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[2])
            stats[4] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[2])
            stats[5] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[2])
            stats[6] = _summarize(0, maxDets=self.params.maxDets[0])
            stats[7] = _summarize(0, maxDets=self.params.maxDets[1])
            stats[8] = _summarize(0, maxDets=self.params.maxDets[2])
            stats[9] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[2])
            stats[10] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[2])
            stats[11] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[2])
            return stats
        def _summarizeDetRecogs():
            '''
            - word_acc: Accuracy in word level.
            - word_acc_ignore_case: Accuracy in word level, ignore letter case.
            - word_acc_ignore_case_symbol: Accuracy in word level, ignore
                letter case and symbol. (default metric for
                academic evaluation)
            - char_recall: Recall in character level, ignore
                letter case and symbol.
            - char_precision: Precision in character level, ignore
                letter case and symbol.
            - 1-N.E.D: 1 - normalized_edit_distance.
            '''
            stats = np.zeros((12,))
            stats[0] = _summarizeRecog()
            stats[1] = _summarizeRecog(iouThr=.5, maxDets=self.params.maxDets[2])
            stats[2] = _summarizeRecog(iouThr=.75, maxDets=self.params.maxDets[2])
            stats[3] = _summarizeRecog(areaRng='small', maxDets=self.params.maxDets[2])
            stats[4] = _summarizeRecog(areaRng='medium', maxDets=self.params.maxDets[2])
            stats[5] = _summarizeRecog(areaRng='large', maxDets=self.params.maxDets[2])
        def _summarizeKps():
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
        self.stats = summarize()
        if 'ocr_scores' in self.eval:
            _summarizeDetRecogs()

    def plotCocoPRCurve(self, filename, classIdx=None):
        '''
        Plot COCO-style Precision-Recall curves
        :param filename: output filename
        :param classIdx: to plot for a specific class
        :return: None
        '''
        if not self.eval:
            raise Exception('Please run accumulate() first')

        p = self.params

        if classIdx is not None:
            className = p.catIdsToCatNms[classIdx]

        precisions = self.eval['precision']

        if classIdx is not None:
            prArray = precisions[:, :, classIdx, 0, 2]
        else:
            prArray = np.mean(precisions[:, :, :, 0, 2], axis=2)

        x = np.arange(0.0, 1.01, 0.01)
        plt.figure()
        if classIdx is None:
            title = f'P-R curve'
        else:
            title = f'P-R curve for class={className}'

        for idx, iouThr in enumerate(p.iouThrs):
            plt.plot(x, prArray[idx, :], label=f'iou={iouThr:0.2f}')

        plt.title(title)
        plt.xlabel('recall')
        plt.ylabel('precision')
        plt.xlim(0, 1.0)
        plt.ylim(0, 1.01)
        plt.grid(True)
        plt.legend(loc='lower left')
        plt.savefig(filename)

    def accumulateFBeta(self):
        print('Accumulating F-beta evaluation results...')
        tic = time.time()
        if not self.evalImgs:
            print('Please run evaluate() first')
        p = self.params

        T           = len(p.iouThrs)
        A           = len(p.areaRng)
        K           = len(p.catIds) if p.useCats else 1
        tpCum = np.zeros((A, K, T, len(p.confThrs)))
        fpCum = np.zeros((A, K, T, len(p.confThrs)))
        fnCum = np.zeros((A, K, T, len(p.confThrs)))
        numGtCum = np.zeros((A, K), dtype=np.int32)

        # create dictionary for future indexing
        _pe = self._paramsEval
        catIds = _pe.catIds if _pe.useCats else [-1]
        setK = set(catIds)
        setA = set(map(tuple, _pe.areaRng))
        setI = set(_pe.imgIds)
        # get inds to evaluate
        k_list = [n for n, k in enumerate(p.catIds)  if k in setK]
        a_list = [n for n, a in enumerate(map(lambda x: tuple(x), p.areaRng)) if a in setA]
        i_list = [n for n, i in enumerate(p.imgIds)  if i in setI]
        I0 = len(_pe.imgIds)
        A0 = len(_pe.areaRng)
        # retrieve evalImgs at each category and area range
        for k, k0 in enumerate(k_list):
            Nk = k0*A0*I0
            for a, a0 in enumerate(a_list):
                Na = a0*I0
                evalImgs = [self.evalImgs[Nk + Na + i] for i in i_list]
                evalImgs = [e for e in evalImgs if not e is None]
                if len(evalImgs) == 0:
                    continue

                for imageDict in evalImgs:
                    numGtCum[a,k] += len(imageDict['gtIds']) - imageDict['gtIgnore'].astype(int).sum()

                    scoreMask = np.full((len(p.confThrs), T, len(imageDict['dtScores'])), False)
                    for idx, score in enumerate(imageDict['dtScores']):
                        score_idx = np.searchsorted(p.confThrs, score, side='right')
                        scoreMask[:score_idx, :, idx] = True

                    dtIgnore = imageDict['dtIgnore']
                    finalMask = scoreMask & ~dtIgnore
                    filteredArr = np.where(finalMask==True, imageDict['dtMatches'], -np.ones((10,1)))
                    filteredArr = np.swapaxes(filteredArr, 0, 1)

                    tpCum[a,k,:,:] += np.sum(filteredArr > 0, axis=2)
                    fpCum[a,k,:,:] += np.sum(filteredArr == 0, axis=2)
                fnCum = numGtCum[:, :, np.newaxis, np.newaxis] - tpCum

        self.evalFBeta = {
            'tp': tpCum,
            'fp': fpCum,
            'fn': fnCum,
            'numGt': numGtCum,
        }
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format(toc-tic))

    def _filterCum(self, iouThr, areaRng='all', classIdx=None, confThr=None):
        p = self.params

        tpCums = self.evalFBeta['tp'] # (A, K, T, len(p.confThrs))
        fpCums = self.evalFBeta['fp'] # (A, K, T, len(p.confThrs))
        fnCums = self.evalFBeta['fn'] # (A, K, T, len(p.confThrs))
        numGtCum = self.evalFBeta['numGt'] # (A, K)

        iouThrIdx = np.where(p.iouThrs == iouThr)[0].item()
        areaIdx = p.areaRngLbl.index(areaRng)

        tpCum = tpCums[areaIdx, :, iouThrIdx, :]
        fpCum = fpCums[areaIdx, :, iouThrIdx, :]
        fnCum = fnCums[areaIdx, :, iouThrIdx, :]
        numGtCum = numGtCum[areaIdx]
        numGtCum = numGtCum[:, np.newaxis]

        if classIdx is not None:
            classIdIdx = p.catIds.index(classIdx)
            numGtCum = numGtCum[classIdIdx]
            tpCum = tpCum[classIdIdx, np.newaxis]
            fpCum = fpCum[classIdIdx, np.newaxis]
            fnCum = fnCum[classIdIdx, np.newaxis]

        if confThr is not None:
            confThrIdx = np.where(p.confThrs == confThr)[0].item()
            tpCum = tpCum[:, confThrIdx, np.newaxis]
            fpCum = fpCum[:, confThrIdx, np.newaxis]
            fnCum = fnCum[:, confThrIdx, np.newaxis]

        return tpCum, fpCum, fnCum, numGtCum

    @staticmethod
    def _calculatePrecisionRecall(tpCum, fpCum, fnCum, numGtCum, average=None):
        if average == 'micro':
            tpCum = np.sum(tpCum, axis=0)
            fpCum = np.sum(fpCum, axis=0)
            fnCum = np.sum(fnCum, axis=0)
            numGtCum = np.sum(numGtCum, axis=0)

        precision = np.divide(tpCum, (tpCum + fpCum), out=np.full(tpCum.shape, np.nan, np.float), where=(tpCum + fpCum)>0)
        recall = np.divide(tpCum, numGtCum, out=np.full(tpCum.shape, np.nan, np.float), where=numGtCum>0)

        # take care of edge cases (https://github.com/dice-group/gerbil/wiki/Precision,-Recall-and-F1-measure)
        edge1 = np.equal(tpCum, 0) & np.equal(fpCum, 0) & np.equal(fnCum, 0) & np.greater(numGtCum, 0)
        precision[edge1] = recall[edge1] = 1
        edge2 = np.equal(tpCum, 0) & np.equal(fpCum, 0) & np.greater(fnCum, 0)
        precision[edge2] = 1

        if average == 'macro':
            precision = np.mean(precision, axis=0)
            recall = np.mean(recall, axis=0)
        elif average == 'weighted':
            precision = np.average(precision, axis=0, weights=numGtCum.squeeze(axis=1))
            recall = np.average(recall, axis=0, weights=numGtCum.squeeze(axis=1))

        return precision, recall

    @staticmethod
    def _calculateFBetaScore(beta, precision, recall, tpCum, fpCum, fnCum, numGtCum):
        precision = np.copy(precision)
        recall = np.copy(recall)

        # take care of edge cases (https://github.com/dice-group/gerbil/wiki/Precision,-Recall-and-F1-measure)
        edge1 = np.equal(tpCum, 0) & np.equal(fpCum, 0) & np.equal(fnCum, 0) & np.greater(numGtCum, 0)
        precision[edge1] = recall[edge1] = 1
        edge2 = np.equal(tpCum, 0) & np.equal(fpCum, 0) & np.greater(fnCum, 0)
        precision[edge2] = 1

        score = np.divide(
                    (1 + beta**2) * precision * recall,
                    (beta**2 * precision) + recall,
                    out=np.full(precision.shape, 0, np.float),
                    where=(precision + recall)!=0)
        return score

    def _getFBetaScore(self, beta=1, iouThr=0.5, areaRng='all', average='macro'):
        '''
        Calculate F-beta scores
        :param beta: F-beta score to calculate
        :param iouThr: IOU threshold
        :param areaRng: object area range (options: 'all', 'small', 'medium', 'large')
        :param average: averaging method (options: 'micro', 'macro', 'weighted')
        :return: (precision, recall, fscore, numGt)
        '''
        iStr = ' F{:<1} @[ IoU={:<4} | area={:>6s} | precision={:<5} | recall={:<5} ] = {:0.3f}'

        tpCum, fpCum, fnCum, numGtCum = self._filterCum(iouThr, areaRng, confThr=0)
        precisions, recalls = self._calculatePrecisionRecall(tpCum, fpCum, fnCum, numGtCum)
        fscores = self._calculateFBetaScore(beta, precisions, recalls, tpCum, fpCum, fnCum, numGtCum)

        if average == 'micro':
            avg_precision, avg_recall = self._calculatePrecisionRecall(tpCum, fpCum, fnCum, numGtCum, average='micro')
            avg_fscore = self._calculateFBetaScore(beta, avg_precision, avg_recall, sum(tpCum), sum(fpCum), sum(fnCum), sum(numGtCum))
            avg_precision = avg_precision.item()
            avg_recall = avg_recall.item()
            avg_fscore = avg_fscore.item()
        elif average == 'macro':
            avg_precision = np.nanmean(precisions)
            avg_recall = np.nanmean(recalls)
            avg_fscore = np.nanmean(fscores)
        elif average == 'weighted':
            avg_precision = 0
            masked_precision = np.ma.masked_array(precisions, mask=np.isnan(precisions))
            if np.all(np.isnan(precisions)):
                avg_precision = np.nan
            elif np.any(masked_precision):
                avg_precision = np.ma.average(masked_precision, weights=numGtCum)

            avg_recall = np.nan
            if not np.all(np.isnan(recalls)):
                masked_recall = np.ma.masked_array(recalls, mask=np.isnan(recalls))
                avg_recall = np.ma.average(masked_recall, weights=numGtCum)

            avg_fscore = np.nan
            if not np.all(np.isnan(fscores)):
                masked_fscore = np.ma.masked_array(fscores, mask=np.isnan(fscores))
                avg_fscore = np.ma.average(masked_fscore, weights=numGtCum)

        precisionStr = f'{avg_precision:0.3f}'
        recallStr = f'{avg_recall:0.3f}'
        print(iStr.format(beta, iouThr, areaRng, precisionStr, recallStr, avg_fscore))

        return avg_fscore

    def summarizeFBetaScores(self, average='macro'):
        '''
        Compute and display summary metrics for F-beta scores.
        :param average: averaging method (options: 'micro', 'macro', 'weighted')
        :return: None
        '''
        def _summarizeDets():
            stats = np.zeros((10,))
            stats[0] = self._getFBetaScore(beta=1, iouThr=.5, areaRng='all', average=average)
            stats[1] = self._getFBetaScore(beta=1, iouThr=.75, areaRng='all', average=average)
            stats[2] = self._getFBetaScore(beta=1, iouThr=.5, areaRng='small', average=average)
            stats[3] = self._getFBetaScore(beta=1, iouThr=.5, areaRng='medium', average=average)
            stats[4] = self._getFBetaScore(beta=1, iouThr=.5, areaRng='large', average=average)
            stats[5] = self._getFBetaScore(beta=2, iouThr=.5, areaRng='all', average=average)
            stats[6] = self._getFBetaScore(beta=2, iouThr=.75, areaRng='all', average=average)
            stats[7] = self._getFBetaScore(beta=2, iouThr=.5, areaRng='small', average=average)
            stats[8] = self._getFBetaScore(beta=2, iouThr=.5, areaRng='medium', average=average)
            stats[9] = self._getFBetaScore(beta=2, iouThr=.5, areaRng='large', average=average)
            return stats

        print(f'\nCalculating ({average}) F-beta scores...')
        if not self.evalFBeta:
            raise Exception('Please run accumulateFBeta() first')
        iouType = self.params.iouType
        if iouType == 'bbox':
            self.fstats = _summarizeDets()
            print()
        else:
            print('F-scores calculation only supported for bounding boxes.')

    def printReport(self, beta=1, iouThr=0.5, confThr=0):
        '''
        Build a text report showing the main metrics
        :param beta: F-beta score to calculate
        :param iouThr: IOU threshold
        :param confThr: confidence threshold for detections
        :return: None
        '''
        if not self.evalFBeta:
            raise Exception('Please run accumulateFBeta() first')

        p = self.params

        headers = ['precision', 'recall', f'f{beta}-score', 'support']
        average_options = ('micro', 'macro', 'weighted')
        catNms = p.catNms

        tpCum, fpCum, fnCum, numGtCum = self._filterCum(iouThr, confThr=confThr)
        precisions, recalls = self._calculatePrecisionRecall(tpCum, fpCum, fnCum, numGtCum)
        fscores = self._calculateFBetaScore(beta, precisions, recalls, tpCum, fpCum, fnCum, numGtCum)
        rows = zip(catNms, precisions.flatten(), recalls.flatten(), fscores.flatten(), numGtCum.flatten())
        numGtCum_sum = sum(numGtCum).item()

        longest_last_line_heading = 'weighted avg'
        name_width = max(len(cn) for cn in catNms)
        width = max(name_width, len(longest_last_line_heading))
        head_fmt = '{:>{width}s} ' + ' {:>9}' * len(headers)
        report = head_fmt.format('', *headers, width=width)
        report += '\n\n'
        row_fmt = '{:>{width}s} ' + ' {:>9.3f}' * 3 + ' {:>9}\n'
        for row in rows:
            report += row_fmt.format(*row, width=width)
        report += '\n'

        for average in average_options:
            line_heading = f'{average} avg'
            if average == 'micro':
                avg_precision, avg_recall = self._calculatePrecisionRecall(tpCum, fpCum, fnCum, numGtCum, average='micro')
                avg_fscore = self._calculateFBetaScore(beta, avg_precision, avg_recall, sum(tpCum), sum(fpCum), sum(fnCum), sum(numGtCum))
                avg_precision = avg_precision.item()
                avg_recall = avg_recall.item()
                avg_fscore = avg_fscore.item()
            elif average == 'macro':
                avg_precision = np.mean(precisions)
                avg_recall = np.mean(recalls)
                avg_fscore = np.mean(fscores)
            elif average == 'weighted':
                avg_precision = sum([p * s for p, s in zip(precisions, numGtCum)]).item() / numGtCum_sum
                avg_recall = sum([r * s for r, s in zip(recalls, numGtCum)]).item() / numGtCum_sum
                avg_fscore = sum([f * s for f, s in zip(fscores, numGtCum)]).item() / numGtCum_sum
            else:
                raise Exception(f'{average} average is not supported')

            avg = [avg_precision, avg_recall, avg_fscore, numGtCum_sum]
            report += row_fmt.format(line_heading, *avg, width=width)

        print(f'F{beta} for IOU threshold {iouThr} and confidence threshold {confThr}:')
        print(report)

    def plotFBetaCurve(self, filename, betas=[1], iouThr=0.5, areaRng='all', classIdx=None, average='macro'):
        '''
        Plot F-beta curves
        :param filename: output filename
        :param betas: F-beta scores to plot
        :param iouThr: IOU threshold
        :param areaRng: object area range (options: 'all', 'small', 'medium', 'large')
        :param classIdx: to plot for a specific class
        :param average: averaging method (options: 'micro', 'macro', 'weighted')
        :return: None
        '''
        if not self.evalFBeta:
            raise Exception('Please run accumulateFBeta() first')

        p = self.params

        if classIdx is not None:
            className = p.catIdsToCatNms[classIdx]

        tpCum, fpCum, fnCum, numGtCum = self._filterCum(iouThr, areaRng, classIdx)
        precision, recall = self._calculatePrecisionRecall(tpCum, fpCum, fnCum, numGtCum, average)

        plt.figure()
        plt.plot(p.confThrs, precision, label='precision')
        plt.plot(p.confThrs, recall, label='recall')

        for beta in betas:
            score = np.divide(
                        (1 + beta**2) * precision * recall,
                        (beta**2 * precision) + recall,
                        out=np.full(precision.shape, 0, np.float),
                        where=(precision + recall)!=0)

            maxIdx = np.argmax(score)
            plt.plot(p.confThrs, score, label=f'F{beta}: {score[maxIdx]:0.3f} at {p.confThrs[maxIdx]:0.2f}')
            if classIdx is None:
                print(f'Best {average} F{beta} for iouThr {iouThr} is {score[maxIdx]:0.3f} at confThr {p.confThrs[maxIdx]:0.2f}: precision {precision[maxIdx]:0.3f}, recall {recall[maxIdx]:0.3f}')
            else:
                print(f'Best F{beta} for class {className} is {score[maxIdx]:0.3f} at confThr {p.confThrs[maxIdx]:0.2f}: precision {precision[maxIdx]:0.3f}, recall {recall[maxIdx]:0.3f}')
        print()

        if classIdx is None:
            title = f'{average} Fscores for iouThr={iouThr}'
        else:
            title = f'Fscores for class={className}, iouThr={iouThr}'
        plt.title(title)
        plt.xlabel('confidence threshold')
        plt.ylabel('score')
        plt.xlim(0, 1.0)
        plt.ylim(0, 1.01)
        plt.grid(True)
        plt.legend(loc='lower left')
        plt.savefig(filename)

    def plotPRCurve(self, filename, areaRng='all', classIdx=None, average='macro'):
        '''
        Plot PR curves
        :param filename: output filename
        :param classIdx: to plot for a specific class
        :param average: averaging method (options: 'micro', 'macro', 'weighted')
        :return: None
        '''
        if not self.evalFBeta:
            raise Exception('Please run accumulateFBeta() first')

        p = self.params

        if classIdx is not None:
            className = p.catIdsToCatNms[classIdx]

        tpCums = self.evalFBeta['tp']
        fpCums = self.evalFBeta['fp']
        fnCums = self.evalFBeta['fn']
        numGtCum = self.evalFBeta['numGt']

        plt.figure()
        if classIdx is None:
            title = f'P-R curve'
        else:
            title = f'P-R curve for class={className}'

        for iouThr in p.iouThrs:
            tpCum, fpCum, fnCum, numGtCum = self._filterCum(iouThr, areaRng, classIdx)
            precision, recall = self._calculatePrecisionRecall(tpCum, fpCum, fnCum, numGtCum, average=average)
            precision = np.insert(precision, 0, [0, 0])
            recall = np.insert(recall, 0, [1, recall[0]+0.01])
            plt.plot(recall, precision, label=f'iou={iouThr}')

        plt.title(title)
        plt.xlabel('recall')
        plt.ylabel('precision')
        plt.xlim(0, 1.0)
        plt.ylim(0, 1.01)
        plt.grid(True)
        plt.legend(loc='lower left')
        plt.savefig(filename)

    def __str__(self):
        self.summarize()

class Params:
    '''
    Params for coco evaluation api
    '''
    def setDetParams(self):
        self.imgIds = []
        self.catIds = []
        # np.arange causes trouble.  the data point on arange is slightly larger than the true value
        self.iouThrs = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        self.iouThrs = np.around(self.iouThrs, 2)
        self.recThrs = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)
        self.confThrs = np.linspace(0, 1, int(np.round((1 - 0) / .01)) + 1, endpoint=True)
        self.maxDets = [1, 10, 100, 1000]
        self.areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, 32 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
        self.areaRngLbl = ['all', 'small', 'medium', 'large']
        self.useCats = 1

    def setKpParams(self):
        self.imgIds = []
        self.catIds = []
        # np.arange causes trouble.  the data point on arange is slightly larger than the true value
        self.iouThrs = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        self.recThrs = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)
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
