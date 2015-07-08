__author__ = 'tsungyi'

import numpy as np
import datetime
from collections import defaultdict
import mask

class COCOeval:
    # Interface for evaluating detection on the Microsoft COCO dataset.
    #
    # The usage for CocoEval is as follows:
    #  cocoGt=..., cocoDt=...       # load dataset and results
    #  E = CocoEval(cocoGt,cocoDt); # initialize CocoEval object
    #  E.params.recThrs = ...;      # set parameters as desired
    #  E.evaluate();                # run per image evaluation
    #  E.accumulate();              # accumulate per image results
    # For example usage see evalDemo.m and http://mscoco.org/.
    #
    # The evaluation parameters are as follows (defaults in brackets):
    #  imgIds     - [all] N img ids to use for evaluation
    #  catIds     - [all] K cat ids to use for evaluation
    #  iouThrs    - [1/T:1/T:1] T=20 IoU thresholds for evaluation
    #  recThrs    - [1/R:1/R:1] R=1000 recall thresholds for evaluation
    #  maxDets    - [100] max number of allowed detections per image
    #  areaRng    - [0 1e10] object area range for evaluation
    #  useSegm    - [1] if true evaluate against ground-truth segments
    #  useCats    - [1] if true use category labels for evaluation
    # Note: if useSegm=0 the evaluation is run on bounding boxes.
    # Note: if useCats=0 category labels are ignored as in proposal scoring.
    # Note: multiple areaRngs [Ax2] and maxDets [Mx1] can be specified.
    #
    # evaluate(): evaluates detections on every image and every category and
    # concats the results into the struct array "evalImgs" with fields:
    #  imgId      - results for the img with the given id
    #  catId      - results for the cat with the given id
    #  areaRng    - results for objects in the given areaRng
    #  maxDets    - results given the specified max number of detections
    #  dtIds      - [1xD] id for each of the D detections (dt)
    #  gtIds      - [1xG] id for each of the G ground truths (gt)
    #  dtMatches  - [TxD] matching gt id at each IoU or 0
    #  gtMatches  - [TxG] matching dt id at each IoU or 0
    #  dtScores   - [1xD] confidence of each dt
    #  gtIgnore   - [1xG] ignore flag for each gt
    #  dtIgnore   - [TxD] ignore flag for each dt at each IoU
    #  ious       - [DxG] iou between every dt and gt
    #
    # accumulate(): accumulates the per-image, per-category evaluation
    # results in "evalImgs" into the struct "eval" with fields:
    #  params     - parameters used for evaluation
    #  date       - date evaluation was performed
    #  counts     - [T,R,K,A,M] parameter dimensions (see above)
    #  precision  - [TxRxKxAxM] precision for every evaluation setting
    #  recall     - [TxKxAxM] max recall for every evaluation setting
    #  ap         - average across all non-negative precision values
    #  ar         - average across all non-negative recall values
    # Note: precision and recall==-1 for settings with no gt objects.
    #
    # See also coco, mask, pycocoDemo, pycocoEvalDemo
    #
    # Microsoft COCO Toolbox.      Version 1.0
    # Data, paper, and tutorials available at:  http://mscoco.org/
    # Code written by Piotr Dollar and Tsung-Yi Lin, 2015.
    # Licensed under the Simplified BSD License [see coco/license.txt]
    def __init__(self, cocoGt=None, cocoDt=None):
        '''
        Initialize CocoEval using coco APIs for gt and dt
        :param cocoGt: coco object with ground truth annotations
        :param cocoDt: coco object with detection results
        :return: None
        '''
        self.cocoGt   = cocoGt              # ground truth COCO API
        self.cocoDt   = cocoDt              # detections COCO API
        self.params   = {}                  # evaluation parameters
        self.evalImgs = defaultdict(list)   # per-image per-category evaluation results
        self.eval     = {}                  # accumulated evaluation results
        self._gts = defaultdict(list)       # gt for evaluation
        self._dts = defaultdict(list)       # dt for evaluation
        self.params = Params()
        if not cocoGt is None:
            self.params.imgIds = sorted(cocoGt.getImgIds())
            self.params.catIds = sorted(cocoGt.getCatIds())


    def _prepare(self):
        '''
        Prepare ._gts and ._dts for evaluation based on params
        :return: None
        '''
        #
        def _toMask(objs, coco):
            # modify segmentation by reference
            for obj in objs:
                t = coco.imgs[obj['image_id']]
                if type(obj['segmentation']) == list:
                    if type(obj['segmentation'][0]) == dict:
                        print 'debug'
                    obj['segmentation'] = mask.frPyObjects(obj['segmentation'],t['height'],t['width'])
                    if len(obj['segmentation']) == 1:
                        obj['segmentation'] = obj['segmentation'][0]
                    else:
                        # an object can have multiple polygon regions
                        # merge them into one RLE mask
                        obj['segmentation'] = mask.merge(obj['segmentation'])
                elif type(obj['segmentation']) == dict and type(obj['segmentation']['counts']) == list:
                    obj['segmentation'] = mask.frPyObjects([obj['segmentation']],t['height'],t['width'])[0]
                elif type(obj['segmentation']) == dict and \
                     type(obj['segmentation']['counts'] == unicode or type(obj['segmentation']['counts']) == str):
                    pass
                else:
                    raise Exception('segmentation format not supported.')
        p = self.params
        if p.useCats:
            gts=self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
            dts=self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
        else:
            gts=self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds))
            dts=self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds))

        if p.useSegm:
            _toMask(gts, self.cocoGt)
            _toMask(dts, self.cocoDt)
        for gt in gts:
            self._gts[gt['image_id'], gt['category_id']].append(gt)
        for dt in dts:
            self._dts[dt['image_id'], dt['category_id']].append(dt)

    def evaluate(self):
        '''
        Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
        :return: None
        '''
        tic = datetime.datetime.utcnow()
        print 'Running per image evaluation...      '
        p = self.params
        p.imgIds = list(np.unique(p.imgIds))
        if p.useCats:
            p.catIds = list(np.unique(p.catIds))
        self.params=p
        self._prepare()
        # loop through images, area range, max detection number
        for imgId in p.imgIds:
            for j, areaRng in enumerate(p.areaRng): # areaRng: [min, max] array
                for k, maxDet in enumerate(p.maxDets):  # maxDet: int
                    if p.useCats:
                        for catId in p.catIds:
                            evalImg = self.evaluateImg(imgId, catId, areaRng, maxDet)
                            if not evalImg is None:
                                self.evalImgs[imgId,catId,tuple(areaRng),maxDet] = evalImg
                    else:
                        catId = -1
                        evalImg = self.evaluateImg(imgId, catId, areaRng, maxDet)
                        if not evalImg is None:
                            self.evalImgs[imgId,catId,tuple(areaRng),maxDet] = evalImg

        toc = datetime.datetime.utcnow()
        print 'DONE (t=%0.2fs).'%( (toc-tic).total_seconds() )

    def evaluateImg(self, imgId, catId, aRng, maxDet):
        '''
        perform evaluation for single category and image
        :return: dict (single image results)
        '''
        #
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
            if 'ignore' not in g:
                g['ignore'] = 0
            if g['iscrowd'] == 1 or g['ignore'] or (g['area']<aRng[0] or g['area']>aRng[1]):
                g['_ignore'] = 1
            else:
                g['_ignore'] = 0

        # sort dt highest score first, sort gt ignore last
        gt = sorted(gt, key=lambda x: x['_ignore'])
        dt = sorted(dt, key=lambda x: -x['score'])
        if len(dt) > maxDet:
            dt=dt[0:maxDet]

        if p.useSegm:
            g = [g['segmentation'] for g in gt]
            d = [d['segmentation'] for d in dt]
        else:
            g = [g['bbox'] for g in gt]
            d = [d['bbox'] for d in dt]

        # compute iou between each dt and gt region
        iscrowd = [int(o['iscrowd']) for o in gt]
        ious = mask.iou(d,g,iscrowd)
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
                        iou_tmp = ious[dind,gind]
                        if ious[dind,gind] < iou:
                            continue
                        # match successful and best so far, store appropriately
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
        return {
                'imgId':        imgId,
                'catId':        catId,
                'dtIds':        [d['id'] for d in dt],
                'gtIds':        [g['id'] for g in gt],
                'dtMatches':    dtm,
                'gtMatches':    gtm,
                'dtScores':     [d['score'] for d in dt],
                'gtIgnore':     gtIg,
                'dtIgnore':     dtIg,
                'ious':         ious,
                'areaRng':      aRng,
                'maxDet':       maxDet,
            }

    def accumulate(self, p = None):
        '''
        Accumulate per image evaluation results and store the result in self.eval
        :param p: input params for evaluation
        :return: None
        '''
        print 'Accumulating evaluation results...   '
        tic = datetime.datetime.utcnow()
        if not self.evalImgs:
            print 'Please run evaluate() first'
        # allows input customized parameters
        if p is None:
            p = self.params
        T           = len(p.iouThrs)
        R           = len(p.recThrs)
        K           = len(p.catIds) if p.useCats else 1
        A           = len(p.areaRng)
        M           = len(p.maxDets)
        precision   = -np.ones((T,R,K,A,M)) # -1 for the precision of absent categories
        recall      = -np.ones((T,K,A,M))

        # create dictionary for future indexing
        catToK = defaultdict(lambda:-1)
        if p.useCats:
            for k, catId in enumerate(p.catIds):
                catToK[catId] = k
        else:
            catToK[-1] = 0
        areaRngToM = defaultdict(lambda:-1)
        for m, areaRng in enumerate(p.areaRng):
            areaRngToM[tuple(areaRng)] = m
        maxDetsToA = defaultdict(lambda:-1)
        for a, maxDets in enumerate(p.maxDets):
            maxDetsToA[maxDets] = a
        indsToEvalImg = defaultdict(list)
        for key, e in self.evalImgs.items():
            k = catToK[key[1]]; a = areaRngToM[tuple(key[2])]; m = maxDetsToA[key[3]]
            if  k >-1 and m >-1 and a>-1:
                indsToEvalImg[k, a, m].append(e)

        # retrieve E at each category, area range, and max number of detections
        for key, E in indsToEvalImg.items():
            E = sorted(E, key=lambda x: x['imgId'])
            k = key[0]; a = key[1]; m = key[2]
            dtScores = np.hstack( [e['dtScores']  for e in E] )

            # different sorting method generates slightly different results.
            # mergesort is used to be consistent as Matlab implementation.
            inds = np.argsort(-dtScores, kind='mergesort')

            dtm  = np.hstack([e['dtMatches'] for e in E])[:,inds]
            dtIg = np.hstack([e['dtIgnore']  for e in E])[:,inds]
            gtIg = np.hstack([e['gtIgnore']  for e in E])
            npig = len([ig for ig in gtIg if ig == 0])

            if npig == 0:
                continue
            tps = np.logical_and(               dtm,  np.logical_not(dtIg) )
            fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg) )

            for t in range(T):
                tp = np.array(np.cumsum(tps[t,:]), dtype=np.float)
                fp = np.array(np.cumsum(fps[t,:]), dtype=np.float)

                nd = len(tp)
                rc = tp / npig
                pr = tp / (fp+tp+np.spacing(1))
                q  = np.zeros((R,), dtype=np.float)

                if nd:
                    recall[t,k,a,m] = rc[-1]
                else:
                    recall[t,k,a,m] = 0
                for i in range(nd-1, 0, -1):
                    if pr[i] > pr[i-1]:
                        pr[i-1] = pr[i]
                i = 0
                r = 0

                while r<R and i<nd:
                    if rc[i] < p.recThrs[r]:
                        i += 1
                    else:
                        q[r] = pr[i]
                        r += 1
                precision[t,:,k,a,m] = q
        self.eval = {
            'params': p,
            'counts': [T, R, K, A, M],
            'date': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'precision': precision,
            'recall':   recall,
            'ap': np.mean(precision[precision>-1]),
            'ar': np.mean(recall[recall>-1]),
        }
        toc = datetime.datetime.utcnow()
        print 'DONE (t=%0.2fs).'%( (toc-tic).total_seconds() )

    def __str__(self):
        if self.params.useSegm:
            annType = 'segm'
        else:
            annType = 'bbox'
        return 'AP of %s results is %.4f\n'%(annType, self.eval['ap'])

class Params:
    '''
    Params for coco evaluation api
    '''
    def __init__(self):
        self.imgIds = []
        self.catIds = []
        # np.arange causes trouble.  the data point on arange is slightly larger than the true value
        # self.iouThrs = np.arange(.05, 1.001, .05)
        # self.recThrs = np.arange(.001,1.001, .001)
        self.iouThrs = np.linspace(.05, 1.000, np.round((1.000-0.05)/.05) +1, endpoint=True)
        self.recThrs = np.linspace(.001,1.000, np.round((1.000-0.001)/.001)+1, endpoint=True)
        self.maxDets = [100]
        self.areaRng = [[0, 1e10]]
        self.useSegm = 0
        self.useCats = 1