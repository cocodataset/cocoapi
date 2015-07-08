__author__ = 'tsungyi'

import os
import sys
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import json
import mask

##Demo demonstrating the algorithm result formats for COCO
##select results type for demo (either bbox or segm)
annType = ['segm','bbox']
annType = annType[0]      #specify type here
print 'Running demo for *%s* results.\n\n'%annType

##initialize COCO ground truth api
dataDir='../'
dataType='val2014'
annFile = '%s/annotations/instances_%s.json'%(dataDir,dataType)
cocoGt=COCO(annFile)

##initialize COCO detections api
resFile='%s/results/instances_%s_fake%s100_results.json'
resFile = resFile%(dataDir, dataType, annType)
cocoDt=cocoGt.loadRes(resFile)

##run Coco evaluation code (see CocoEval.m)
cocoEval = COCOeval(cocoGt,cocoDt)
imgIds = sorted(cocoGt.getImgIds())
imgIds = imgIds[0:100]
cocoEval.params.imgIds  = imgIds
cocoEval.params.useSegm = (annType == 'segm')
cocoEval.params.useCats = True
cocoEval.evaluate()
cocoEval.accumulate()
print cocoEval