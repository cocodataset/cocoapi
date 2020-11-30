import argparse
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

parser = argparse.ArgumentParser()
parser.add_argument('GTfile')
parser.add_argument('DTfile')
args = parser.parse_args()

annFile = Path(args.GTfile)
resFile = Path(args.DTfile)
assert annFile.is_file()
assert resFile.is_file()

annType = ['segm','bbox','keypoints', 'obb']
annType = 'obb'      # specify type here
print ('Running demo for *%s* results.'%(annType))

#initialize COCO ground truth api
cocoGt=COCO(annFile)

#initialize COCO detections api
cocoDt=cocoGt.loadRes(str(resFile))

imgIds=sorted(cocoGt.getImgIds())
imgIds=imgIds

# running evaluation
cocoEval = COCOeval(cocoGt,cocoDt,annType)
cocoEval.params.imgIds  = imgIds
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()