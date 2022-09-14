import argparse
import numpy as np
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

'''
Q: Any way we could combine multiple jsons into one?
A: Based on experience, pain in the ass, but doable
'''

def eval(gtFile, dtFile):
    GTpath = Path(gtFile)
    DTpath = Path(dtFile)
    assert GTpath.is_file()
    assert DTpath.is_file()

    # coco = COCO()

    cocoGt=COCO(str(GTpath))
    cocoDt=cocoGt.loadRes(str(DTpath))

    # print(cocoGt.getImgIds())
    # imgIds=sorted(cocoGt.getImgIds())
    # imgIds=imgIds[0:100]
    # imgId = imgIds[np.random.randint(100)]

    cocoEval = COCOeval(cocoGt,cocoDt,'bbox')
    # cocoEval.params.imgIds  = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.accumulateText()
    cocoEval.summarize()
