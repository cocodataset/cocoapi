import argparse
import numpy as np
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from coco_combinator import COCO_Combinator

'''
Q: Any way we could combine multiple jsons into one?
A: Based on experience, pain in the ass, but doable

1. ids and image_ids
2. category remap if necessary
3. if category is remapped, annotations need remapping too
'''

def eval(gtFile, dtFile):
    GTpath = Path(gtFile)
    DTpath = Path(dtFile)

    GTpath = GTpath if GTpath.is_file() else COCO_Combinator(gtFile, 'gtMerge.json').merge()
    DTpath = DTpath if DTpath.is_file() else COCO_Combinator(dtFile, 'dtMerge.json').merge()
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
