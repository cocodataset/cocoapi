import argparse
from pathlib import Path
from tidecv import TIDE, datasets

parser = argparse.ArgumentParser()
parser.add_argument('GTfile')
parser.add_argument('DTfile')
args = parser.parse_args()
annFile = Path(args.GTfile)
resFile = Path(args.DTfile)
assert annFile.is_file()
assert resFile.is_file()

gt = datasets.COCO(annFile)
bbox_results = datasets.COCOResult(resFile)

tide = TIDE()
tide.evaluate(gt, bbox_results, mode=TIDE.OBB)

print (tide.summarize())
print (tide.plot())
