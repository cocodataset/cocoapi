# COCO API - http://cocodataset.org/

COCO is a large image dataset designed for object detection, segmentation, person keypoints detection, stuff segmentation, and caption generation. This package provides Matlab, Python, and Lua APIs that assists in loading, parsing, and visualizing the annotations in COCO. Please visit http://cocodataset.org/ for more information on COCO, including for the data, paper, and tutorials. The exact format of the annotations is also described on the COCO website. The Matlab and Python APIs are complete, the Lua API provides only basic functionality.

In addition to this API, please download both the COCO images and annotations in order to run the demos and use the API. Both are available on the project website.
- Please download, unzip, and place the images in: coco/images/
- Please download and place the annotations in: coco/annotations/
For substantially more details on the API please see http://cocodataset.org/#download.

After downloading the images and annotations, run the Matlab, Python, or Lua demos for example usage.

## Installation

- For Matlab, add coco/MatlabApi to the Matlab path (OSX/Linux binaries provided)
- For Python, run "make" under coco/PythonAPI
- For Lua, run “luarocks make LuaAPI/rocks/coco-scm-1.rockspec” under coco/

```
pip3 install 'git+https://github.com/yhsmiley/cocoapi.git#subdirectory=PythonAPI'
```

## Usage

```
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
cocoGt = COCO(true_path)  # initialize COCO ground truth api
cocoDt = cocoGt.loadRes(pred_path)  # initialize COCO prediction api
cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')  # initialize COCO evaluation api
cocoEval.evaluate()
```

To get F-beta scores of detections:
```
cocoEval.summarizeFBetaScores()
cocoEval.printReport(beta=1, iouThr=0.5, confThr=0)
cocoEval.plotFBetaCurve(filename, betas=[1], iouThr=0.5)
```

For original COCO-style evaluation:
```
cocoEval.accumulate()
cocoEval.summarize()
mapAll, map50 = cocoEval.stats[:2]

cocoEval.plotCocoPRCurve(filename, classIdx=None)
```
