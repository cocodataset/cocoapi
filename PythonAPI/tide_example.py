from tidecv import TIDE, datasets

gt = datasets.COCO('test.json')

bbox_results = datasets.COCOResult('filtered_coco.json')

tide = TIDE()
tide.evaluate(gt, bbox_results, mode=TIDE.BOX)

print (tide.summarize())
print (tide.plot())