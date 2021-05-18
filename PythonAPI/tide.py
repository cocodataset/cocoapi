from tidecv import TIDE, datasets

gt = datasets.COCO('test_vehicle.json')

bbox_results = datasets.COCOResult('ans_vehicle.json')

tide = TIDE()
tide.evaluate(gt, bbox_results, mode=TIDE.OBB)

print (tide.summarize())
print (tide.plot())
