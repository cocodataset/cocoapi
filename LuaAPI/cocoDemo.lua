-- Demo for the CocoApi (see CocoApi.lua)
coco = require 'coco'
image = require 'image'

-- initialize COCO api for instance annotations
dataDir, dataType = '../', 'val2014'
annFile = dataDir..'/annotations/instances_'..dataType..'.json'
if not cocoApi then cocoApi=coco.CocoApi(annFile) end

-- get all image ids, select one at random
imgIds = cocoApi:getImgIds()
imgId = imgIds[torch.random(imgIds:numel())]

-- load image
img = cocoApi:loadImgs(imgId)[1]
I = image.load(dataDir..'/images/'..dataType..'/'..img.file_name,3)

-- load and display instance annotations
annIds = cocoApi:getAnnIds({imgId=imgId})
anns = cocoApi:loadAnns(annIds)
J = cocoApi:showAnns(I,anns)
image.save('RES_'..img.file_name,J:double())
