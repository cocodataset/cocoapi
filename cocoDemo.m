%% Demo for the CocoApi (see CocoApi.m)

%% initialize coco data structure
coco = CocoApi('data/instances_val2014.json','data/val2014');

%% get all images containging first 4 categories
catIds = coco.getCatIds({'person','dog','skateboard'});
imgIds = coco.getImgIds('catIds',catIds); length(imgIds)

%% get annotations for one such image
annIds = coco.getAnnIds('imgIds',imgIds(2),'catIds',catIds);

%% load image and annotations
I = coco.loadImg( imgIds(2) );
anns = coco.loadAnns( annIds );

%% display annotation
figure(1); coco.showAnns( anns );
