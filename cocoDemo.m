%% initialize coco data structure
coco=cocoApi('initialize','data/instances_val2014.json','data/val2014');

%% get all images containging first 4 categories
catIds = coco.getCatIds({'person','dog','skateboard'});
imgIds = coco.getImgIds('catIds',catIds); length(imgIds)

%% get annotations for first such image
annIds = coco.getAnnIds('imgIds',imgIds(1),'catIds',catIds);

%% load image and annotations
I = coco.loadImg( imgIds(1) );
anns = coco.loadAnns( annIds );
figure(1); im(I);
