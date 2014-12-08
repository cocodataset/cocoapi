%% initialize coco data structure
coco=cocoApi('initialize','data/instances_val2014.json','data/val2014');

%% get all images containging first 4 categories
catIds = cocoApi('getCatIds',coco,{'person','dog','skateboard'});
imgIds = cocoApi('getImgIds',coco','catIds',catIds); length(imgIds)

%% get annotations for first such image
annIds = cocoApi('getAnnIds',coco,'imgIds',imgIds(1),'catIds',catIds);

%% load image and annotations
I = cocoApi('loadImg',coco,imgIds(1) );
anns = cocoApi('loadAnns',coco,annIds );
figure(1); im(I);
