%% initialize coco data structure
coco=cocoApi('initialize','data/instances_val2014.json','data/val2014');

%% get all images containging first 4 categories
catIds = cocoApi('getCatIds',{'person','dog','skateboard'});
imgIds = cocoApi('getImgIds','catIds',catIds); length(imgIds)

%% get annotations for first such image
annIds = cocoApi('getAnnIds','imgIds',imgIds(1),'catIds',catIds);

%% load image and annotations
I = cocoApi('loadImg',imgIds(1) );
anns = cocoApi('loadAnns',annIds );
figure(1); im(I);
