%% initialize coco data structure
coco=cocoLoad('data/instances_val2014.json','data/val2014');

%% get all images containging first 4 categories
imgIds = coco.getImgIds( 'catIds',1:4 ); length(imgIds)
for i=1:4, disp(coco.categories(i).name); end

%% get annotations for first such image
annIds = coco.getAnnIds( 'imgIds',imgIds(1) );

%% load image and annotations
I = coco.loadImg( imgIds(1) );
anns = coco.loadAnns( annIds );
figure(1); im(I);
