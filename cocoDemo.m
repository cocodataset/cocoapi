%% Demo for the CocoApi (see CocoApi.m)

%% initialize coco api for instance annotations
imgDir = 'data/val2014';
annFile = 'data/instances_val2014.json';
if(~exist('coco','var'))
  coco = CocoApi( imgDir, annFile ); end

%% get all images containging given categories, select one at random
catIds = coco.getCatIds( {'person','dog','skateboard'} );
imgIds = coco.getImgIds( 'catIds',catIds );
imgId = imgIds( randi(length(imgIds)) );

%% load and display image
I = coco.loadImg(imgId);
figure(1); im(I,[],0);

%% load and display instance annotations
annIds = coco.getAnnIds( 'imgIds',imgId, 'catIds',catIds );
anns = coco.loadAnns( annIds );
coco.showAnns( anns );

%% initialize coco api for caption annotations
annFile = 'data/sentences_val2014.json';
if(~exist('cocoCap','var'))
  cocoCap = CocoApi( imgDir, annFile ); end

%% load and display caption annotations
annIds = cocoCap.getAnnIds( 'imgIds',imgId );
anns = cocoCap.loadAnns( annIds );
cocoCap.showAnns( anns );
