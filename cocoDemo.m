%% Demo for the CocoApi (see CocoApi.m)

%% initialize COCO api for instance annotations
imgDir = 'data/val2014';
annFile = 'data/instances_val2014.json';
if(~exist('coco','var'))
  coco = CocoApi( imgDir, annFile ); end

%% display COCO categories and supercategories
cats = coco.loadCats( coco.getCatIds() );
nms={cats.name}; fprintf('COCO categories: ');
fprintf('%s ',nms{:}); fprintf('\n');
nms=unique({cats.supercategory}); fprintf('COCO supercategories: ');
fprintf('%s ',nms{:}); fprintf('\n');

%% get all images containging given categories, select one at random
catIds = coco.getCatIds( 'catNms', {'person','dog','skateboard'} );
imgIds = coco.getImgIds( 'catIds',catIds );
imgId = imgIds( randi(length(imgIds)) );

%% load and display image
imgs = coco.loadImgs( imgId, true );
figure(1); imagesc( imgs(1).image );
axis('image'); set(gca,'XTick',[],'YTick',[])

%% load and display instance annotations
annIds = coco.getAnnIds( 'imgIds',imgId, 'catIds',catIds, 'iscrowd',0 );
anns = coco.loadAnns( annIds );
coco.showAnns( anns );

%% initialize COCO api for caption annotations
annFile = 'data/captions_val2014.json';
if(~exist('cocoCap','var'))
  cocoCap = CocoApi( imgDir, annFile ); end

%% load and display caption annotations
annIds = cocoCap.getAnnIds( 'imgIds',imgId );
anns = cocoCap.loadAnns( annIds );
cocoCap.showAnns( anns );
