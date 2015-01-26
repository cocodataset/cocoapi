%% Demo for the CocoApi (see CocoApi.m)

%% initialize COCO api for instance annotations
dataDir='../'; dataType='val2014';
annFile=sprintf('%s/annotations/instances_%s.json',dataDir,dataType);
if(~exist('coco','var')), coco=CocoApi(annFile); end

%% display COCO categories and supercategories
cats = coco.loadCats(coco.getCatIds());
nms={cats.name}; fprintf('COCO categories: ');
fprintf('%s, ',nms{:}); fprintf('\n');
nms=unique({cats.supercategory}); fprintf('COCO supercategories: ');
fprintf('%s, ',nms{:}); fprintf('\n');

%% get all images containing given categories, select one at random
catIds = coco.getCatIds('catNms',{'person','dog','skateboard'});
imgIds = coco.getImgIds('catIds',catIds );
imgId = imgIds(randi(length(imgIds)));

%% load and display image
img = coco.loadImgs(imgId);
I = imread(sprintf('%s/images/%s/%s',dataDir,dataType,img.file_name));
figure(1); imagesc(I); axis('image'); set(gca,'XTick',[],'YTick',[])

%% load and display instance annotations
annIds = coco.getAnnIds('imgIds',imgId,'catIds',catIds,'iscrowd',[]);
anns = coco.loadAnns(annIds); coco.showAnns(anns);

%% initialize COCO api for caption annotations
annFile=sprintf('%s/annotations/captions_%s.json',dataDir,dataType);
if(~exist(annFile,'file')), return; end
if(~exist('caps','var')), caps=CocoApi(annFile); end

%% load and display caption annotations
annIds = caps.getAnnIds('imgIds',imgId);
anns = caps.loadAnns(annIds); caps.showAnns(anns);
