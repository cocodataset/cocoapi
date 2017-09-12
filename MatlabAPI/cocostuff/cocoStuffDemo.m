% A preview script that shows how to use the COCO API with stuff annotations.
%
% It lists the categories and super-categories and shows the
% annotations of an example image.
%
% Microsoft COCO Toolbox.      version 2.0
% Data, paper, and tutorials available at:  http://mscoco.org/
% Code written by Piotr Dollar and Tsung-Yi Lin, 2015.
% Licensed under the Simplified BSD License [see coco/license.txt]

% Define path
dataDir= '../..';
dataType = 'train2017';
annFile = sprintf('%s/annotations/stuff_%s.json', dataDir, dataType);

% Initialize COCO ground-truth API
cocoGt = CocoApi(annFile);

% Display COCO stuff categories and supercategories
categories = cocoGt.loadCats(cocoGt.getCatIds());
categoryNames = {categories.name}';
fprintf('COCO Stuff leaf categories: %s\n', strjoin(categoryNames));
    
superCategoryNames = unique({categories.supercategory}');
fprintf('COCO Stuff super categories: %s\n', strjoin(superCategoryNames));

% Load info for a random image
imgIds = cocoGt.getImgIds();
imgId = imgIds(randi(numel(imgIds)));
fprintf('Processing image %d\n', imgId);
img = cocoGt.loadImgs(imgId);

% Load and display image
I = imread(img.coco_url);
figure();
subplot(1, 2, 1);
imshow(I);

% Load and display stuff annotations
annIds = cocoGt.getAnnIds('imgIds', img.id);
anns = cocoGt.loadAnns(annIds);
subplot(1, 2, 2);
imshow(I);
cocoGt.showAnns(anns);