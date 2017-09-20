% Converts COCO segmentation .json files (GT or results) to one .png file per image.
%
% This script can be used for visualization of ground-truth and result files.
% Furthermore it can convert the ground-truth annotations to a more easily
% accessible .png format that is supported by many semantic segmentation methods.
%
% Note: To convert a result file to .png, we need to have both a valid GT file
% and the result file and set isAnnotation=False.
%
% The .png images are stored as indexed images, which means they contain both the
% segmentation map, as well as a color palette for visualization.
%
% Microsoft COCO Toolbox.      version 2.0
% Data, paper, and tutorials available at:  http://mscoco.org/
% Code written by Piotr Dollar and Tsung-Yi Lin, 2015.
% Licensed under the Simplified BSD License [see coco/license.txt]

dataDir = '../..';
dataTypeAnn = 'train2017';
dataTypeRes = 'examples';
pngFolderName = 'export_png';
isAnnotation = true;
exportImageLimit = 10;

% Define paths
annPath = sprintf('%s/annotations/stuff_%s.json', dataDir, dataTypeAnn);
if isAnnotation
    pngFolder = sprintf('%s/annotations/%s', dataDir, pngFolderName);
else
    pngFolder = sprintf('%s/results/%s', dataDir, pngFolderName);
    resPath = sprintf('%s/results/stuff_%s_results.json', dataDir, dataTypeRes);
end

% Create output folder
if ~exist(pngFolder, 'dir')
    mkdir(pngFolder);
end

% Initialize COCO ground-truth API
coco = CocoApi(annPath);
imgIds = coco.getImgIds();

% Initialize COCO result
if ~isAnnotation
    coco = coco.loadRes(resPath);
    imgIds = unique([coco.data.annotations.image_id]');
end

% Limit number of images
if exportImageLimit < numel(imgIds)
    imgIds = imgIds(1:exportImageLimit);
end
    
% Convert each image to a png
imgCount = numel(imgIds);
for i = 1 : imgCount
    imgId = imgIds(i);
    imgName = strrep(coco.loadImgs(imgId).file_name, '.jpg', '');
    fprintf('Exporting image %d of %d: %s\n', i, imgCount, imgName);
    segmentationPath = sprintf('%s/%s.png', pngFolder, imgName);
    CocoStuffHelper.cocoSegmentationToPng(coco, imgId, segmentationPath);
end

% Visualize the last image
originalImage = imread(coco.loadImgs(imgId).coco_url);
[segmentationImage, cmap] = imread(segmentationPath);
segmentationImage = ind2rgb(segmentationImage, cmap);
figure();
subplot(121);
imshow(originalImage);
axis('off');
title('original image');
    
subplot(122);
imshow(segmentationImage);
axis('off');
title('annotated image');