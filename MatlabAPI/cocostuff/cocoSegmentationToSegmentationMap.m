function[labelMap] = cocoSegmentationToSegmentationMap(coco, imgId, varargin)
% Convert COCO GT or results for a single image to a segmentation map.
%
% USAGE
%  [labelMap] = cocoSegmentationToSegmentationMap(coco, imgId, varargin)
%
% INPUTS
%  coco       - initialized CocoStuffEval object
%  imgId      - id of the image to convert to a segmentation map
%  params     - filtering parameters (struct or name/value pairs)
%               setting any filter to [] skips that filter
%    .checkUniquePixelLabel - [true] whether every pixel can have at most one label
%    .includeCrowd          - [false] whether to include 'crowd' thing annotations as 'other' (or void)
%
% OUTPUTS
%  labelMap   - [h x w] segmentation map that indicates the label of each pixel
%
% Microsoft COCO Toolbox.      version 2.0
% Data, paper, and tutorials available at:  http://mscoco.org/
% Code written by Piotr Dollar and Tsung-Yi Lin, 2015.
% Licensed under the Simplified BSD License [see coco/license.txt]

p = inputParser;
p.addParameter('checkUniquePixelLabel', true);
p.addParameter('includeCrowd', false);
parse(p, varargin{:});
checkUniquePixelLabel = p.Results.checkUniquePixelLabel;
includeCrowd = p.Results.includeCrowd;

% Init
curImg = coco.loadImgs(imgId);
imageSize = [curImg.height, curImg.width];
labelMap = zeros(imageSize);

% Get annotations of the current image (may be empty)
% Note that coco.getAnnIds doesn't work here, as it does not support
% iscrowd for COCO result instances.
imgAnnotIds = [coco.data.annotations.id];
imgAnnotImgIds = [coco.data.annotations.image_id];
if ~includeCrowd && isfield(coco.data.annotations, 'iscrowd')
    imgAnnotIsCrowds = [coco.data.annotations.iscrowd];
    imgAnnotIds = imgAnnotIds(imgAnnotImgIds == imgId & imgAnnotIsCrowds ~= 1);
else
    imgAnnotIds = imgAnnotIds(imgAnnotImgIds == imgId);
end
imgAnnots = coco.loadAnns(imgAnnotIds);

% Combine all annotations of this image in labelMap
for a = 1 : numel(imgAnnots)
    labelMask = annToMask(coco, imgAnnots(a)) == 1;
    newLabel = imgAnnots(a).category_id;
    
    if checkUniquePixelLabel && any(labelMap(labelMask) ~= 0)
        error('Error: Some pixels have more than one label (image %d)!', imgId);
    end
    
    labelMap(labelMask) = newLabel;
end
end

function[rle] = annToRLE(coco, ann)
% Convert annotation which can be polygons, uncompressed RLE to RLE.
% Migrated from the coco.py class.
%
% USAGE
%  [rle] = annToRLE(coco, ann)
%
% INPUTS
%  coco           - initialized CocoStuffEval object
%  ann            - an annotation struct
%
% OUTPUTS
%  rle            - binary mask

t = coco.loadImgs(ann.image_id);
h = t.height;
w = t.width;
segm = ann.segmentation;
if numel(segm) > 1  % TODO: check whether this is equivalent to: type(segm) == list
    % polygon -- a single object might consist of multiple parts
    % we merge all parts into one mask rle code
    rles = MaskApi.frPyObjects(segm, h, w);
    rle = MaskApi.merge(rles);
elseif ~ischar(segm.counts) % TODO: check whether this is equivalent to: type(segm.counts) == list:
    % uncompressed RLE
    rle = MaskApi.frPyObjects(segm, h, w);
else
    % rle
    rle = ann.segmentation;
end
end

function[m] = annToMask(coco, ann)
% Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
% Migrated from the coco.py class.
%
% USAGE
%  [m] = annToMask(coco, ann)
%
% INPUTS
%  coco           - initialized CocoStuffEval object
%  ann            - an annotation struct
%
% OUTPUTS
%  m              - binary mask

rle = annToRLE(coco, ann);
m = MaskApi.decode(rle);
end