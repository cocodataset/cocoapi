classdef CocoStuffHelper
    % Helper functions used to convert between different formats for the
    % COCO Stuff Segmentation Challenge.
    %
    % Microsoft COCO Toolbox.      version 2.0
    % Data, paper, and tutorials available at:  http://mscoco.org/
    % Code written by Piotr Dollar and Tsung-Yi Lin, 2015.
    % Licensed under the Simplified BSD License [see coco/license.txt]
    
    methods(Access = public, Static)
        function[Rs] = segmentationToCocoMask(labelMap, labelId)
            % Encodes a segmentation mask using the Mask API.
            %
            % USAGE
            %  [Rs] = segmentationToCocoMask(labelMap, labelId)
            %
            % INPUTS
            %  labelMap   - [h x w] segmentation map that indicates the label of each pixel
            %  labelId    - the label from labelMap that will be encoded
            %
            % OUTPUTS
            %  Rs         - the encoded label mask for label 'labelId'
            labelMask = labelMap == labelId;
            labelMask = uint8(labelMask);
            Rs = MaskApi.encode(labelMask);
            assert(numel(Rs) == 1);
        end
        
        function[anns] = segmentationToCocoResult(labelMap, imgId, varargin)
            % Convert a segmentation map to COCO stuff segmentation result format.
            %
            % USAGE
            %  [anns] = segmentationToCocoResult(labelMap, imgId, varargin)
            %
            % INPUTS
            %  labelMap   - [h x w] segmentation map that indicates the label of each pixel
            %  imgId      - the id of the COCO image (last part of the file name)
            %  params     - filtering parameters (struct or name/value pairs)
            %               setting any filter to [] skips that filter
            %    .stuffStartId - index where stuff classes start
            %
            % OUTPUTS
            %  anns       - a list of dicts for each label in this image
            %    .image_id     - the id of the COCO image
            %    .category_id  - the id of the stuff class of this annotation
            %    .segmentation - the RLE encoded segmentation of this class
            
            % Parse arguments
            p = inputParser;
            p.addParameter('stuffStartId', 92);
            parse(p, varargin{:});
            stuffStartId = p.Results.stuffStartId;
            
            % Get stuff labels
            shape = size(labelMap);
            if numel(shape) ~= 2
                error(['Error: Image has %d instead of 2 channels! Most likely you ' ...
                    'provided an RGB image instead of an indexed image (with or without color palette).'], numel(shape));
            end
            h = shape(1);
            w = shape(2);
            assert(h > 0 && w > 0);
            labelsAll = unique(labelMap);
            labelsStuff = labelsAll(labelsAll >= stuffStartId);
            
            % Add stuff annotations
            anns = struct('image_id', {}, 'category_id', {}, 'segmentation', {});
            for i = 1 : numel(labelsStuff)
                labelId = labelsStuff(i);
                
                % Create mask and encode it
                Rs = CocoStuffHelper.segmentationToCocoMask(labelMap, labelId);
                
                % Create annotation data and add it to the list
                anndata = struct();
                anndata.image_id = int32(imgId);
                anndata.category_id = int32(labelId);
                anndata.segmentation = Rs;
                anns(end+1) = anndata; %#ok<AGROW>
            end
        end
        
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
            
            % Parse arguments
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
            % labelMasks = MaskApi.decode([imgAnnots.segmentation]);
            for a = 1 : numel(imgAnnots)
                labelMask = CocoStuffHelper.annToMask(coco, imgAnnots(a)) == 1;
                % labelMask = labelMasks(:, :, a) == 1;
                newLabel = imgAnnots(a).category_id;
                
                if checkUniquePixelLabel && any(labelMap(labelMask) ~= 0)
                    error('Error: Some pixels have more than one label (image %d)!', imgId);
                end
                
                labelMap(labelMask) = newLabel;
            end
        end
        
        function[anns] = pngToCocoResult(pngPath, imgId, varargin)
            % Reads an indexed .png file with a label map from disk and converts it to COCO result format.
            %
            % USAGE
            %  [anns] = pngToCocoResult(pngPath, imgId, varargin)
            %
            % INPUTS
            %  pngPath   - the path of the .png file
            %  imgId     - the COCO id of the image (last part of the file name)
            %  params    - filtering parameters (struct or name/value pairs)
            %              setting any filter to [] skips that filter
            %    .stuffStartId - index where stuff classes start
            %
            % OUTPUTS
            %  anns       - a list of dicts for each label in this image
            %    .image_id     - the id of the COCO image
            %    .category_id  - the id of the stuff class of this annotation
            %    .segmentation - the RLE encoded segmentation of this class
            
            % Parse arguments
            p = inputParser;
            p.addParameter('stuffStartId', 92);
            parse(p, varargin{:});
            stuffStartId = p.Results.stuffStartId;
            
            % Read indexed .png file from disk
            labelMap = imread(pngPath);
            
            % Convert label map to COCO result format
            anns = CocoStuffHelper.segmentationToCocoResult(labelMap, imgId, 'stuffStartId', stuffStartId);
        end
        
        function cocoSegmentationToPng(coco, imgId, pngPath, varargin)
            % Convert COCO GT or results for a single image to a segmentation map and write it to disk.
            %
            % USAGE
            %  cocoSegmentationToPng(coco, imgId, pngPath, varargin)
            %
            % INPUTS
            %  coco       - an instance of the COCO API (ground-truth or result)
            %  imgId      - the COCO id of the image (last part of the file name)
            %  pngPath    - the path of the .png file
            %  params     - filtering parameters (struct or name/value pairs)
            %               setting any filter to [] skips that filter
            %    .includeCrowd - whether to include 'crowd' thing annotations as 'other' (or void)
            
            % Parse arguments
            p = inputParser;
            p.addParameter('includeCrowd', false);
            parse(p, varargin{:});
            includeCrowd = p.Results.includeCrowd;
            
            % Create label map
            labelMap = CocoStuffHelper.cocoSegmentationToSegmentationMap(coco, imgId, 'includeCrowd', includeCrowd);
            labelMap = uint8(labelMap);
            
            % Get color map (note that this varies from the Python
            % implementation, where the color map needs to have 3 * 256
            % entries)
            cmap = CocoStuffHelper.getCMap();
            
            % Write to png file
            imwrite(labelMap, cmap, pngPath);
        end
        
        function[cmap] = getCMap(varargin)
            % Create a color map for the classes in the COCO Stuff Segmentation Challenge.
            %
            % USAGE
            %  [cmap] = getCMap(varargin)
            %
            % INPUTS
            %  params - filtering parameters (struct or name/value pairs)
            %           setting any filter to [] skips that filter
            %    .stuffStartId - index where stuff classes start
            %    .stuffEndId   - index where stuff classes end
            %    .cmapFunc     - Matlab color map function
            %    .addThings    - whether to add a color for the 91 thing classes
            %    .addUnlabeled - whether to add a color for the 'unlabeled' class
            %    .addOther     - whether to add a color for the 'other' class
            %
            % OUTPUTS
            %  cmap   - [c, 3] a color map for c colors where the columns indicate the RGB values
            
            % Parse arguments
            p = inputParser;
            p.addParameter('stuffStartId', 92);
            p.addParameter('stuffEndId', 182);
            p.addParameter('cmapFunc', @jet);
            p.addParameter('addThings', true);
            p.addParameter('addUnlabeled', true);
            p.addParameter('addOther', true);
            parse(p, varargin{:});
            stuffStartId = p.Results.stuffStartId;
            stuffEndId = p.Results.stuffEndId;
            cmapFunc = p.Results.cmapFunc;
            addThings = p.Results.addThings;
            addUnlabeled = p.Results.addUnlabeled;
            addOther = p.Results.addOther;
            
            % Get jet color map from Matlab
            labelCount = stuffEndId - stuffStartId + 1;
            
            cmap = cmapFunc(labelCount);
            
            % Reduce value/brightness of stuff colors (easier in HSV format)
            cmap = rgb2hsv(cmap);
            cmap(:, 3) = cmap(:, 3) * 0.7;
            cmap = hsv2rgb(cmap);
            
            % Permute entries to avoid classes with similar name having similar colors
            st0 = rng();
            rng(42);
            perm = randperm(labelCount)';
            rng(st0.Seed);
            cmap = cmap(perm, :);
            
            % Add black (or any other) color for each thing class
            if addThings
                cmap = [zeros(stuffStartId - 1, 3); cmap];
            end
            
            % Add black color for 'unlabeled' class
            if addUnlabeled
                cmap = [0.0, 0.0, 0.0; cmap];
            end
            
            % Add yellow/orange color for 'other' class
            if addOther
                cmap = [cmap; 1.0, 0.843, 0.0];
            end
        end
    end
    
    methods(Access = private, Static)
        function[rle] = annToRLE(coco, ann) %#ok<INUSL>
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
            
            segm = ann.segmentation;
            if numel(segm) > 1 || ~ischar(segm.counts)
                error('Error: These cases are currently not supported!');
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
            
            rle = CocoStuffHelper.annToRLE(coco, ann);
            m = MaskApi.decode(rle);
        end
    end
end