% Converts a folder of .png images with segmentation results back
% to the COCO result format.
%
% The .png images should be indexed images with or without a color
% palette for visualization.
%
% Note that this script only works with image names in COCO 2017
% format (000000000934.jpg). The older format
% (COCO_train2014_000000000934.jpg) is not supported.
%
% Microsoft COCO Toolbox.      version 2.0
% Data, paper, and tutorials available at:  http://mscoco.org/
% Code written by Piotr Dollar and Tsung-Yi Lin, 2015.
% Licensed under the Simplified BSD License [see coco/license.txt]

% Define paths
dataDir = '../..';
resType = 'examples';
pngFolder = sprintf('%s/results/segmentations/%s', dataDir, resType);
jsonPath = sprintf('%s/results/stuff_%s_results.json', dataDir, resType);

% Get images in png folder
imgNames = dir([pngFolder, '/*.png']);
imgNames = {imgNames.name}';
imgNames = cellfun(@(x) x(1:end-4), imgNames, 'UniformOutput', false);
imgNames = sort(imgNames);
imgCount = numel(imgNames);

% Init
annCount = 0;

output = fopen(jsonPath, 'w');
fprintf('Writing results to: %s\n', jsonPath);

% Annotation start
fprintf(output, '[\n');

for i = 1 : imgCount
    imgName = imgNames{i};
    fprintf('Converting png image %d of %d: %s\n', i, imgCount, imgName);
    
    % Add stuff annotations
    pngPath = sprintf('%s/%s.png', pngFolder, imgName);
    tokens = strsplit(imgName, '_');
    if numel(tokens) == 1
        % COCO 2017 format
        imgId = str2double(imgName);
    elseif numel(tokens) == 3
        % Previous COCO format
        imgId = str2double(tokens(3));
    else
        error('Error: Invalid COCO file format!');
    end
    anns = CocoStuffHelper.pngToCocoResult(pngPath, imgId);
    
    % Write JSON
    str_ = gason(anns);
    str_ = str_(2:end-1);
    str_ = strrep(str_, '\', '\\'); % Matlab replaces escaped characters, so we need to escape them twice
    if numel(str_) > 0
        fprintf(output, str_);
        annCount = annCount + 1;
    end
    
    % Add comma separator
    if i < imgCount && numel(str_) > 0
        fprintf(output, ',');
    end
    
    % Add line break
    fprintf(output, '\n');
end

% Annotation end
fprintf(output, ']');

% Create an error if there are no annotations
if annCount == 0
    error('Error: The output file has 0 annotations and will not work with the COCO API!')
end