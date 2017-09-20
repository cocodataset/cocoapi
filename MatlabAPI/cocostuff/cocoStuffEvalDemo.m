% Shows how to use the evaluation script of the Stuff Segmentation
% Challenge.
%
% This script takes ground-truth annotations and result
% annotations of a semantic segmentation method and computes
% several performance metrics. See *CocoStuffEval.m* for more
% details.
%
% Microsoft COCO Toolbox.      version 2.0
% Data, paper, and tutorials available at:  http://mscoco.org/
% Code written by Piotr Dollar and Tsung-Yi Lin, 2015.
% Licensed under the Simplified BSD License [see coco/license.txt]

% Define paths
dataDir= '../..';
dataType = 'examples';
resType = 'examples';
annFile = sprintf('%s/annotations/stuff_%s.json', dataDir, dataType);
resFile = sprintf('%s/results/stuff_%s_results.json', dataDir, resType);

% Initialize COCO ground-truth API
cocoGt = CocoApi(annFile);

% Initialize COCO result API
cocoRes = cocoGt.loadRes(resFile);

% Initialize the evaluation
cocoEval = CocoStuffEval(cocoGt, cocoRes);

% Modify this to use only a subset of the images for evaluation
% imgIds = unique(cell2mat({cocoRes.data.annotations.image_id}))';
% cocoEval.params.imgIds = imgIds;

% Run evaluation on the example images
timer = tic;
cocoEval.evaluate();
cocoEval.summarize();
time = toc(timer);
fprintf('Evaluation took %.2fs!\n', time);