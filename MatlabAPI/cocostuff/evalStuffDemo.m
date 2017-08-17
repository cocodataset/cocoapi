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
dataType = 'train2017';
resType = 'examples';
annFile = sprintf('%s/annotations/stuff_%s.json', dataDir, dataType);
resFile = sprintf('%s/results/instances_stuff_%s_results.json', dataDir, resType);

% Initialize COCO ground-truth API
cocoGt = CocoApi(annFile);

% Initialize COCO result API
cocoRes = cocoGt.loadRes(resFile);

% Run evaluation on the example images
imgIds = unique(cell2mat({cocoRes.data.annotations.image_id}))';
cocoEval = CocoStuffEval(cocoGt, cocoRes);
cocoEval.params.imgIds = imgIds;
cocoEval.evaluate();
cocoEval.summarize();