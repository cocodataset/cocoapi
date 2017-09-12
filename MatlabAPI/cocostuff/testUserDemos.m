% A simple test case that runs all COCO Stuff demos relevant to a user.
%
% Note: Some demos require user interaction (e.g. to close a figure).
%
% Microsoft COCO Toolbox.      version 2.0
% Data, paper, and tutorials available at:  http://mscoco.org/
% Code written by Piotr Dollar and Tsung-Yi Lin, 2015.
% Licensed under the Simplified BSD License [see coco/license.txt]

% Run demos
fprintf('Running test...\n');
cocoSegmentationToPngDemo();
pngToCocoResultDemo();
cocoStuffDemo();
cocoStuffEvalDemo();
fprintf('Test successfully finished!\n');