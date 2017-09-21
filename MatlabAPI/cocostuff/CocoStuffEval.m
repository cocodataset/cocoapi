classdef CocoStuffEval < handle
    % Internal functions for evaluating stuff segmentations against a ground-truth.
    %
    % The usage for COCOStuffeval is as follows:
    %  cocoGt=..., cocoRes=...            % load dataset and results
    %  E = CocoStuffEval(cocoGt, cocoRes);% initialize CocoStuffEval object
    %  E.params.imgIds = ...;             % set parameters as desired
    %  E.evaluate();                      % run per image evaluation
    %  E.summarize();                     % display summary metrics of results
    % For example usage see evalStuffDemo.m.
    %
    % Note: Our evaluation has to take place on all classes. If we remove one class
    % from evaluation, it is not clear what should happen to pixels for which a
    % method output that class (but where the ground-truth has a different class).
    %
    % The evaluation parameters are as follows (defaults in brackets):
    %  imgIds     - [all] N img ids to use for evaluation
    %
    % evaluate(): evaluates segmentations on each image and
    % stores the results in the 'eval' struct with fields:
    %  params     - parameters used for evaluation
    %  date       - date evaluation was performed
    %  confusion  - confusion matrix used for the final metrics
    %
    % summarize(): computes and prints the evaluation metrics.
    % results are printed to stdout and stored in:
    %  stats      - a numpy array of the evaluation metrics (mean IOU etc.)
    %  statsClass - a dict that stores per-class results in ious and maccs
    %
    % See also CocoApi, MaskApi, evalStuffDemo
    %
    % Microsoft COCO Toolbox.      version 2.0
    % Data, paper, and tutorials available at:  http://mscoco.org/
    % Code written by Piotr Dollar and Tsung-Yi Lin, 2015.
    % Licensed under the Simplified BSD License [see coco/license.txt]
    
    properties
        cocoGt        % Ground truth COCO API
        cocoRes        % Result COCO API
        stuffStartId  % Id of the first stuff class
        stuffEndId    % Id of the last stuff class
        addOther      % Whether to add a class that subsumes all thing classes
        
        eval          % Accumulated evaluation results
        confusion     % Confusion matrix that all metrics are computed on
        stats         % Result summarization
        statsClass    % Per-class results
        params        % Evaluation parameters (imgIds)
        catIds        % Categories taken into account for the evaluation
    end
    
    methods(Access = public)
        function ev = CocoStuffEval(cocoGt, cocoRes, varargin)
            % Initialize CocoStuffEval using COCO APIs for gt and resul
            %
            % USAGE
            %  ev = CocoStuffEval(cocoGt, cocoRes, varargin)
            %
            % INPUTS
            %  cocoGt    - initialized COCO ground-truth instance
            %  cocoRes   - initialized COCO result instance
            %  params    - filtering parameters (struct or name/value pairs)
            %              setting any filter to [] skips that filter
            %    .stuffStartId   - [92] id of the first stuff class
            %    .stuffEndId     - [182] id of the last stuff class
            %    .addOther       - [true] whether to add a class that subsumes all thing classes
            %
            % OUTPUTS
            %  ev        - initialized CocoStuffEval object
            p = inputParser;
            p.addParameter('stuffStartId', 92);
            p.addParameter('stuffEndId', 182);
            p.addParameter('addOther', true);
            parse(p, varargin{:});
            stuffStartId = p.Results.stuffStartId;
            stuffEndId = p.Results.stuffEndId;
            addOther = p.Results.addOther;
            
            ev.cocoGt = cocoGt;
            ev.cocoRes = cocoRes;
            ev.stuffStartId = stuffStartId;
            ev.stuffEndId = stuffEndId;
            ev.addOther = addOther;
            ev.params.imgIds = sort(cocoGt.getImgIds()); % By default we use all images from the GT file
            ev.catIds = ev.stuffStartId : (ev.stuffEndId + ev.addOther); % Take into account all stuff classes and one 'other' class
        end
        
        function evaluate(coco)
            % Run per image evaluation on given images and store results in self.confusion.
            % Note that this can take up to several hours.
            %
            % USAGE
            %  evaluate(coco)
            %
            % INPUTS
            %  coco      - initialized CocoStuffEval object
            timer = tic();
            imgIds = coco.params.imgIds;
            fprintf('Evaluating stuff segmentation on %d images and %d classes...\n', numel(imgIds), numel(coco.catIds));
            
            % Check that all images in params occur in GT and results
            gtImgIds = unique(coco.cocoGt.getImgIds());
            resImgIds = unique(coco.cocoRes.inds.annImgIds);
            missingInGt = imgIds(~ismember(imgIds, gtImgIds));
            missingInRes = imgIds(~ismember(imgIds, resImgIds));
            if ~isempty(missingInGt)
                error('Error: Some images specified in params.imgIds do not occur in the GT: %s', num2str(missingInGt));
            end
            if ~isempty(missingInRes)
                error('Error: %d evaluation images not found in the result!\n', numel(missingInRes));
            end
            
            % Create confusion matrix
            maxLabelCount = max(coco.cocoGt.inds.catIds);
            confusionCur = zeros(maxLabelCount, maxLabelCount);
            for i = 1 : numel(imgIds)
                imgId = imgIds(i);
                if i == 1 || i == numel(imgIds) || mod(i, 10) == 0
                    fprintf('Evaluating image %d of %d: %d\n', i, numel(imgIds), imgId);
                end
                confusionCur = coco.accumulateConfusion(coco.cocoGt, coco.cocoRes, confusionCur, imgId);
            end
            coco.confusion = confusionCur;
            
            % Set eval struct to be used later
            coco.eval.params = coco.params;
            coco.eval.date = datestr(now, 'yyyy-mm-dd HH:MM:SS');
            coco.eval.confusion = coco.confusion;
            
            time = toc(timer);
            fprintf('DONE (t=%0.2fs).\n', time);
        end
        
        function[confusion] = accumulateConfusion(coco, cocoGt, cocoRes, confusion, imgId)
            % Accumulate the pixels of the current image in the specified confusion matrix.
            % Note: For simplicity we do not map the labels to range [1, L],
            %       but keep the original indices when indexing 'confusion'.
            %
            % USAGE
            %  [confusion] = accumulateConfusion(coco, cocoGt, cocoRes, confusion, imgId)
            %
            % INPUTS
            %  coco           - initialized CocoStuffEval object
            %  cocoGt         - initialized COCO ground-truth instance
            %  cocoRes        - initialized COCO result instance
            %  confusion      - confusion matrix that all metrics are computed on
            %  imgId          - id of the image to compute the confusion on
            %
            % OUTPUTS
            %  confusion      - confusion matrix updated with pixels of the current image
            
            % Combine all annotations of this image in labelMapGt and labelMapRes
            labelMapGt  = CocoStuffHelper.cocoSegmentationToSegmentationMap(cocoGt,  imgId, 'includeCrowd', false);
            labelMapRes = CocoStuffHelper.cocoSegmentationToSegmentationMap(cocoRes, imgId, 'includeCrowd', false);
            
            % Check that the result has only valid labels
            labelMapResUn = unique(labelMapRes);
            invalidLabels = labelMapResUn(~ismember(labelMapResUn, coco.catIds));
            if ~isempty(invalidLabels)
                error('Error: Invalid classes predicted in the result file: %s. Please insert only labels in the range [%d, %d]!', sprintf('%s', num2str(invalidLabels')), min(coco.catIds), max(coco.catIds));
            end
            
            % Filter labels that are not in catIds (includes the 0 label)
            valid = ismember(labelMapGt, coco.catIds);
            
            % Gather annotations in confusion matrix
            confusion = confusion + accumarray([labelMapGt(valid), labelMapRes(valid)], 1, size(confusion));
        end
        
        function[stats, statsClass] = summarize(coco)
            % Compute and display the metrics for leaf nodes and super categories.
            %
            % USAGE
            %  [stats, statsClass] = summarize(coco)
            %
            % INPUTS
            %  coco       - initialized CocoStuffEval object
            %
            % OUTPUTS
            %  stats      - array of evaluation metrics 
            %  statsClass - struct of per-class metrics
            
            % Check if evaluate was run and then compute performance metrics
            if isempty(coco.eval)
                error('Error: Please run evaluate() first!');
            end
            
            % Compute confusion matrix for supercategories
            confusionCur = coco.confusion;
            confusionSup = coco.getSupCatConfusion(confusionCur);
            
            % Compute performance
            [miou, fwiou, macc, pacc, ious, maccs] = CocoStuffEval.computeMetrics(confusionCur);
            [miouSup, fwiouSup, maccSup, paccSup, iousSup, maccsSup] = CocoStuffEval.computeMetrics(confusionSup);
            
            % Store metrics
            stats = nan(8, 1);
            stats(1) = CocoStuffEval.printSummary('Mean IOU', 'leaves', miou);
            stats(2) = CocoStuffEval.printSummary('FW IOU', 'leaves', fwiou);
            stats(3) = CocoStuffEval.printSummary('Mean accuracy', 'leaves', macc);
            stats(4) = CocoStuffEval.printSummary('Pixel accuracy', 'leaves', pacc);
            stats(5) = CocoStuffEval.printSummary('Mean IOU', 'supercats', miouSup);
            stats(6) = CocoStuffEval.printSummary('FW IOU', 'supercats', fwiouSup);
            stats(7) = CocoStuffEval.printSummary('Mean accuracy', 'supercats', maccSup);
            stats(8) = CocoStuffEval.printSummary('Pixel accuracy', 'supercats', paccSup);
            
            % Store statsClass
            statsClass.ious = ious;
            statsClass.maccs = maccs;
            statsClass.iousSup = iousSup;
            statsClass.maccsSup = maccsSup;
            coco.stats = stats;
            coco.statsClass = statsClass;
        end
    end
    
    methods(Access = private)    
        function[confusionSup] = getSupCatConfusion(coco, confusion)
            % Maps the leaf category confusion matrix to a super category confusion matrix.
            %
            % USAGE
            %  [confusionSup] = getSupCatConfusion(coco, confusion)
            %
            % INPUTS
            %  coco         - initialized CocoStuffEval object
            %  confusion    - confusion matrix on the leaf node level
            %
            % OUTPUTS
            %  confusionSup - confusion matrix on the super category level
            
            % Retrieve supercategory mapping
            supCats = {coco.cocoGt.data.categories.supercategory}';
            supCatsUn = unique(supCats);
            supCatIds = cellfun(@(x) find(ismember(supCatsUn, x)), supCats);
            supCatCount = numel(supCatsUn);
            
            % Compute confusion matrix for supercategories
            confusionSup = zeros(supCatCount, supCatCount);
            for supCatIdA = 1 : supCatCount
                for supCatIdB = 1 : supCatCount
                    curLeavesA = find(supCatIds == supCatIdA) + coco.stuffStartId - 1;
                    curLeavesB = find(supCatIds == supCatIdB) + coco.stuffStartId - 1;
                    confusionLeaves = confusion(curLeavesA, curLeavesB);
                    confusionSup(supCatIdA, supCatIdB) = sum(confusionLeaves(:));
                end
            end
            assert(sum(confusionSup(:)) == sum(confusion(:)));
        end
    end
    
    methods(Access = private, Static)
        function[miou, fwiou, macc, pacc, ious, maccs] = computeMetrics(confusion)
            % Compute evaluation metrics given a confusion matrix.
            %
            % USAGE
            %  [miou, fwiou, macc, pacc, ious, maccs] = computeMetrics(confusion)
            %
            % INPUTS
            %  confusion - confusion matrix (leaf or super category level)
            %
            % OUTPUTS
            %  miou      - mean Intersection-over-Union (IoU)
            %  fwiou     - frequency-weighted IoU
            %  macc      - mean accuracy
            %  pacc      - pixel accuracy
            %  ious      - IoUs of each class (nan if missing)
            %  maccs     - accuracies of each class (nan if missing)
            
            % Init
            labelCount = size(confusion, 1);
            ious = nan(labelCount, 1);
            maccs = nan(labelCount, 1);
            
            % Get true positives, positive predictions and positive ground-truth
            total = sum(confusion(:));
            if total <= 0
                error('Error: Confusion matrix is empty!');
            end
            tp = diag(confusion);
            posPred = sum(confusion, 1)';
            posGt = sum(confusion, 2);
            
            % Check which classes have elements
            valid = posGt > 0;
            iousValid = valid & posGt + posPred - tp > 0;
            
            % Compute per-class results and frequencies
            ious(iousValid) = tp(iousValid) ./ (posGt(iousValid) + posPred(iousValid) - tp(iousValid));
            maccs(valid) = tp(valid) ./ posGt(valid);
            freqs = posGt / total;
            
            % Compute evaluation metrics
            miou = mean(ious(iousValid));
            fwiou = sum(ious(iousValid) .* freqs(iousValid));
            macc = mean(maccs(valid));
            pacc = sum(tp) / total;
        end
        
        function[res] = padStr(inStr, padding)
            % Pad a string with whitespaces on the right.
            %
            % USAGE
            %  padStr(inStr, padding)
            %
            % INPUTS
            %  inStr     - the string to be padded
            %  padding   - the target length of the string
            %
            % OUTPUTS
            %  res       - a padded string of length 'padding'
            res = [inStr, repmat(' ', [1, padding - numel(inStr)])];
        end
        
        function[val] = printSummary(titleStr, classStr, val)
            % Prints the current metric title, class type and value.
            %
            % USAGE
            %  [val] = printSummary(titleStr, classStr, val)
            %
            % INPUTS
            %  titleStr  - the name of the current metric
            %  classStr  - the type of category (leaves / supercats)
            %  val       - the value of the current metric
            %
            % OUTPUTS
            %  val       - the same val that was input (for convenience)
            fprintf(' %s @[ classes=%s ] = %0.4f\n', CocoStuffEval.padStr(titleStr, 14), CocoStuffEval.padStr(classStr, 8), val);
        end
    end
end