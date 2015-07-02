classdef CocoEval < handle
  % Interface for evaluating detection on the Microsoft COCO dataset.
  %
  % The usage for CocoEval is as follows:
  %  cocoGt=..., cocoDt=...       % load dataset and results
  %  E = CocoEval(cocoGt,cocoDt); % initialize CocoEval object
  %  E.params.recThrs = ...;      % set parameters as desired
  %  E.evaluate();                % run per image evaluation
  %  disp( E.evalImgs )           % inspect per image results
  %  E.accumulate();              % accumulate per image results
  %  disp( E.eval )               % inspect accumulated results
  % For example usage see evalDemo.m and http://mscoco.org/.
  %
  % The evaluation parameters are as follows (defaults in brackets):
  %  imgIds     - [all] N img ids to use for evaluation
  %  catIds     - [all] K cat ids to use for evaluation
  %  iouThrs    - [1/T:1/T:1] T=20 IoU thresholds for evaluation
  %  recThrs    - [1/R:1/R:1] R=1000 recall thresholds for evaluation
  %  maxDets    - [100] max number of allowed detections per image
  %  areaRng    - [0 1e10] object area range for evaluation
  %  useSegm    - [1] if true evaluate against ground-truth segments
  %  useCats    - [1] if true use category labels for evaluation
  % Note: if useSegm=0 the evaluation is run on bounding boxes.
  % Note: if useCats=0 category labels are ignored as in proposal scoring.
  %
  % evaluate(): evaluates detections on every image and every category and
  % concats the ~N*K results into the struct array "evalImgs" with fields:
  %  imgId      - results for the img with the given id
  %  catId      - results for the cat with the given id
  %  dtIds      - [1xD] id for each of the D detections (dt)
  %  gtIds      - [1xG] id for each of the G ground truths (gt)
  %  dtMatches  - [TxD] matching gt id at each IoU or 0
  %  gtMatches  - [TxG] matching dt id at each IoU or 0
  %  dtScores   - [1xD] confidence of each dt
  %  gtIgnore   - [1xG] ignore flag for each gt
  %  dtIgnore   - [TxD] ignore flag for each dt at each IoU
  %  ious       - [DxG] iou between every dt and gt
  %
  % accumulate(): accumulates the per-image, per-category evaluation
  % results in "evalImgs" into the struct "eval" with fields:
  %  params     - parameters used for evaluation
  %  date       - date evaluation was performed
  %  counts     - [T,R,K] counts for ious/recalls/categories
  %  precision  - [TxRxK] precision for every iou/recall/category
  %  recall     - [TxK] maximum recall for every iou/category
  %  ap         - average precision across all T*R*K precision values
  %  ar         - average recall across all T*K recall values
  % Note: categories with no instances are removed (K may decrease).
  %
  % See also CocoApi, MaskApi, cocoDemo, evalDemo
  %
  % Microsoft COCO Toolbox.      Version 1.0
  % Data, paper, and tutorials available at:  http://mscoco.org/
  % Code written by Piotr Dollar and Tsung-Yi Lin, 2015.
  % Licensed under the Simplified BSD License [see coco/license.txt]
  
  properties
    cocoGt      % ground truth COCO API
    cocoDt      % detections COCO API
    params      % evaluation parameters
    evalImgs    % per-image per-category evaluation results
    eval        % accumulated evaluation results
  end
  
  methods
    function ev = CocoEval( cocoGt, cocoDt )
      % Initialize CocoEval using coco APIs for gt and dt.
      if(nargin>0), ev.cocoGt = cocoGt; end
      if(nargin>1), ev.cocoDt = cocoDt; end
      if(nargin>0), ev.params.imgIds = sort(ev.cocoGt.getImgIds()); end
      if(nargin>0), ev.params.catIds = sort(ev.cocoGt.getCatIds()); end
      ev.params.iouThrs = .05:.05:1;
      ev.params.recThrs = .001:.001:1;
      ev.params.maxDets = 100;
      ev.params.areaRng = [0 1e10];
      ev.params.useSegm = 1;
      ev.params.useCats = 1;
    end
    
    function evaluate( ev )
      % Run per image evaluation on given images.
      fprintf('Running per image evaluation...      '); clk=clock;
      p=ev.params; if(~p.useCats), p.catIds=1; end
      p.imgIds=unique(p.imgIds); p.catIds=unique(p.catIds);
      ev.params=p; N=length(p.imgIds); K=length(p.catIds);
      [gts,nGt,cGt]=sortAnns(ev.cocoGt,p.imgIds,p.catIds,p.useCats);
      [dts,nDt,cDt]=sortAnns(ev.cocoDt,p.imgIds,p.catIds,p.useCats);
      [is,ks]=ndgrid(1:N,1:K); ev.evalImgs=cell(N*K,1);
      for i=1:K*N, if(nGt(i)==0 && nDt(i)==0), continue; end
        gt=gts(cGt(i)+1:cGt(i+1)); dt=dts(cDt(i)+1:cDt(i+1));
        if( p.useSegm )
          im=ev.cocoGt.loadImgs(p.imgIds(is(i))); h=im.height; w=im.width;
          for g=1:nGt(i), s=gt(g).segmentation; if(~isstruct(s))
              gt(g).segmentation=MaskApi.frPoly(s,h,w); end; end
          for d=1:nDt(i), s=dt(d).segmentation; if(isempty(s))
              dt(d).segmentation=MaskApi.frBbox(dt(d).bbox,h,w); end; end
        end
        q=p; q.catIds=p.catIds(ks(i)); q.imgIds=p.imgIds(is(i));
        ev.evalImgs{i}=CocoEval.evaluateImg(gt,dt,q);
      end
      ev.evalImgs=[ev.evalImgs{nGt>0|nDt>0}];
      fprintf('DONE (t=%0.2fs).\n',etime(clock,clk));
      
      function [anns,ns,cs] = sortAnns( coco, imgIds, catIds, useCats )
        % Return all annotations sorted by catId then imgId.
        a={'imgIds',imgIds}; if(useCats), a=[a,'catIds',catIds]; end
        anns=coco.loadAnns(coco.getAnnIds(a));
        ns=zeros(length(imgIds)*length(catIds),1); n=length(anns);
        [~,a]=ismember([anns.image_id],imgIds); b=1;
        if(useCats), [~,b]=ismember([anns.category_id],catIds); end
        a=(b-1)*length(imgIds)+a; [~,o]=sort(a); anns=anns(o);
        for b=1:n, ns(a(b))=ns(a(b))+1; end; cs=cumsum([0; ns]);
      end
    end
    
    function accumulate( ev )
      % Accumulate per image evaluation results.
      fprintf('Accumulating evaluation results...   '); clk=clock;
      if(isempty(ev.evalImgs)), error('Please run evaluate() first'); end
      p=ev.params; catIds=[ev.evalImgs.catId];
      T=length(p.iouThrs); R=length(p.recThrs); K=length(p.catIds);
      precision=zeros(T,R,K); recall=zeros(T,K); keep=true(1,K);
      for k=1:K
        E=ev.evalImgs(catIds==p.catIds(k));
        dtm=[E.dtMatches]; dtIg=[E.dtIgnore]; gtIg=[E.gtIgnore];
        np=nnz(~gtIg); if(np==0), keep(k)=0; continue; end
        [~,o]=sort([E.dtScores],'descend');
        tps= dtm & ~dtIg; tps=tps(:,o);
        fps=~dtm & ~dtIg; fps=fps(:,o);
        for t=1:T
          tp=cumsum(tps(t,:)); fp=cumsum(fps(t,:)); nd=length(tp);
          rc=tp/np; pr=tp./(fp+tp); if(nd), recall(t,k)=rc(end); end
          for i=nd-1:-1:1, pr(i)=max(pr(i+1),pr(i)); end
          i=1; r=1; while(r<=R && i<=nd), if(rc(i)<p.recThrs(r))
              i=i+1; else precision(t,r,k)=pr(i); r=r+1; end; end
        end
      end
      precision=precision(:,:,keep); recall=recall(:,keep);
      p.catIds=p.catIds(keep); K=length(p.catIds); ev.params=p;
      ev.eval=struct('params',p,'date',date,'counts',[T R K],...
        'precision',precision,'recall',recall,...
        'ap',mean(precision(:)),'ar',mean(recall(:)));
      fprintf('DONE (t=%0.2fs).\n',etime(clock,clk));
    end
  end
  
  methods( Static )
    function e = evaluateImg( gt, dt, params )
      % Run evaluation for a single image and category.
      p=params; T=length(p.iouThrs); aRng=p.areaRng;
      if(~isfield(gt,'ignore')), [gt(:).ignore]=deal(0); end
      a=[gt.area]; gtIg=[gt.iscrowd]|[gt.ignore]|a<aRng(1)|a>aRng(2);
      G=length(gt); D=length(dt); for g=1:G, gt(g).ignore=gtIg(g); end
      % sort dt highest score first, sort gt ignore last
      [~,o]=sort([gt.ignore],'ascend'); gt=gt(o);
      [~,o]=sort([dt.score],'descend'); dt=dt(o);
      if(D>p.maxDets), D=p.maxDets; dt=dt(1:D); end
      % compute iou between each dt and gt region
      iscrowd = uint8([gt.iscrowd]); t=p.useSegm;
      if(t), g=[gt.segmentation]; else g=reshape([gt.bbox],4,[])'; end
      if(t), d=[dt.segmentation]; else d=reshape([dt.bbox],4,[])'; end
      ious=MaskApi.iou(d,g,iscrowd);
      % attempt to match each (sorted) dt to each (sorted) gt
      gtm=zeros(T,G); gtIds=[gt.id]; gtIg=[gt.ignore];
      dtm=zeros(T,D); dtIds=[dt.id]; dtIg=zeros(T,D);
      for t=1:T
        for d=1:D
          % information about best match so far (m=0 -> unmatched)
          iou=min(p.iouThrs(t),1-1e-10); m=0;
          for g=1:G
            % if this gt already matched, and not a crowd, continue
            if( gtm(t,g)>0 && ~iscrowd(g) ), continue; end
            % if dt matched to reg gt, and on ignore gt, stop
            if( m>0 && gtIg(m)==0 && gtIg(g)==1 ), break; end
            % if match successful and best so far, store appropriately
            if( ious(d,g)>=iou ), iou=ious(d,g); m=g; end
          end
          % if match made store id of match for both dt and gt
          if(~m), continue; end; dtIg(t,d)=gtIg(m);
          dtm(t,d)=gtIds(m); gtm(t,m)=dtIds(d);
        end
      end
      % set unmatched detections outside of area range to ignore
      if(isempty(dt)), a=zeros(1,0); else a=[dt.area]; end
      dtIg = dtIg | (dtm==0 & repmat(a<aRng(1)|a>aRng(2),T,1));
      % store results for given image and category
      e=struct('imgId',p.imgIds,'catId',p.catIds,'dtIds',dtIds,'gtIds',...
        gtIds,'dtMatches',dtm,'gtMatches',gtm,'dtScores',[dt.score],...
        'gtIgnore',gtIg,'dtIgnore',dtIg,'ious',ious);
    end
  end
end
