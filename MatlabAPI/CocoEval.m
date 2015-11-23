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
  %  E.summarize();               % display summary metrics of results
  % For example usage see evalDemo.m and http://mscoco.org/.
  %
  % The evaluation parameters are as follows (defaults in brackets):
  %  imgIds     - [all] N img ids to use for evaluation
  %  catIds     - [all] K cat ids to use for evaluation
  %  iouThrs    - [.5:.05:.95] T=10 IoU thresholds for evaluation
  %  recThrs    - [0:.01:1] R=101 recall thresholds for evaluation
  %  areaRng    - [...] A=4 object area ranges for evaluation
  %  maxDets    - [1 10 100] M=3 thresholds on max detections per image
  %  useSegm    - [1] if true evaluate against ground-truth segments
  %  useCats    - [1] if true use category labels for evaluation
  % Note: if useSegm=0 the evaluation is run on bounding boxes.
  % Note: if useCats=0 category labels are ignored as in proposal scoring.
  % Note: by default areaRng=[0 1e5; 0 32; 32 96; 96 1e5].^2. These A=4
  % settings correspond to all, small, medium, and large objects, resp.
  %
  % evaluate(): evaluates detections on every image and setting and concats
  % the results into the KxA struct array "evalImgs" with fields:
  %  dtIds      - [1xD] id for each of the D detections (dt)
  %  gtIds      - [1xG] id for each of the G ground truths (gt)
  %  dtImgIds   - [1xD] image id for each dt
  %  gtImgIds   - [1xG] image id for each gt
  %  dtMatches  - [TxD] matching gt id at each IoU or 0
  %  gtMatches  - [TxG] matching dt id at each IoU or 0
  %  dtScores   - [1xD] confidence of each dt
  %  dtIgnore   - [TxD] ignore flag for each dt at each IoU
  %  gtIgnore   - [1xG] ignore flag for each gt
  %
  % accumulate(): accumulates the per-image, per-category evaluation
  % results in "evalImgs" into the struct "eval" with fields:
  %  params     - parameters used for evaluation
  %  date       - date evaluation was performed
  %  counts     - [T,R,K,A,M] parameter dimensions (see above)
  %  precision  - [TxRxKxAxM] precision for every evaluation setting
  %  recall     - [TxKxAxM] max recall for every evaluation setting
  % Note: precision and recall==-1 for settings with no gt objects.
  %
  % See also CocoApi, MaskApi, cocoDemo, evalDemo
  %
  % Microsoft COCO Toolbox.      version 2.0
  % Data, paper, and tutorials available at:  http://mscoco.org/
  % Code written by Piotr Dollar and Tsung-Yi Lin, 2015.
  % Licensed under the Simplified BSD License [see coco/license.txt]
  
  properties
    cocoGt      % ground truth COCO API
    cocoDt      % detections COCO API
    params      % evaluation parameters
    evalImgs    % per-image per-category evaluation results
    eval        % accumulated evaluation results
    stats       % evaluation summary statistics
  end
  
  methods
    function ev = CocoEval( cocoGt, cocoDt )
      % Initialize CocoEval using coco APIs for gt and dt.
      if(nargin>0), ev.cocoGt = cocoGt; end
      if(nargin>1), ev.cocoDt = cocoDt; end
      if(nargin>0), ev.params.imgIds = sort(ev.cocoGt.getImgIds()); end
      if(nargin>0), ev.params.catIds = sort(ev.cocoGt.getCatIds()); end
      ev.params.iouThrs = .5:.05:.95;
      ev.params.recThrs = 0:.01:1;
      ev.params.areaRng = [0 1e5; 0 32; 32 96; 96 1e5].^2;
      ev.params.maxDets = [1 10 100];
      ev.params.useSegm = 1;
      ev.params.useCats = 1;
    end
    
    function evaluate( ev )
      % Run per image evaluation on given images.
      fprintf('Running per image evaluation...      '); clk=clock;
      p=ev.params; if(~p.useCats), p.catIds=1; end
      p.imgIds=unique(p.imgIds); p.catIds=unique(p.catIds); ev.params=p;
      N=length(p.imgIds); K=length(p.catIds); A=size(p.areaRng,1);
      [nGt,iGt]=getAnnCounts(ev.cocoGt,p.imgIds,p.catIds,p.useCats);
      [nDt,iDt]=getAnnCounts(ev.cocoDt,p.imgIds,p.catIds,p.useCats);
      [ks,is]=ndgrid(1:K,1:N); ev.evalImgs=cell(N,K,A);
      for i=1:K*N, if(nGt(i)==0 && nDt(i)==0), continue; end
        gt=ev.cocoGt.data.annotations(iGt(i):iGt(i)+nGt(i)-1);
        dt=ev.cocoDt.data.annotations(iDt(i):iDt(i)+nDt(i)-1);
        if( p.useSegm )
          im=ev.cocoGt.loadImgs(p.imgIds(is(i))); h=im.height; w=im.width;
          for g=1:nGt(i), s=gt(g).segmentation; if(~isstruct(s))
              gt(g).segmentation=MaskApi.frPoly(s,h,w); end; end
          f='segmentation'; if(isempty(dt)), [dt(:).(f)]=deal(); end
          if(~isfield(dt,f)), s=MaskApi.frBbox(cat(1,dt.bbox),h,w);
            for d=1:nDt(i), dt(d).(f)=s(d); end; end
        else
          f='bbox'; if(isempty(dt)), [dt(:).(f)]=deal(); end
          if(~isfield(dt,f)), s=MaskApi.toBbox([dt.segmentation]);
            for d=1:nDt(i), dt(d).(f)=s(d,:); end; end
        end
        q=p; q.imgIds=p.imgIds(is(i)); q.maxDets=max(p.maxDets);
        for j=1:A, q.areaRng=p.areaRng(j,:);
          ev.evalImgs{is(i),ks(i),j}=CocoEval.evaluateImg(gt,dt,q); end
      end
      E=ev.evalImgs; nms={'dtIds','gtIds','dtImgIds','gtImgIds',...
        'dtMatches','gtMatches','dtScores','dtIgnore','gtIgnore'};
      ev.evalImgs=repmat(cell2struct(cell(9,1),nms,1),K,A);
      for i=1:K, is=find(nGt(i,:)>0|nDt(i,:)>0);
        if(~isempty(is)), for j=1:A, E0=[E{is,i,j}]; for k=1:9
              ev.evalImgs(i,j).(nms{k})=[E0{k:9:end}]; end; end; end
      end
      fprintf('DONE (t=%0.2fs).\n',etime(clock,clk));
      
      function [ns,is] = getAnnCounts( coco, imgIds, catIds, useCats )
        % Return ann counts and indices for given imgIds and catIds.
        as=sort(coco.getCatIds()); [~,a]=ismember(coco.inds.annCatIds,as);
        bs=sort(coco.getImgIds()); [~,b]=ismember(coco.inds.annImgIds,bs);
        if(~useCats), a(:)=1; as=1; end; ns=zeros(length(as),length(bs));
        for ind=1:length(a), ns(a(ind),b(ind))=ns(a(ind),b(ind))+1; end
        is=reshape(cumsum([0 ns(1:end-1)])+1,size(ns));
        [~,a]=ismember(catIds,as); [~,b]=ismember(imgIds,bs);
        ns=ns(a,b); is=is(a,b);
      end
    end
    
    function accumulate( ev )
      % Accumulate per image evaluation results.
      fprintf('Accumulating evaluation results...   '); clk=clock;
      if(isempty(ev.evalImgs)), error('Please run evaluate() first'); end
      p=ev.params; T=length(p.iouThrs); R=length(p.recThrs);
      K=length(p.catIds); A=size(p.areaRng,1); M=length(p.maxDets);
      precision=-ones(T,R,K,A,M); recall=-ones(T,K,A,M);
      [ks,as,ms]=ndgrid(1:K,1:A,1:M);
      for k=1:K*A*M
        E=ev.evalImgs(ks(k),as(k)); is=E.dtImgIds; mx=p.maxDets(ms(k));
        np=nnz(~E.gtIgnore); if(np==0), continue; end
        t=[0 find(diff(is)) length(is)]; t=t(2:end)-t(1:end-1); is=is<0;
        r=0; for i=1:length(t), is(r+1:r+min(mx,t(i)))=1; r=r+t(i); end
        dtm=E.dtMatches(:,is); dtIg=E.dtIgnore(:,is);
        [~,o]=sort(E.dtScores(is),'descend');
        tps=reshape( dtm & ~dtIg,T,[]); tps=tps(:,o);
        fps=reshape(~dtm & ~dtIg,T,[]); fps=fps(:,o);
        precision(:,:,k)=0; recall(:,k)=0;
        for t=1:T
          tp=cumsum(tps(t,:)); fp=cumsum(fps(t,:)); nd=length(tp);
          rc=tp/np; pr=tp./(fp+tp); q=zeros(1,R); thrs=p.recThrs;
          if(nd==0 || tp(nd)==0), continue; end; recall(t,k)=rc(end);
          for i=nd-1:-1:1, pr(i)=max(pr(i+1),pr(i)); end; i=1; r=1; s=100;
          while(r<=R && i<=nd), if(rc(i)>=thrs(r)), q(r)=pr(i); r=r+1; else
              i=i+1; if(i+s<=nd && rc(i+s)<thrs(r)), i=i+s; end; end; end
          precision(t,:,k)=q;
        end
      end
      ev.eval=struct('params',p,'date',date,'counts',[T R K A M],...
        'precision',precision,'recall',recall);
      fprintf('DONE (t=%0.2fs).\n',etime(clock,clk));
    end
    
    function summarize( ev )
      % Compute and display summary metrics for evaluation results.
      if(isempty(ev.eval)), error('Please run accumulate() first'); end
      ev.stats=zeros(1,12);
      ev.stats(1) = summarize1(1,':','all',100);
      ev.stats(2) = summarize1(1,.50,'all',100);
      ev.stats(3) = summarize1(1,.75,'all',100);
      ev.stats(4) = summarize1(1,':','small',100);
      ev.stats(5) = summarize1(1,':','medium',100);
      ev.stats(6) = summarize1(1,':','large',100);
      ev.stats(7) = summarize1(0,':','all',1);
      ev.stats(8) = summarize1(0,':','all',10);
      ev.stats(9) = summarize1(0,':','all',100);
      ev.stats(10) = summarize1(0,':','small',100);
      ev.stats(11) = summarize1(0,':','medium',100);
      ev.stats(12) = summarize1(0,':','large',100);
      
      function s = summarize1( ap, iouThr, areaRng, maxDets )
        p=ev.params; i=iouThr; m=find(p.maxDets==maxDets);
        if(i~=':'), iStr=sprintf('%.2f     ',i); i=find(p.iouThrs==i);
        else iStr=sprintf('%.2f:%.2f',min(p.iouThrs),max(p.iouThrs)); end
        as=[0 1e5; 0 32; 32 96; 96 1e5].^2; a=find(areaRng(1)=='asml');
        a=find(p.areaRng(:,1)==as(a,1) & p.areaRng(:,2)==as(a,2));
        if(ap), tStr='Precision (AP)'; s=ev.eval.precision(i,:,:,a,m);
        else    tStr='Recall    (AR)'; s=ev.eval.recall(i,:,a,m); end
        fStr=' Average %s @[ IoU=%s | area=%6s | maxDets=%3i ] = %.3f\n';
        s=mean(s(s>=0)); fprintf(fStr,tStr,iStr,areaRng,maxDets,s);
      end
    end
    
    function visualize( ev, varargin )
      % Crop detector bbox results after evaluation (fp, tp, or fn).
      %  Preliminary implementation, undocumented. Use at your own risk.
      %  Require's Piotr's Toolbox (https://github.com/pdollar/toolbox/).
      def = { 'imgDir','../images/val2014/', 'outDir','visualize', ...
        'catIds',[], 'areaIds',1:4, 'type',{'tp','fp','fn'}, ...
        'dim',200, 'pad',1.5, 'ds',[10 10 1] };
      p = getPrmDflt(varargin,def,0);
      if(isempty(p.catIds)), p.catIds=ev.params.catIds; end
      type=p.type; d=p.dim; pad=p.pad; ds=p.ds;
      % recursive call unless performing singleton task
      if(length(p.catIds)>1), q=p; for i=1:length(p.catIds)
          q.catIds=p.catIds(i); ev.visualize(q); end; return; end
      if(length(p.areaIds)>1), q=p; for i=1:length(p.areaIds)
          q.areaIds=p.areaIds(i); ev.visualize(q); end; return; end
      if(iscell(p.type)), q=p; for i=1:length(p.type)
          q.type=p.type{i}; ev.visualize(q); end; return; end
      % generate file name for result
      areaNms={'all','small','medium','large'};
      catNm=regexprep(ev.cocoGt.loadCats(p.catIds).name,' ','_');
      fn=sprintf('%s/%s-%s-%s%%03i.jpg',p.outDir,...
        catNm,areaNms{p.areaIds},type); disp(fn);
      if(exist(sprintf(fn,1),'file')), return; end
      % select appropriate gt and dt according to type
      E=ev.evalImgs(p.catIds==ev.params.catIds,p.areaIds);
      E.dtMatches=E.dtMatches(1,:); E=select(E,1,~E.dtIgnore(1,:));
      E.gtMatches=E.gtMatches(1,:); E=select(E,0,~E.gtIgnore(1,:));
      [~,o]=sort(E.dtScores,'descend'); E=select(E,1,o);
      if(strcmp(type,'fn'))
        E=select(E,0,~E.gtMatches); gt=E.gtIds; G=1; D=0;
      elseif(strcmp(type,'tp'))
        E=select(E,1,E.dtMatches>0); dt=E.dtIds; gt=E.dtMatches; G=1; D=1;
      elseif(strcmp(type,'fp'))
        E=select(E,1,~E.dtMatches); dt=E.dtIds; G=0; D=1;
      end
      % load dt, gt, and im and crop region bbs
      if(D), is=E.dtImgIds; else is=E.gtImgIds; end
      n=min(prod(ds),length(is)); is=ev.cocoGt.loadImgs(is(1:n));
      if(G), gt=ev.cocoGt.loadAnns(gt(1:n)); bb=gt; end
      if(D), dt=ev.cocoDt.loadAnns(dt(1:n)); bb=dt; end
      if(~n), return; end; bb=cat(1,bb.bbox); bb(:,1:2)=bb(:,1:2)+1;
      r=max(bb(:,3:4),[],2)*pad/d; r=[r r r r];
      bb=bbApply('resize',bbApply('squarify',bb,0),pad,pad);
      % get dt and gt bbs in relative coordinates
      if(G), gtBb=cat(1,gt.bbox); gtBb(:,1:2)=gtBb(:,1:2)-bb(:,1:2);
        gtBb=gtBb./r; if(~D), gtBb=[gtBb round([gt(1:n).area])']; end; end
      if(D), dtBb=cat(1,dt.bbox); dtBb(:,1:2)=dtBb(:,1:2)-bb(:,1:2);
        dtBb=dtBb./r; dtBb=[dtBb E.dtScores(1:n)']; end
      % crop image samples appropriately
      ds(3)=ceil(n/prod(ds(1:2))); Is=cell(ds);
      for i=1:n
        I=imread(sprintf('%s/%s',p.imgDir,is(i).file_name));
        I=bbApply('crop',I,bb(i,:),0,[d d]); I=I{1};
        if(D), I=bbApply('embed',I,dtBb(i,:),'col',[0 0 255]); end
        if(G), I=bbApply('embed',I,gtBb(i,:),'col',[0 255 0]); end
        Is{i}=I;
      end
      for i=n+1:prod(ds), Is{i}=zeros(d,d,3,'uint8'); end
      I=reshape(cell2mat(permute(Is,[2 1 3])),ds(1)*d,ds(2)*d,3,ds(3));
      for i=1:ds(3), imwrite(imresize(I(:,:,:,i),.5),sprintf(fn,i)); end
      % helper function for taking subset of E
      function E = select( E, D, kp )
        fs={'Matches','Ids','ImgIds','Scores'}; pr={'gt','dt'};
        for f=1:3+D, fd=[pr{D+1} fs{f}]; E.(fd)=E.(fd)(kp); end
      end
    end
    
    function analyze( ev )
      % Derek Hoiem style analyis of false positives.
      %  Preliminary implementation, undocumented. Use at your own risk.
      if(~isfield(ev.cocoGt.data.annotations,'ignore')),
        [ev.cocoGt.data.annotations.ignore]=deal(0); end
      ev.params.maxDets=100; ev.params.areaRng=[0 1e10];
      ev.params.useCats=0; cocoDtAll=ev.cocoDt;
      if(~exist('./analyze','dir')), mkdir('analyze'); end
      catIds=ev.cocoGt.getCatIds(); K=length(catIds);
      rs=ev.params.recThrs; ps=zeros(7,length(rs),K);
      for k=1:K, catId=catIds(k);
        nm=ev.cocoGt.loadCats(catId); nm=[nm.supercategory '-' nm.name];
        fprintf('Analyzing %s...\n',nm); clk=clock;
        dt=ev.cocoDt.data.annotations; dt=dt([dt.category_id]==catId);
        D=ev.cocoGt.data; D.annotations=dt; ev.cocoDt=CocoApi(D);
        ev.params.iouThrs=[.75 .5 .1]; ps(1:3,:,k)=precision(ev,0,catId);
        ev.params.iouThrs=.1; ps(4,:,k)=precision(ev,1,catId);
        ps(5,:,k)=precision(ev,2,catId); ev.cocoDt=cocoDtAll;
        ps(6,:,k)=ps(5,:,k)>0; ps(7,:,k)=1; makeplot(rs,ps(:,:,k),nm);
        fprintf('DONE (t=%0.2fs).\n\n',etime(clock,clk));
      end
      makeplot(rs,mean(ps,3),'overall');
      sup={ev.cocoGt.data.categories.supercategory};
      for k=unique(sup), ps1=mean(ps(:,:,strcmp(sup,k)),3);
        makeplot(rs,ps1,k{1}); end
      
      function p = precision( ev, type, catId )
        % Helper that computes different types of precision.
        gt=ev.cocoGt; gtOrig=gt; ann=gt.data.annotations;
        if( type==0 )
          ann=ann([ann.category_id]==catId);
          data=gt.data; data.annotations=ann; gt=CocoApi(data);
        elseif( type==1 )
          is=gt.getCatIds('supNms',gt.loadCats(catId).supercategory);
          ann=ann(ismember([ann.category_id],is));
          [ann([ann.category_id]~=catId).ignore]=deal(1);
          data=gt.data; data.annotations=ann; gt=CocoApi(data);
        end
        [gt.data.annotations(gt.inds.annCatIds~=catId).ignore]=deal(1);
        ev.cocoGt=gt; ev.evaluate(); ev.accumulate();
        p=ev.eval.precision; ev.cocoGt=gtOrig;
      end
      
      function makeplot( rs, ps, nm )
        % Plot FP breakdown using area plot (optionally pdfcrop).
        cs=[ones(2,3); .31 .51 .74; .75 .31 .30;
          .36 .90 .38; .50 .39 .64; 1 .6 0]; m=size(ps,1);
        ap=round(mean(ps,2)*1000); ds=[ps(1,:); diff(ps)]';
        ls={'C75','C50','Loc','Sim','Oth','BG','[1.00] FN'};
        for i=1:m-1, ls{i}=sprintf('[.%3i] %s',ap(i),ls{i}); end
        figure(1); h=area(rs,ds); title(nm); legend(ls,'location','sw');
        for i=1:m, set(h(i),'FaceColor',cs(i,:)); end
        xlabel('recall'); ylabel('precision'); set(gca,'fontsize',20)
        nm=['analyze/' regexprep(nm,' ','_')]; print(nm,'-dpdf')
        if(0), setenv('PATH',[getenv('PATH') ':/Library/TeX/texbin/']); end
        if(0), system(['pdfcrop ' nm '.pdf ' nm '.pdf']); end
      end
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
      if(t), g=[gt.segmentation]; else g=cat(1,gt.bbox); end
      if(t), d=[dt.segmentation]; else d=cat(1,dt.bbox); end
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
      dtImgIds=ones(1,D)*p.imgIds; gtImgIds=ones(1,G)*p.imgIds;
      e = {dtIds,gtIds,dtImgIds,gtImgIds,dtm,gtm,[dt.score],dtIg,gtIg};
    end
  end
end
