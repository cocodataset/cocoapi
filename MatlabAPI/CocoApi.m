classdef CocoApi
  % Interface for accessing the Microsoft COCO dataset.
  %
  % Microsoft COCO is a large image dataset designed for object detection,
  % segmentation, and caption generation. CocoApi.m is a Matlab API that
  % assists in loading, parsing and visualizing the annotations in COCO.
  % Please visit http://mscoco.org/ for more information on COCO, including
  % for the data, paper, and tutorials. The exact format of the annotations
  % is also described on the COCO website. For example usage of the CocoApi
  % please see cocoDemo.m. In addition to this API, please download both
  % the COCO images and annotations in order to run the demo.
  %
  % An alternative to using the API is to load the annotations directly
  % into a Matlab struct. This can be achieved via:
  %  data = gason(fileread(annFile));
  % Using the API provides additional utility functions. Note that this API
  % supports both *instance* and *caption* annotations. In the case of
  % captions not all functions are defined (e.g. categories are undefined).
  %
  % The following API functions are defined:
  %  CocoApi    - Load COCO annotation file and prepare data structures.
  %  decodeMask - Decode binary mask M encoded via run-length encoding.
  %  encodeMask - Encode binary mask M using run-length encoding.
  %  getAnnIds  - Get ann ids that satisfy given filter conditions.
  %  getCatIds  - Get cat ids that satisfy given filter conditions.
  %  getImgIds  - Get img ids that satisfy given filter conditions.
  %  loadAnns   - Load anns with the specified ids.
  %  loadCats   - Load cats with the specified ids.
  %  loadImgs   - Load imgs with the specified ids.
  %  segToMask  - Convert polygon segmentation to binary mask.
  %  showAnns   - Display the specified annotations.
  % Throughout the API "ann"=annotation, "cat"=category, and "img"=image.
  % Help on each functions can be accessed by: "help CocoApi>function".
  %
  % See also cocoDemo, CocoApi>CocoApi, CocoApi>decodeMask,
  % CocoApi>encodeMask, CocoApi>getAnnIds, CocoApi>getCatIds,
  % CocoApi>getImgIds, CocoApi>loadAnns, CocoApi>loadCats,
  % CocoApi>loadImgs, CocoApi>segToMask, CocoApi>showAnns
  %
  % Microsoft COCO Toolbox.      Version 1.0
  % Data, paper, and tutorials available at:  http://mscoco.org/
  % Code written by Piotr Dollar and Tsung-Yi Lin, 2015.
  % Licensed under the Simplified BSD License [see coco/license.txt]
  
  properties
    data    % COCO annotation data structure
    inds    % data structures for fast indexing
  end
  
  methods
    function coco = CocoApi( annFile )
      % Load COCO annotation file and prepare data structures.
      %
      % USAGE
      %  coco = CocoApi( annFile )
      %
      % INPUTS
      %  annFile   - COCO annotation filename
      %
      % OUTPUTS
      %  coco      - initialized coco object
      fprintf('Loading and preparing annotations... '); clk=clock;
      coco.data=gason(fileread(annFile));
      anns = coco.data.annotations;
      if( strcmp(coco.data.type,'instances') )
        is.annCatIds = [anns.category_id]';
        is.annAreas = [anns.area]';
        is.annIscrowd = [anns.iscrowd]';
      end
      is.annIds = [anns.id]';
      is.annIdsMap = makeMap(is.annIds);
      is.annImgIds = [anns.image_id]';
      is.imgIds = [coco.data.images.id]';
      is.imgIdsMap = makeMap(is.imgIds);
      is.imgAnnIdsMap = makeMultiMap(is.imgIds,...
        is.imgIdsMap,is.annImgIds,is.annIds,0);
      if( strcmp(coco.data.type,'instances') )
        is.catIds = [coco.data.categories.id]';
        is.catIdsMap = makeMap(is.catIds);
        is.catImgIdsMap = makeMultiMap(is.catIds,...
          is.catIdsMap,is.annCatIds,is.annImgIds,1);
      end
      coco.inds=is; fprintf('DONE (t=%0.2fs).\n',etime(clock,clk));
      
      function map = makeMap( keys )
        % Make map from key to integer id associated with key.
        map=containers.Map(keys,1:length(keys));
      end
      
      function map = makeMultiMap( keys, keysMap, keysAll, valsAll, sqz )
        % Make map from keys to set of vals associated with each key.
        js=values(keysMap,num2cell(keysAll)); js=[js{:}];
        m=length(js); n=length(keys); k=zeros(1,n);
        for i=1:m, j=js(i); k(j)=k(j)+1; end; vs=zeros(n,max(k)); k(:)=0;
        for i=1:m, j=js(i); k(j)=k(j)+1; vs(j,k(j))=valsAll(i); end
        map = containers.Map('KeyType','double','ValueType','any');
        if(sqz), for j=1:n, map(keys(j))=unique(vs(j,1:k(j))); end
        else for j=1:n, map(keys(j))=vs(j,1:k(j)); end; end
      end
    end
    
    function ids = getAnnIds( coco, varargin )
      % Get ann ids that satisfy given filter conditions.
      %
      % USAGE
      %  ids = coco.getAnnIds( params )
      %
      % INPUTS
      %  params     - filtering parameters (struct or name/value pairs)
      %               setting any filter to [] skips that filter
      %   .imgIds     - [] get anns for given imgs
      %   .catIds     - [] get anns for given cats
      %   .areaRng    - [] get anns for given area range (e.g. [0 inf])
      %   .iscrowd    - [] get anns for given crowd label (0 or 1)
      %
      % OUTPUTS
      %  ids        - integer array of ann ids
      def = {'imgIds',[],'catIds',[],'areaRng',[],'iscrowd',[]};
      [imgIds,catIds,ar,iscrowd] = getPrmDflt(varargin,def,1);
      if( length(imgIds)==1 )
        t = coco.loadAnns(coco.inds.imgAnnIdsMap(imgIds));
        if(~isempty(catIds)), t = t(ismember([t.category_id],catIds)); end
        if(~isempty(ar)), a=[t.area]; t = t(a>=ar(1) & a<=ar(2)); end
        if(~isempty(iscrowd)), t = t([t.iscrowd]==iscrowd); end
        ids = [t.id];
      else
        ids=coco.inds.annIds; K = true(length(ids),1); t = coco.inds;
        if(~isempty(imgIds)), K = K & ismember(t.annImgIds,imgIds); end
        if(~isempty(catIds)), K = K & ismember(t.annCatIds,catIds); end
        if(~isempty(ar)), a=t.annAreas; K = K & a>=ar(1) & a<=ar(2); end
        if(~isempty(iscrowd)), K = K & t.annIscrowd==iscrowd; end
        ids=ids(K);
      end
    end
    
    function ids = getCatIds( coco, varargin )
      % Get cat ids that satisfy given filter conditions.
      %
      % USAGE
      %  ids = coco.getCatIds( params )
      %
      % INPUTS
      %  params     - filtering parameters (struct or name/value pairs)
      %               setting any filter to [] skips that filter
      %   .catNms     - [] get cats for given cat names
      %   .supNms     - [] get cats for given supercategory names
      %   .catIds     - [] get cats for given cat ids
      %
      % OUTPUTS
      %  ids        - integer array of cat ids
      def={'catNms',[],'supNms',[],'catIds',[]}; t=coco.data.categories;
      [catNms,supNms,catIds] = getPrmDflt(varargin,def,1);
      if(~isempty(catNms)), t = t(ismember({t.name},catNms)); end
      if(~isempty(supNms)), t = t(ismember({t.supercategory},supNms)); end
      if(~isempty(catIds)), t = t(ismember([t.ids],catIds)); end
      ids = [t.id];
    end
    
    function ids = getImgIds( coco, varargin )
      % Get img ids that satisfy given filter conditions.
      %
      % USAGE
      %  ids = coco.getImgIds( params )
      %
      % INPUTS
      %  params     - filtering parameters (struct or name/value pairs)
      %               setting any filter to [] skips that filter
      %   .imgIds     - [] get imgs for given ids
      %   .catIds     - [] get imgs with all given cats
      %
      % OUTPUTS
      %  ids        - integer array of img ids
      def={'imgIds',[],'catIds',[]}; ids=coco.inds.imgIds;
      [imgIds,catIds] = getPrmDflt(varargin,def,1);
      if(~isempty(imgIds)), ids=intersect(ids,imgIds); end
      t=values(coco.inds.catImgIdsMap,num2cell(catIds));
      for i=1:length(t), ids=intersect(ids,t{i}); end
    end
    
    function anns = loadAnns( coco, ids )
      % Load anns with the specified ids.
      %
      % USAGE
      %  anns = coco.loadAnns( ids )
      %
      % INPUTS
      %  ids        - integer ids specifying anns
      %
      % OUTPUTS
      %  anns       - loaded ann objects
      ids = values(coco.inds.annIdsMap,num2cell(ids));
      anns = coco.data.annotations([ids{:}]);
    end
    
    function cats = loadCats( coco, ids )
      % Load cats with the specified ids.
      %
      % USAGE
      %  cats = coco.loadCats( ids )
      %
      % INPUTS
      %  ids        - integer ids specifying cats
      %
      % OUTPUTS
      %  cats       - loaded cat objects
      ids = values(coco.inds.catIdsMap,num2cell(ids));
      cats = coco.data.categories([ids{:}]);
    end
    
    function imgs = loadImgs( coco, ids )
      % Load imgs with the specified ids.
      %
      % USAGE
      %  imgs = coco.loadImgs( ids )
      %
      % INPUTS
      %  ids        - integer ids specifying imgs
      %
      % OUTPUTS
      %  imgs       - loaded img objects
      ids = values(coco.inds.imgIdsMap,num2cell(ids));
      imgs = coco.data.images([ids{:}]);
    end
    
    function hs = showAnns( coco, anns )
      % Display the specified annotations.
      %
      % USAGE
      %  hs = coco.showAnns( anns )
      %
      % INPUTS
      %  anns       - annotations to display
      %
      % OUTPUTS
      %  hs         - handles to segment graphic objects
      n=length(anns); if(n==0), return; end
      if( strcmp(coco.data.type,'instances') )
        S={anns.segmentation}; hs=zeros(10000,1); k=0; hold on;
        pFill={'FaceAlpha',.4,'LineWidth',3};
        for i=1:n, C=rand(1,3);
          if(anns(i).iscrowd), M=double(coco.decodeMask(S{i})); k=k+1;
            hs(k)=imagesc(cat(3,M*.01,M*.65,M*.40),'Alphadata',M*.5);
          else for j=1:length(S{i}), P=S{i}{j}+1; k=k+1;
              hs(k)=fill(P(1:2:end),P(2:2:end),C,pFill{:}); end
          end
        end
        hs=hs(1:k); hold off;
      elseif( strcmp(coco.data.type,'captions') )
        S={anns.caption};
        for i=1:n, S{i}=[int2str(i) ') ' S{i} '\newline']; end
        S=[S{:}]; title(S,'FontSize',12);
      end
      hold off;
    end
  end
  
  methods( Static )
    function M = decodeMask( R )
      % Decode binary mask M encoded via run-length encoding.
      %
      % USAGE
      %  M = CocoApi.decodeMask( R )
      %
      % INPUTS
      %  R          - run-length encoding of binary mask
      %
      % OUTPUTS
      %  M          - decoded binary mask
      M=zeros(R.size,'uint8'); k=1; n=length(R.counts);
      for i=2:2:n, for j=1:R.counts(i-1), k=k+1; end;
        for j=1:R.counts(i), M(k)=1; k=k+1; end; end
    end
    
    function R = encodeMask( M )
      % Encode binary mask M using run-length encoding.
      %
      % USAGE
      %  R = CocoApi.encodeMask( M )
      %
      % INPUTS
      %  M          - binary mask to encode
      %
      % OUTPUTS
      %  R          - run-length encoding of binary mask
      R.size=size(M); if(isempty(M)), R.counts=[]; return; end
      D=M(2:end)~=M(1:end-1); is=uint32([find(D) numel(M)]);
      R.counts=[is(1) diff(is)]; if(M(1)==1), R.counts=[0 R.counts]; end
    end
    
    function M = segToMask( S, h, w )
      % Convert polygon segmentation to binary mask.
      %
      % USAGE
      %  M = CocoApi.segToMask( S, h, w )
      %
      % INPUTS
      %  S          - polygon segmentation mask
      %  h          - target mask height
      %  w          - target mask width
      %
      % OUTPUTS
      %  M          - binary mask
      P=S{1}+1; M=poly2mask(P(1:2:end),P(2:2:end),h,w); n=length(S);
      for i=2:n, P=S{i}+1; M=M | poly2mask(P(1:2:end),P(2:2:end),h,w); end
    end
  end
  
end
