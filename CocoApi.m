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
  %  getAnnIds  - Get ann ids that satisfy given filter conditions.
  %  getCatIds  - Get cat ids that satisfy given filter conditions.
  %  getImgIds  - Get img ids that satisfy given filter conditions.
  %  loadAnns   - Load anns with the specified ids.
  %  loadCats   - Load cats with the specified ids.
  %  loadImgs   - Load imgs with the specified ids.
  %  showAnns   - Display the specified annotations.
  % Throught the API "ann"=annotation, "cat"=category, and "img"=image.
  % Help on each functions can be accessed by: "help CocoApi>function".
  %
  % See also cocoDemo, CocoApi>CocoApi, CocoApi>getAnnIds,
  % CocoApi>getCatIds, CocoApi>getImgIds, CocoApi>loadAnns,
  % CocoApi>loadCats, CocoApi>loadImgs, CocoApi>showAnns
  %
  % Microsoft COCO Toolbox.      Version 0.90
  % Data, paper, and tutorials available at:  http://mscoco.org/
  % Code written by Piotr Dollar and Tsung-Yi Lin, 2014.
  % Licensed under the Simplified BSD License [see private/bsd.txt]
  
  properties
    imgDir  % directory containing images
    data    % COCO annotation data structure
    inds    % data structures for fast indexing
  end
  
  methods
    function coco = CocoApi( imgDir, annFile )
      % Load COCO annotation file and prepare data structures.
      %
      % USAGE
      %  coco = CocoApi( imgDir, annFile )
      %
      % INPUTS
      %  imgDir    - directory containing imgs
      %  annFile   - COCO annotation filename
      %
      % OUTPUTS
      %  coco      - initialized coco object
      fprintf('Loading and preparing annotations... '); clk=clock;
      c=coco; c.imgDir=imgDir; c.data=gason(fileread(annFile));
      if( strcmp(c.data.type,'instances') )
        anns = c.data.instances; t = [c.data.categories.id]';
        c.inds.catIdsMap = containers.Map(t,1:length(t));
        c.inds.annCatIds = [anns.category_id]';
        c.inds.annAreas = [anns.area]';
      elseif( strcmp(c.data.type,'captions') )
        anns = c.data.captions;
      end
      c.inds.annIds = [anns.id]'; t=c.inds.annIds;
      c.inds.annIdsMap = containers.Map(t,1:length(t));
      c.inds.annImgIds = [anns.image_id]';
      c.inds.imgIds = [c.data.images.id]'; t=c.inds.imgIds;
      c.inds.imgIdsMap = containers.Map(t,1:length(t));
      c.inds.imgAnnIdsMap = makeImgAnnIdsMap( c.inds );
      fprintf('DONE (t=%0.2fs).\n',etime(clock,clk)); coco=c;
      
      function map = makeImgAnnIdsMap( inds )
        % Find map from imgIds to annIds associated with each imgId.
        is = values(inds.imgIdsMap,num2cell(inds.annImgIds));
        is=[is{:}]; m=length(is); n=length(inds.imgIds); k=zeros(1,n);
        for i=1:m, j=is(i); k(j)=k(j)+1; end; a=zeros(n,max(k)); k(:)=0;
        for i=1:m, j=is(i); k(j)=k(j)+1; a(j,k(j))=inds.annIds(i); end
        map = containers.Map('KeyType','double','ValueType','any');
        for j=1:n, map(inds.imgIds(j))=a(j,1:k(j)); end
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
      def = {'imgIds',[],'catIds',[]};
      p = getPrmDflt(varargin,def,1); ids = coco.inds.imgIds;
      if(~isempty(p.imgIds)), ids=intersect(ids,p.imgIds); end
      for i=1:length(p.catIds), ids=intersect(ids,unique(...
          coco.inds.annImgIds(coco.inds.annCatIds==p.catIds(i)))); end
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
      %
      % OUTPUTS
      %  ids        - integer array of ann ids
      def = {'imgIds',[],'catIds',[],'areaRng',[]};
      [imgIds,catIds,ar] = getPrmDflt(varargin,def,1);
      if( length(imgIds)==1 )
        t = coco.loadAnns(coco.inds.imgAnnIdsMap(imgIds));
        if(~isempty(catIds)), t = t(ismember([t.category_id],catIds)); end
        if(~isempty(ar)), a=[t.area]; t = t(a>=ar(1) & a<=ar(2)); end
        ids = [t.id];
      else
        ids=coco.inds.annIds; K = true(length(ids),1); t = coco.inds;
        if(~isempty(imgIds)), K = K & ismember(t.annImgIds,imgIds); end
        if(~isempty(catIds)), K = K & ismember(t.annCatIds,catIds); end
        if(~isempty(ar)), a=t.annAreas; K = K & a>=ar(1) & a<=ar(2); end
        ids=ids(K);
      end
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
    
    function imgs = loadImgs( coco, ids, readImg )
      % Load imgs with the specified ids.
      %
      % USAGE
      %  imgs = coco.loadImgs( ids, [readImg] )
      %
      % INPUTS
      %  ids        - integer ids specifying imgs
      %  readImg    - [false] if true load img data
      %
      % OUTPUTS
      %  imgs       - loaded img objects
      ids = values(coco.inds.imgIdsMap,num2cell(ids));
      imgs = coco.data.images([ids{:}]);
      if(nargin<=2 || readImg==0), return; end
      for i=1:length(imgs), f=[coco.imgDir filesep imgs(i).file_name];
        imgs(i).image = imread(f); end
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
      if( strcmp(coco.data.type,'instances') )
        anns = coco.data.instances([ids{:}]);
      elseif(strcmp( coco.data.type,'captions') )
        anns = coco.data.captions([ids{:}]);
      end
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
        for i=1:n, clr=rand(1,3); for j=1:length(S{i}), k=k+1; ...
              hs(k)=fill(S{i}{j}(1:2:end),S{i}{j}(2:2:end),clr); end; end
        hs=hs(1:k); set(hs,'FaceAlpha',.4,'LineWidth',3); hold off;
      elseif( strcmp(coco.data.type,'captions') )
        S={anns.caption};
        for i=1:n, S{i}=[int2str(i) ') ' S{i} '\newline']; end
        S=[S{:}]; title(S,'FontSize',12);
      end
      hold off;
    end
  end
  
end
