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
  %  data = gason(fileread(annName));
  % Using the API provides additional utility functions.
  %
  % The following utility functions are provided:
  %  CocoApi    - Load annotation file and prepare data structures.
  %  getAnnIds  - Get annotation ids that satisfy given filter conditions.
  %  getCatIds  - Get category ids corresponding to category names.
  %  getCats    - Get list of all category names.
  %  getImgIds  - Get image ids that satisfy given filter conditions.
  %  loadAnns   - Load anns with the specified ids.
  %  loadImg    - Load image with the specified id.
  % Help on each functions can be accessed by: "help CocoApi>function".
  %
  % See also cocoDemo, CocoApi>CocoApi, CocoApi>getCats, CocoApi>getCatIds
  % CocoApi>getImgIds, CocoApi>getAnnIds, CocoApi>loadImg, CocoApi>loadAnns
  %
  % Microsoft COCO Toolbox.      Version 0.90
  % Data, paper, and tutorials available at:  http://mscoco.org/
  % Code written by Piotr Dollar and Tsung-Yi Lin, 2014.
  % Licensed under the Simplified BSD License [see private/bsd.txt]
  
  properties
    data      % COCO annotation data
    indexes   % indexes for fast access
    maps      % mapping for fast access
  end
  
  methods
    
    function coco = CocoApi( annName, imgDir )
      % Load annotation file and prepare data structures.
      %
      % USAGE
      %  coco = CocoApi( annName, imgDir )
      %
      % INPUTS
      %  annName   - string specifying annotation file name
      %  imgDir    - directory containing images
      %
      % OUTPUTS
      %  coco      - initialized coco object
      fprintf('loading annotations... '); t=clock;
      C=coco; C.data = gason(fileread(annName));
      C.data.annName=annName; C.data.imgDir=imgDir;
      fprintf('DONE! (t=%0.2fs)\n',etime(clock,t));
      fprintf('initializing data structures... '); t=clock;
      C.indexes.imgIds     = [C.data.images.id];
      C.indexes.annIds     = [C.data.instances.id];
      C.indexes.annImgIds  = [C.data.instances.image_id];
      C.indexes.annCatIds  = [C.data.instances.category_id];
      C.indexes.annAreas   = [C.data.instances.area];
      i=C.indexes.imgIds;  C.maps.imgIds=containers.Map(i,1:length(i));
      i=C.indexes.annIds;  C.maps.annIds=containers.Map(i,1:length(i));
      i=C.data.categories; C.maps.catIds=containers.Map({i.name},[i.id]);
      fprintf('DONE! (t=%0.2fs)\n',etime(clock,t)); coco=C;
    end
    
    function cats = getCats( coco )
      % Get list of all category names.
      %
      % USAGE
      %  cats = coco.getCats()
      %
      % INPUTS
      %
      % OUTPUTS
      %  cats       - string array of category names
      cats={coco.data.categories.name};
    end
    
    function ids = getCatIds( coco, cats )
      % Get category ids corresponding to category names.
      %
      % USAGE
      %  ids = coco.getCatIds( cats )
      %
      % INPUTS
      %  cats       - cell array of category names
      %
      % OUTPUTS
      %  ids        - integer array of img ids
      ids=cell2mat(values(coco.maps.catIds,cats));
    end
    
    function ids = getImgIds( coco, varargin )
      % Get image ids that satisfy given filter conditions.
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
      p = getPrmDflt(varargin,{'imgIds',[],'catIds',[]},1);
      ids = coco.indexes.imgIds; n = length(p.catIds);
      if(~isempty(p.imgIds)), ids=intersect(ids,p.imgIds); end
      iIds = coco.indexes.annImgIds; cIds = coco.indexes.annCatIds;
      for i=1:n, ids=intersect(ids,unique(iIds(cIds==p.catIds(i)))); end
    end
    
    function ids = getAnnIds( coco, varargin )
      % Get annotation ids that satisfy given filter conditions.
      %
      % USAGE
      %  ids = coco.getAnnIds( params )
      %
      % INPUTS
      %  params     - filtering parameters (struct or name/value pairs)
      %               setting any filter to [] skips that filter
      %   .imgIds     - [] get anns for given imgs
      %   .catIds     - [] get anns for given cats
      %   .areaRange  - [] get anns for given area range (e.g. [0 inf])
      %
      % OUTPUTS
      %  anns       - integer array of ann ids
      p = getPrmDflt(varargin,{'imgIds',[],'catIds',[],'areaRange',[]},1);
      ids = coco.indexes.annIds; K = true(1,length(ids));
      if( ~isempty(p.imgIds) ), K = K & ...
          ismember( coco.indexes.annImgIds, p.imgIds ); end
      if( ~isempty(p.catIds) ), K = K & ...
          ismember( coco.indexes.annCatIds, p.catIds ); end
      if( ~isempty(p.areaRange) ), v=coco.indexes.annAreas; K = K & ...
          v>=p.areaRange(1) & v<=p.areaRange(2); end
      ids=ids(K);
    end
    
    function I = loadImg( coco, id )
      % Load image with the specified id.
      %
      % USAGE
      %  I = coco.loadImg( id )
      %
      % INPUTS
      %  id         - integer id specifying image
      %
      % OUTPUTS
      %  I          - loaded image
      img = coco.data.images(coco.maps.imgIds(id));
      I = imread([coco.data.imgDir filesep img.file_name]);
    end
    
    function anns = loadAnns( coco, ids )
      % Load anns with the specified ids.
      %
      % USAGE
      %  anns = coco.loadAnns( ids )
      %
      % INPUTS
      %  ids        - integer id specifying annotations
      %
      % OUTPUTS
      %  anns       - loaded annotations
      inds = values(coco.maps.annIds,num2cell(ids));
      anns = coco.data.instances([inds{:}]);
    end
  end
  
end
