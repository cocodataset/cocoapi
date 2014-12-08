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
  %  Load annotation file and prepare data structures:
  %   coco = CocoApi( annName, imgDir );
  %  Get list of all category names:
  %   cats = coco.getCats();
  %  Get category ids corresponding to category names:
  %   ids = coco.getCatIds( cats )
  %  Get imgage ids that satisfy given filter conditions:
  %   ids = coco.getImgIds( params );
  %  Get annotation ids that satisfy given filter conditions:
  %   ids = coco.getAnnIds( params );
  %  Load image with the specified id:
  %   I = coco.loadImg( id );
  %  Load anns with the specified ids:
  %   anns = coco.loadAnns( ids );
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
      %  coco = CocoApi( 'initialize', annName, imgDir );
      %
      % INPUTS
      %  annName   - string specifying annotation file name
      %  imgDir    - directory containing images
      %
      % OUTPUTS
      %  coco      - initialized coco object
      
      % load annotations
      fprintf('loading annotations... '); t=clock;
      coco.data = gason(fileread(annName));
      coco.data.annName=annName; coco.data.imgDir=imgDir;
      fprintf('DONE! (t=%0.2fs)\n',etime(clock,t));
      % create useful indexes
      fprintf('initializing data structures... '); t=clock;
      coco.indexes.imgIds     = [coco.data.images.id];
      coco.indexes.annIds     = [coco.data.instances.id];
      coco.indexes.annImgIds  = [coco.data.instances.image_id];
      coco.indexes.annCatIds  = [coco.data.instances.category_id];
      coco.indexes.annAreas   = [coco.data.instances.area];
      % create mappings from ids to inds
      i=coco.indexes.imgIds; coco.maps.imgIds=containers.Map(i,1:length(i));
      i=coco.indexes.annIds; coco.maps.annIds=containers.Map(i,1:length(i));
      coco.maps.catIds=containers.Map({coco.data.categories.name},...
        [coco.data.categories.id]);
      fprintf('DONE! (t=%0.2fs)\n',etime(clock,t));
    end
    
    function cats = getCats( coco )
      % Get list of all category names.
      %
      % USAGE
      %  cats = coco.getCats();
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
      %   .imgIds     - [] select images with given ids (if [] keeps all)
      %   .catIds     - [] select images that contain all given categories
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
      %   .imgIds     - [] select anns for given images (if [] keeps all)
      %   .catIds     - [] select anns for given category (e.g. 0)
      %   .areaRange  - [] select anns in given area range (e.g. [0 inf])
      %
      % OUTPUTS
      %  anns       - integer array of ann ids
      p = getPrmDflt(varargin,{'imgIds',[],'catIds',[],'areaRange',[]},1);
      ids = coco.indexes.annIds; keep = true(1,length(ids));
      if(~isempty(p.catIds))
        catIds=coco.indexes.annCatIds; k=false(1,length(catIds));
        for i=1:length(p.catIds), k=k|catIds==p.catIds(i); end
        keep = keep & k;
      end
      if(~isempty(p.imgIds))
        imgIds=coco.indexes.annImgIds; k=false(1,length(imgIds));
        for i=1:length(p.imgIds), k=k|imgIds==p.imgIds(i); end
        keep = keep & k;
      end
      if(~isempty(p.areaRange)), keep = keep ...
          & coco.indexes.annAreas>=p.areaRange(1) ...
          & coco.indexes.annAreas<=p.areaRange(2);
      end
      ids=ids(keep);
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
      %  anns = coco.loadAnns( ids );
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
