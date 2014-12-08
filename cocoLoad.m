function coco = cocoLoad( annName, imgDir )
% Load JSON annotation file and prepare image index.
%
% USAGE
%  data = cocoLoad( annName, imgDir );
%
% INPUTS
%  annName   - string specifying annotation file name
%
% OUTPUTS
%  coco       - loaded annotations in Matlab object [NEED TO DOCUMENT]
%
% EXAMPLE
%  coco = cocoLoad('data/instances_val2014.json','data/val2014');
%
% See also cocoDemo, cocoLoad>getImgIds, bbGt>create
%
% Microsoft COCO Toolbox.      Version 0.90
% Data, paper, and tutorials available at:  http://mscoco.org/
% Code written by Piotr Dollar and Tsung-Yi Lin, 2014.
% Licensed under the Simplified BSD License [see private/bsd.txt]

initialize( annName, imgDir ); return;

  function initialize( annName, imgDir )
    % load annotations
    fprintf('loading annotations... '); t=clock;
    coco = gason(fileread(annName));
    coco.annName=annName; coco.imgDir=imgDir;
    fprintf('DONE! (t=%0.2fs)\n',etime(clock,t));
    % create useful indexes
    fprintf('initializing data structures... '); t=clock;
    coco.indexes.imgIds     = [coco.images.id];
    coco.indexes.annIds     = [coco.instances.id];
    coco.indexes.annImgIds  = [coco.instances.image_id];
    coco.indexes.annCatIds  = [coco.instances.category_id];
    coco.indexes.annAreas   = [coco.instances.area];
    % create mappings from ids to inds
    i=coco.indexes.imgIds; coco.maps.imgIds=containers.Map(i,1:length(i));
    i=coco.indexes.annIds; coco.maps.annIds=containers.Map(i,1:length(i));
    coco.maps.catIds=containers.Map(getCats(),[coco.categories.id]);
    fprintf('DONE! (t=%0.2fs)\n',etime(clock,t));
    % create functions handles
    coco.getCats    = @getCats;
    coco.getCatIds  = @getCatIds;
    coco.getImgIds  = @getImgIds;
    coco.getAnnIds  = @getAnnIds;
    coco.loadImg    = @loadImg;
    coco.loadAnns   = @loadAnns;
  end

  function cats = getCats()
    % Get list of all category names.
    cats={coco.categories.name};
  end

  function ids = getCatIds( cats )
    % Get cat ids corresponding to cell array of category names.
    ids=cell2mat(values(coco.maps.catIds,cats));
  end

  function ids = getImgIds( varargin )
    % Get image ids that satisfy the filter conditions.
    %
    % USAGE
    %  ids = coco.getImgIds( params );
    %
    % INPUTS
    %  params     - filtering parameters (struct or name/value pairs)
    %   .imgIds     - [] select images with given ids (if [] keeps all)
    %   .catIds     - [] select images that contain all given categories
    %
    % OUTPUTS
    %  ids        - integer array of image ids
    p = getPrmDflt(varargin,{'imgIds',[],'catIds',[]},1);
    ids = coco.indexes.imgIds; n = length(p.catIds);
    if(~isempty(p.imgIds)), ids=intersect(ids,p.imgIds); end
    iIds = coco.indexes.annImgIds; cIds = coco.indexes.annCatIds;
    for i=1:n, ids=intersect(ids,unique(iIds(cIds==p.catIds(i)))); end
  end

  function ids = getAnnIds( varargin )
    % Get ann ids that satisfy the filter conditions.
    %
    % USAGE
    %  ids = coco.getAnns( params );
    %
    % INPUTS
    %  params     - filtering parameters (struct or name/value pairs)
    %   .imgIds     - [] select anns for given images (if [] keeps all)
    %   .catIds     - [] select anns for given category (e.g. 0)
    %   .areaRange  - [] select anns in given area range (e.g. [0 inf])
    %
    % OUTPUTS
    %  anns       - integer array of ann ids
    %
    % See also cocoLoad
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

  function I = loadImg( id )
    % Load image with specified id.
    img = coco.images(coco.maps.imgIds(id));
    I = imread([coco.imgDir filesep img.file_name]);
  end

  function anns = loadAnns( ids )
    % Load anns with specified id.
    inds=values(coco.maps.annIds,num2cell(ids));
    anns = coco.instances([inds{:}]);
  end

end
