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
% See also
%
% Microsoft COCO Toolbox.      Version 0.90
% Data, paper, and tutorials available at:  http://mscoco.org/
% Code written by Piotr Dollar and Tsung-Yi Lin, 2014.
% Licensed under the Simplified BSD License [see private/bsd.txt]

% load annotations
fprintf('loading annotations... '); start=clock;
coco = gason(fileread(annName));
fprintf('DONE! (t=%0.2fs)\n',etime(clock,start));

% store image directory and get type
fprintf('initializing data structures... '); start=clock;
coco.annName = annName;
coco.imgDir = imgDir;

% create useful indexes
coco.indexes.imgIds     = [coco.images.id];
coco.indexes.annIds     = [coco.instances.id];
coco.indexes.annImgIds  = [coco.instances.image_id];
coco.indexes.annCatIds  = [coco.instances.category_id];
coco.indexes.annAreas   = [coco.instances.area];

% create mappings from ids to inds
t=coco.indexes.imgIds; coco.maps.imgIds=containers.Map(t,1:length(t));
t=coco.indexes.annIds; coco.maps.annIds=containers.Map(t,1:length(t));
fprintf('DONE! (t=%0.2fs)\n',etime(clock,start));

% create functions handles
coco.getImgIds  = @getImgIds;
coco.getAnnIds  = @getAnnIds;
coco.loadImg    = @loadImg;
coco.loadAnns   = @loadAnns;

  function ids = getImgIds( varargin )
    % Get image ids that satisfy the filter conditions.
    %
    % USAGE
    %  ids = getImgIds( params );
    %
    % INPUTS
    %  params       - filtering parameters (struct or name/value pairs)
    %   .imgIds       - [] select images with given ids (if [] keeps all)
    %   .catIds       - [] select images that contain all given categories
    %
    % OUTPUTS
    %  ids          - an array of image ids satisfying filter conditions
    
    % get filtering parameters
    dfs = { 'imgIds',[], 'catIds',[] };
    filters = getPrmDflt(varargin,dfs,1);
    
    % get list of all image ids
    ids = coco.indexes.imgIds;
    
    % filter by imgIds
    fIds = filters.imgIds;
    if(~isempty(fIds)), ids=intersect(ids,fIds); end
    
    % filter by catIds
    fIds = filters.catIds; n = length(fIds);
    iIds = coco.indexes.annImgIds;
    cIds = coco.indexes.annCatIds;
    for i=1:n, ids=intersect(ids,unique(iIds(cIds==fIds(i)))); end
  end

  function ids = getAnnIds( varargin )
    % Get annotation objects that satisfy the filter conditions.
    %
    % USAGE
    %  instances = getAnns( coco, params );
    %
    % INPUTS
    %  coco         - data structure containing loaded COCO annotations
    %  params       - filtering parameters (struct or name/value pairs)
    %   .imgIds       - [] select anns for given images (if [] keeps all)
    %   .catId        - [] select anns for given category (e.g. 0)
    %   .areaRange    - [] select anns in given area range (e.g. [0 inf])
    %
    % OUTPUTS
    %  anns         - an array of anns satisfying filter conditions
    
    % get filtering parameters
    dfs = { 'imgIds',[], 'catId',[], 'areaRange',[] };
    p = getPrmDflt(varargin,dfs,1);
    
    % all ids
    ids = coco.indexes.annIds;
    imgIds = coco.indexes.annImgIds;
    catIds = coco.indexes.annCatIds;
    keep = true(1,length(ids));
    
    % filter by catId
    if(~isempty(p.catId)), keep=keep & p.catId==catIds; end
    
    % filter by imgIds
    if( ~isempty(p.imgIds) )
      kp = false(1,length(imgIds));
      for i=1:length(p.imgIds), kp=kp|imgIds==p.imgIds(i); end
      keep = keep & kp;
    end
    
    % filter by areaRange
    if(~isempty(p.areaRange)), keep = keep ...
        & coco.indexes.annAreas>=p.areaRange(1) ...
        & coco.indexes.annAreas<=p.areaRange(2);
    end
    
    % return kept subset of ids
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
