function coco = cocoLoad( annName, imageDir )
% Load JSON annotation file and prepare image index.
%
% USAGE
%  data = cocoLoad( annName, imageDir );
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
% See also cocoLoad>getImageIds
%
% Microsoft COCO Toolbox.      Version 0.90
% Data, paper, and tutorials available at:  http://mscoco.org/
% Code written by Piotr Dollar and Tsung-Yi Lin, 2014.
% Licensed under the Simplified BSD License [see private/bsd.txt]

% load annotations
fprintf('loading annotations... '); start=clock;
coco = gason(fileread(annName));
fprintf('DONE! (t=%0.2fs)\n',etime(clock,start));

% starting initialization
fprintf('initializing data structures... '); start=clock;

% store image directory and get type
coco.annName = annName;
coco.imageDir = imageDir;

% create useful indexes
coco.indexes.imageIds             = [coco.images.id];
coco.indexes.instanceIds          = [coco.instances.id];
coco.indexes.instanceImageIds     = [coco.instances.image_id];
coco.indexes.instanceCategoryIds  = [coco.instances.category_id];
coco.indexes.instanceAreas        = [coco.instances.area];

% create mappings from ids to inds
t=coco.indexes.imageIds;
coco.maps.imageIds=containers.Map(t,1:length(t));
t=coco.indexes.instanceIds;
coco.maps.instanceIds=containers.Map(t,1:length(t));

% create functions handles
coco.getImageIds = @getImageIds;
coco.loadImage   = @loadImage;
coco.getAnnIds   = @getAnnIds;
coco.loadAnns    = @loadAnns;

% done with initialization
fprintf('DONE! (t=%0.2fs)\n',etime(clock,start));

  function ids = getImageIds( varargin )
    % Get image ids that satisfy the filter conditions.
    %
    % USAGE
    %  ids = getImageIds( params );
    %
    % INPUTS
    %  params       - filtering parameters (struct or name/value pairs)
    %   .imageIds     - [] select images with given ids (if [] keeps all)
    %   .categoryIds  - [] select images that contain all given categories
    %
    % OUTPUTS
    %  ids          - an array of image ids satisfying filter conditions
    
    % get filtering parameters
    dfs = { 'imageIds',[], 'categoryIds',[] };
    filters = getPrmDflt(varargin,dfs,1);
    
    % get list of all image ids
    ids = coco.indexes.imageIds;
    
    % filter by imageIds
    fIds = filters.imageIds;
    if(~isempty(fIds)), ids=intersect(ids,fIds); end
    
    % filter by categoryIds
    fIds = filters.categoryIds; n = length(fIds);
    iIds = coco.indexes.instanceImageIds;
    cIds = coco.indexes.instanceCategoryIds;
    for i=1:n, ids=intersect(ids,unique(iIds(cIds==fIds(i)))); end
  end

  function I = loadImage( id )
    % Load image with specified id.
    img = coco.images(coco.maps.imageIds(id));
    I = imread([coco.imageDir filesep img.file_name]);
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
    %   .imageIds     - [] select anns for given images (if [] keeps all)
    %   .categoryId   - [] select anns for given category (e.g. 0)
    %   .areaRange    - [] select anns in given area range (e.g. [0 inf])
    %
    % OUTPUTS
    %  anns         - an array of anns satisfying filter conditions
    
    % get filtering parameters
    dfs = { 'imageIds',[], 'categoryId',[], 'areaRange',[] };
    p = getPrmDflt(varargin,dfs,1);
    
    % all ids
    ids = coco.indexes.instanceIds;
    imgIds = coco.indexes.instanceImageIds;
    catIds = coco.indexes.instanceCategoryIds;
    keep = true(1,length(ids));
    
    % filter by categoryId
    if(~isempty(p.categoryId)), keep=keep & p.categoryId==catIds; end
    
    % filter by imageIds
    if( ~isempty(p.imageIds) )
      kp = false(1,length(imgIds));
      for i=1:length(p.imageIds), kp=kp|imgIds==p.imageIds(i); end
      keep = keep & kp;
    end
    
    % filter by areaRange
    if(~isempty(p.areaRange)), keep = keep ...
        & coco.indexes.instanceAreas>=p.areaRange(1) ...
        & coco.indexes.instanceAreas<=p.areaRange(2);
    end
    
    % return kept subset of ids
    ids=ids(keep);
  end

  function anns = loadAnns( ids )
    % Load anns with specified id.
    inds=values(coco.maps.instanceIds,num2cell(ids));
    anns = coco.instances([inds{:}]);
  end

end
