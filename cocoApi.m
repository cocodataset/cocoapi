function varargout = cocoApi( action, varargin )
% Interface for accessing the Microsoft COCO dataset.
%
% Microsoft COCO is a large image dataset designed for object detection,
% segmentation, and caption generation. cocoApi.m is a Matlab API that
% assists in loading, parsing and visualizing the annotations in COCO.
% Please visit http://mscoco.org/ for more information on the COCO dataset,
% including for the data, paper, and tutorials. The exact format of the
% annotations is likewise described on the COCO website. For example usage
% of the cocoApi please see cocoDemo.m. In addition to this API, please
% download both the COCO images and annotations in order to run the demo.
%
% cocoApi contains a number of utility functions, accessed using:
%  outputs = cocoApi( 'action', inputs );
% The list of functions and help for each is given below. Also, help on
% individual subfunctions can be accessed by: "help cocoApi>action". Before
% accessing any other actions, make sure to call cocoApi('initialize').
%
% Note that after initialization, the 'coco' data structure contains all of
% the data in the annotation file. An alternative to using the cocoApi is
% to access and use the fields in the 'coco' struct directly. To avoid
% using the API altogether, the JSON annotation file can be loaded via:
%  coco = gason(fileread(annName));
% Note that the coco strcut created by 'initialize' contains extra fields.
%
% USAGE
%  varargout = cocoApi( action, varargin );
%
% ACTIONS
%  Load annotation file and prepare data structures:
%   coco = cocoApi( 'initialize', annName, imgDir );
%  Get list of all category names:
%   cats = cocoApi( 'getCats' );
%  Get category ids corresponding to category names:
%   ids = cocoApi( 'getCatIds', cats )
%  Get imgage ids that satisfy given filter conditions:
%   ids = cocoApi( 'getImgIds', params );
%  Get annotation ids that satisfy given filter conditions:
%   ids = cocoApi( 'getAnnIds', params );
%  Load image with the specified id:
%   I = cocoApi( 'loadImg', id );
%  Load anns with the specified ids:
%   anns = cocoApi( 'loadAnns', ids );
%
% INPUTS
%  action     - string specifying action
%  varargin   - depends on action, see above
%
% OUTPUTS
%  varargout  - depends on action, see above
%
% EXAMPLE
%
% See also cocoDemo, cocoApi>initialize, cocoApi>getCats, cocoApi>getCatIds
% cocoApi>getImgIds, cocoApi>getAnnIds, cocoApi>loadImg, cocoApi>loadAnns
%
% Microsoft COCO Toolbox.      Version 0.90
% Data, paper, and tutorials available at:  http://mscoco.org/
% Code written by Piotr Dollar and Tsung-Yi Lin, 2014.
% Licensed under the Simplified BSD License [see private/bsd.txt]

%#ok<*DEFNU>
persistent coco;
if(strcmp(action,'initialize'))
  coco=initialize(varargin{:}); varargout={coco};
else
  if(isempty(coco)), error('coco not initialized.'); end
  varargout = cell(1,max(1,nargout));
  [varargout{:}] = feval(action,coco,varargin{:});
end

end

function coco = initialize( annName, imgDir )
% Load annotation file and prepare data structures.
%
% USAGE
%  coco = cocoApi( 'initialize', annName, imgDir );
%
% INPUTS
%  annName   - string specifying annotation file name
%  imgDir    - [DOCUMENT]
%
% OUTPUTS
%  coco      - loaded annotations in Matlab object [DOCUMENT]

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
coco.maps.catIds=containers.Map(getCats(coco),[coco.categories.id]);
% bind functions (don't use "coco.getCats=@()getCats(coco)", is slow!)
coco.getCats    = bind('getCats');
coco.getCatIds  = bind('getCatIds');
coco.getImgIds  = bind('getImgIds');
coco.getAnnIds  = bind('getAnnIds');
coco.loadImg    = bind('loadImg');
coco.loadAnns   = bind('loadAnns');
fprintf('DONE! (t=%0.2fs)\n',etime(clock,t));
end

function f = bind( action )
% Helper function for binding, absolutely necessary to define explicitly.
f = @(varargin) cocoApi(action,varargin{:});
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
cats={coco.categories.name};
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
%
% See also cocoApi
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
img = coco.images(coco.maps.imgIds(id));
I = imread([coco.imgDir filesep img.file_name]);
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
inds=values(coco.maps.annIds,num2cell(ids));
anns = coco.instances([inds{:}]);
end
