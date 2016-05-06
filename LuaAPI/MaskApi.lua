--[[----------------------------------------------------------------------------

Interface for manipulating masks stored in RLE format.

For an overview of RLE please see http://mscoco.org/dataset/#download.
Additionally, more detailed information can be found in the Matlab MaskApi.m:
  https://github.com/pdollar/coco/blob/master/MatlabAPI/MaskApi.m

The following API functions are defined:
  encode - Encode binary masks using RLE.
  decode - Decode binary masks encoded via RLE.
  merge  - Compute union or intersection of encoded masks.
  iou    - Compute intersection over union between masks.
  area   - Compute area of encoded masks.
  toBbox - Get bounding boxes surrounding encoded masks.
  frBbox - Convert bounding boxes to encoded masks.
  frPoly - Convert polygon to encoded mask.

Usage:
  Rs     = MaskApi.encode( masks )
  masks  = MaskApi.decode( Rs )
  R      = MaskApi.merge( Rs, [intersect=false] )
  o      = MaskApi.iou( dt, gt, [iscrowd=false] )
  a      = MaskApi.area( Rs )
  bbs    = MaskApi.toBbox( Rs )
  Rs     = MaskApi.frBbox( bbs, h, w )
  R      = MaskApi.frPoly( poly, h, w )
  img    = MaskApi.drawMasks( img, masks, [maxn=n], [alpha=.4], [clrs] )
For detailed usage information please see cocoDemo.lua (coming soon).

In the API the following formats are used:
  R,Rs   - [table] Run-length encoding of binary mask(s)
  masks  - [nxhxw] Binary mask(s)
  bbs    - [nx4] Bounding box(es) stored as [x y w h]
  poly   - Polygon stored as {[x1 y1 x2 y2...],[x1 y1 ...],...}
  dt,gt  - May be either bounding boxes or encoded masks
Both poly and bbs are 0-indexed (bbox=[0 0 1 1] encloses first pixel).

Common Objects in COntext (COCO) Toolbox.      version 3.0
Data, paper, and tutorials available at:  http://mscoco.org/
Code written by Pedro O. Pinheiro and Piotr Dollar, 2016.
Licensed under the Simplified BSD License [see coco/license.txt]

------------------------------------------------------------------------------]]

local ffi = require 'ffi'
local coco = require 'coco.env'

coco.MaskApi = {}
local MaskApi = coco.MaskApi

coco.libmaskapi = ffi.load(package.searchpath('libmaskapi',package.cpath))
local libmaskapi = coco.libmaskapi

--------------------------------------------------------------------------------

MaskApi.encode = function( masks )
  local n, h, w = masks:size(1), masks:size(2), masks:size(3)
  masks = masks:type('torch.ByteTensor'):transpose(2,3)
  local data = masks:contiguous():data()
  local Qs = MaskApi._rlesInit(n)
  libmaskapi.rleEncode(Qs[0],data,h,w,n)
  return MaskApi._rlesToLua(Qs,n)
end

MaskApi.decode = function( Rs )
  local Qs, n, h, w = MaskApi._rlesFrLua(Rs)
  local masks = torch.ByteTensor(n,w,h):zero():contiguous()
  libmaskapi.rleDecode(Qs,masks:data(),n)
  MaskApi._rlesFree(Qs,n)
  return masks:transpose(2,3)
end

MaskApi.merge = function( Rs, intersect )
  intersect = intersect or 0
  local Qs, n, h, w = MaskApi._rlesFrLua(Rs)
  local Q = MaskApi._rlesInit(1)
  libmaskapi.rleMerge(Qs,Q,n,intersect)
  MaskApi._rlesFree(Qs,n)
  return MaskApi._rlesToLua(Q,1)[1]
end

MaskApi.iou = function( dt, gt, iscrowd )
  if not iscrowd then iscrowd = NULL else
    iscrowd = iscrowd:type('torch.ByteTensor'):contiguous():data()
  end
  if torch.isTensor(gt) and torch.isTensor(dt) then
    local nDt, k = dt:size(1), dt:size(2); assert(k==4)
    local nGt, k = gt:size(1), gt:size(2); assert(k==4)
    local dDt = dt:type('torch.DoubleTensor'):contiguous():data()
    local dGt = gt:type('torch.DoubleTensor'):contiguous():data()
    local o = torch.DoubleTensor(nGt,nDt):contiguous()
    libmaskapi.bbIou(dDt,dGt,nDt,nGt,iscrowd,o:data())
    return o:transpose(1,2)
  else
    local qDt, nDt = MaskApi._rlesFrLua(dt)
    local qGt, nGt = MaskApi._rlesFrLua(gt)
    local o = torch.DoubleTensor(nGt,nDt):contiguous()
    libmaskapi.rleIou(qDt,qGt,nDt,nGt,iscrowd,o:data())
    MaskApi._rlesFree(qDt,nDt); MaskApi._rlesFree(qGt,nGt)
    return o:transpose(1,2)
  end
end

MaskApi.area = function( Rs )
  local Qs, n, h, w = MaskApi._rlesFrLua(Rs)
  local a = torch.IntTensor(n):contiguous()
  libmaskapi.rleArea(Qs,n,a:data())
  MaskApi._rlesFree(Qs,n)
  return a
end

MaskApi.toBbox = function( Rs )
  local Qs, n, h, w = MaskApi._rlesFrLua(Rs)
  local bb = torch.DoubleTensor(n,4):contiguous()
  libmaskapi.rleToBbox(Qs,bb:data(),n)
  MaskApi._rlesFree(Qs,n)
  return bb
end

MaskApi.frBbox = function( bbs, h, w )
  if bbs:dim()==1 then bbs=bbs:view(1,bbs:size(1)) end
  local n, k = bbs:size(1), bbs:size(2); assert(k==4)
  local data = bbs:type('torch.DoubleTensor'):contiguous():data()
  local Qs = MaskApi._rlesInit(n)
  libmaskapi.rleFrBbox(Qs[0],data,h,w,n)
  return MaskApi._rlesToLua(Qs,n)
end

MaskApi.frPoly = function( poly, h, w )
  local n = #poly
  local Qs, Q = MaskApi._rlesInit(n), MaskApi._rlesInit(1)
  for i,p in pairs(poly) do
    local xy = p:type('torch.DoubleTensor'):contiguous():data()
    libmaskapi.rleFrPoly(Qs[i-1],xy,p:size(1)/2,h,w)
  end
  libmaskapi.rleMerge(Qs,Q[0],n,0)
  MaskApi._rlesFree(Qs,n)
  return MaskApi._rlesToLua(Q,1)[1]
end

MaskApi.drawMasks = function( img, masks, maxn, alpha, clrs )
  local n, h, w = masks:size(1), masks:size(2), masks:size(3)
  if not maxn then maxn=n end
  if not alpha then alpha=.4 end
  if not clrs then clrs=torch.rand(n,3)*1.4 end
  local out = img:clone():contiguous():float()
  for i=1,math.min(maxn,n) do
    local M = masks[i]:contiguous():data()
    local B = torch.ByteTensor(h,w):zero():contiguous():data()
    -- get boundaries B in masks M quickly
    for y=0,h-2 do for x=0,w-2 do
      local k=y*w+x
      if M[k]~=M[k+1] then B[k],B[k+1]=1,1 end
      if M[k]~=M[k+w] then B[k],B[k+w]=1,1 end
      if M[k]~=M[k+1+w] then B[k],B[k+1+w]=1,1 end
    end end
    -- softly embed masks into image and add solid boundaries
    for j=1,3 do
      local O,c,a = out[j]:data(), math.min(clrs[i][j],1), alpha
      for k=0,w*h-1 do if M[k]==1 then O[k]=O[k]*(1-a)+c*a end end
      for k=0,w*h-1 do if B[k]==1 then O[k]=c end end
    end
  end
  return out
end

--------------------------------------------------------------------------------

MaskApi._rlesToLua = function( Qs, n )
  local h, w, Rs = tonumber(Qs[0].h), tonumber(Qs[0].w), {}
  for i=1,n do Rs[i]={size={h,w}, counts={}} end
  for i=1,n do
    local s = libmaskapi.rleToString(Qs[i-1])
    Rs[i].counts=ffi.string(s)
    ffi.C.free(s)
  end
  MaskApi._rlesFree(Qs,n)
  return Rs
end

MaskApi._rlesFrLua = function( Rs )
  if #Rs==0 then Rs={Rs} end
  local n, h, w = #Rs, Rs[1].size[1], Rs[1].size[2]
  local Qs = MaskApi._rlesInit(n)
  for i=1,n do
    local c = Rs[i].counts
    if( torch.type(c)=='string' ) then
      local s=ffi.new("char[?]",#c+1); ffi.copy(s,c)
      libmaskapi.rleFrString(Qs[i-1],s,h,w)
    elseif( torch.type(c)=='torch.IntTensor' ) then
      libmaskapi.rleInit(Qs[i-1],h,w,c:size(1),c:contiguous():data())
    else
      assert(false,"invalid RLE")
    end
  end
  return Qs, n, h, w
end

MaskApi._rlesInit = function( n )
  local Qs = ffi.new("RLE[?]",n)
  for i=1,n do libmaskapi.rleInit(Qs[i-1],0,0,0,NULL) end
  return Qs
end

MaskApi._rlesFree = function( Qs, n )
  for i=1,n do libmaskapi.rleFree(Qs[i-1]) end
end

--------------------------------------------------------------------------------

ffi.cdef[[
  void free(void *ptr);
  typedef unsigned int uint;
  typedef unsigned long siz;
  typedef unsigned char byte;
  typedef double* BB;
  typedef struct { siz h, w, m; uint *cnts; } RLE;
  void rleInit( RLE *R, siz h, siz w, siz m, uint *cnts );
  void rleFree( RLE *R );
  void rlesInit( RLE **R, siz n );
  void rlesFree( RLE **R, siz n );
  void rleEncode( RLE *R, const byte *mask, siz h, siz w, siz n );
  void rleDecode( const RLE *R, byte *mask, siz n );
  void rleMerge( const RLE *R, RLE *M, siz n, int intersect );
  void rleArea( const RLE *R, siz n, uint *a );
  void rleIou( RLE *dt, RLE *gt, siz m, siz n, byte *iscrowd, double *o );
  void bbIou( BB dt, BB gt, siz m, siz n, byte *iscrowd, double *o );
  void rleToBbox( const RLE *R, BB bb, siz n );
  void rleFrBbox( RLE *R, const BB bb, siz h, siz w, siz n );
  void rleFrPoly( RLE *R, const double *xy, siz k, siz h, siz w );
  char* rleToString( const RLE *R );
  void rleFrString( RLE *R, char *s, siz h, siz w );
]]
