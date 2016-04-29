--[[----------------------------------------------------------------------------

Interface for manipulating masks stored in RLE format.
For more details please see http://mscoco.org/dataset/#download.
More detailed information on RLE can be found in the Matlab MaskApi here:
https://github.com/pdollar/coco/blob/master/MatlabAPI/MaskApi.m.

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
 maskApi = MaskApi()
 Rs     = maskApi:encode( masks )
 masks  = maskApi:decode( Rs )
 R      = maskApi:merge( Rs, [intersect=false] )
 o      = maskApi:iou( dt, gt, [iscrowd=false] )
 a      = maskApi:area( Rs )
 bbs    = maskApi:toBbox( Rs )
 Rs     = maskApi:frBbox( bbs, h, w )
 R      = maskApi:frPoly( poly, h, w )

In the API the following formats are used:
 R,Rs   - [table] Run-length encoding of binary mask(s)
 masks  - [nxhxw] Binary mask(s)
 bbs    - [nx4] Bounding box(es) stored as [x y w h]
 poly   - Polygon stored as {[x1 y1 x2 y2...],[x1 y1 ...],...}
 dt,gt  - May be either bounding boxes or encoded masks
Both poly and bbs are 0-indexed (bbox=[0 0 1 1] encloses first pixel).

To compile use the following (some precompiled binaries are included):
  cd coco/common/
  gcc -shared     -fPIC -std=c99 -Wall -o libmaskApi.so maskApi.c #LINUX
  gcc -dynamiclib -fPIC -std=c99 -Wall -o libmaskApi.dylib maskApi.c #OSX
Please do not contact us for help with compiling.

Common Object in COntext (COCO) Toolbox.      version 3.0
Data, paper, and tutorials available at:  http://mscoco.org/
Code written by Pedro O. Pinheiro and Piotr Dollar, 2016.
Licensed under the Simplified BSD License [see coco/license.txt]

------------------------------------------------------------------------------]]

local ffi = require 'ffi'
local M = {}
local MaskApi = torch.class('MaskApi', M)

--------------------------------------------------------------------------------
-- main functions

function MaskApi:__init()
  local d = paths.dirname(paths.thisfile())
  ffi.cdef(assert(io.open(d..'/../common/maskApi.h','r')):read('*all'))
  ffi.cdef('void free(void *ptr)')
  local ext = ffi.os=='OSX' and 'dylib' or 'so'
  self.mask = ffi.load(d..'/../common/libmaskApi.'..ext)
end

function MaskApi:encode( masks )
  local n, h, w = masks:size(1), masks:size(2), masks:size(3)
  masks = masks:type('torch.ByteTensor'):transpose(2,3)
  local data = masks:contiguous():data()
  local Qs = self:rlesInit(n)
  self.mask.rleEncode(Qs[0],data,h,w,n)
  return self:rlesToLua(Qs,n)
end

function MaskApi:decode( Rs )
  local Qs, n, h, w = self:rlesFrLua(Rs)
  local masks = torch.ByteTensor(n,w,h):zero():contiguous()
  self.mask.rleDecode(Qs,masks:data(),n)
  self:rlesFree(Qs,n)
  return masks:transpose(2,3)
end

function MaskApi:merge( Rs, intersect )
  intersect = intersect or 0
  local Qs, n, h, w = self:rlesFrLua(Rs)
  local Q = self:rlesInit(1)
  self.mask.rleMerge(Qs,Q,n,intersect)
  self:rlesFree(Qs,n)
  return self:rlesToLua(Q,1)
end

function MaskApi:iou( dt, gt, iscrowd )
  if not iscrowd then iscrowd = NULL else
    iscrowd = iscrowd:type('torch.ByteTensor'):contiguous():data()
  end
  if torch.isTensor(gt) and torch.isTensor(dt) then
    local nDt, k = dt:size(1), dt:size(2); assert(k==4)
    local nGt, k = gt:size(1), gt:size(2); assert(k==4)
    local dDt = dt:type('torch.DoubleTensor'):contiguous():data()
    local dGt = gt:type('torch.DoubleTensor'):contiguous():data()
    local o = torch.DoubleTensor(nGt,nDt):contiguous()
    self.mask.bbIou(dDt,dGt,nDt,nGt,iscrowd,o:data())
    return o:transpose(1,2)
  else
    local qDt, nDt = self:rlesFrLua(dt)
    local qGt, nGt = self:rlesFrLua(gt)
    local o = torch.DoubleTensor(nGt,nDt):contiguous()
    self.mask.rleIou(qDt,qGt,nDt,nGt,iscrowd,o:data())
    self:rlesFree(qDt,nDt); self:rlesFree(qGt,nGt)
    return o:transpose(1,2)
  end
end

function MaskApi:area( Rs )
  local Qs, n, h, w = self:rlesFrLua(Rs)
  local a = torch.IntTensor(n):contiguous()
  self.mask.rleArea(Qs,n,a:data())
  self:rlesFree(Qs,n)
  return a
end

function MaskApi:toBbox( Rs )
  local Qs, n, h, w = self:rlesFrLua(Rs)
  local bb = torch.DoubleTensor(n,4):contiguous()
  self.mask.rleToBbox(Qs,bb:data(),n)
  self:rlesFree(Qs,n)
  return bb
end

function MaskApi:frBbox( bbs, h, w )
  local n, k = bbs:size(1), bbs:size(2); assert(k==4)
  local data = bbs:type('torch.DoubleTensor'):contiguous():data()
  local Qs = self:rlesInit(n)
  self.mask.rleFrBbox(Qs[0],data,h,w,n)
  return self:rlesToLua(Qs,n)
end

function MaskApi:frPoly( poly, h, w )
  local n = #poly
  local Qs, Q = self:rlesInit(n), self:rlesInit(1)
  for i,p in pairs(poly) do
    local xy = p:type('torch.DoubleTensor'):contiguous():data()
    self.mask.rleFrPoly(Qs[i-1],xy,p:size(1)/2,h,w)
  end
  self.mask.rleMerge(Qs,Q[0],n,0)
  self:rlesFree(Qs,n)
  return self:rlesToLua(Q,1)
end

--------------------------------------------------------------------------------
-- private helper functions

function MaskApi:rlesToLua( Qs, n )
  local h, w, Rs = tonumber(Qs[0].h), tonumber(Qs[0].w), {}
  for i=1,n do Rs[i]={size={h,w}, counts={}} end
  for i=1,n do
    local s = self.mask.rleToString(Qs[i-1])
    Rs[i].counts=ffi.string(s)
    ffi.C.free(s)
  end
  self:rlesFree(Qs,n)
  return Rs
end

function MaskApi:rlesFrLua( Rs )
  if #Rs==0 then Rs={Rs} end
  local n, h, w = #Rs, Rs[1].size[1], Rs[1].size[2]
  local Qs = self:rlesInit(n)
  for i=1,n do
    local c = Rs[i].counts
    if( torch.type(c)=='string' ) then
      local s=ffi.new("char[?]",#c+1); ffi.copy(s,c)
      self.mask.rleFrString(Qs[i-1],s,h,w)
    elseif( torch.type(c)=='torch.IntTensor' ) then
      self.mask.rleInit(Qs[i-1],h,w,c:size(1),c:contiguous():data())
    else
      assert(false,"invalid RLE")
    end
  end
  return Qs, n, h, w
end

function MaskApi:rlesInit( n )
  local Qs = ffi.new("RLE[?]",n)
  for i=1,n do self.mask.rleInit(Qs[i-1],0,0,0,NULL) end
  return Qs
end

function MaskApi:rlesFree( Qs, n )
  for i=1,n do self.mask.rleFree(Qs[i-1]) end
end

--------------------------------------------------------------------------------
return M.MaskApi
