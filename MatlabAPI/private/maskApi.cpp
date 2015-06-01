/**************************************************************************
* Microsoft COCO Toolbox.      Version 1.0
* Data, paper, and tutorials available at:  http://mscoco.org/
* Code written by Piotr Dollar and Tsung-Yi Lin, 2015.
* Licensed under the Simplified BSD License [see coco/license.txt]
**************************************************************************/
#include "maskApi.h"
#include <math.h>
#include <stdlib.h>
#include <algorithm>

using namespace std;

void RLE::encode( const byte *mask, siz h0, siz w0 ) {
  h=h0; w=w0; counts.clear(); uint c=0; byte p=0;
  for( siz i=0; i<w*h; i++ ) {
    if(mask[i]!=p) { counts.push_back(c); c=0; p=mask[i]; } c++; }
  counts.push_back(c);
}

void RLE::decode( byte *mask ) const {
  byte v=0; for( siz i=0; i<counts.size(); i++ ) {
    for( siz j=0; j<counts[i]; j++ ) *(mask++)=v; v=!v; }
}

void RLE::merge( const RLE &A, const RLE &B, bool intersect ) {
  counts.clear(); w=A.w; h=B.h;
  if( A.h!=B.h || A.w!=B.w ) { w=h=0; return; }
  siz ka, kb, a, b; uint c, ca, cb, cc, ct; bool v, va, vb, vp;
  ca=A.counts[0]; ka=A.counts.size(); v=va=vb=0;
  cb=B.counts[0]; kb=B.counts.size(); a=b=1; cc=0; ct=1;
  while( ct>0 ) {
    c=min(ca,cb); cc+=c; ct=0;
    ca-=c; if(!ca && a<ka) { ca=A.counts[a++]; va=!va; } ct+=ca;
    cb-=c; if(!cb && b<kb) { cb=B.counts[b++]; vb=!vb; } ct+=cb;
    vp=v; if(intersect) v=va&&vb; else v=va||vb;
    if( v!=vp||ct==0 ) { counts.push_back(cc); cc=0; }
  }
}

void RLE::merge( const RLES &Rs, bool intersect ) {
  counts.clear(); w=h=0; siz i, n=Rs.size(); if(n==0) return;
  for(i=1; i<n; i++) if(Rs[i].h!=Rs[0].h||Rs[i].w!=Rs[0].w) return;
  if(n==1) *this=Rs[0]; else merge(Rs[0],Rs[1],intersect);
  for(i=2; i<n; i++) { RLE T=*this; merge(T,Rs[i],intersect); }
}

double RLE::iou( RLE &dt, RLE &gt, byte iscrowd ) {
  if( dt.h!=gt.h || dt.w!=gt.w ) return -1;
  siz ka, kb, a, b; uint c, ca, cb, ct, i, u; bool va, vb;
  ca=dt.counts[0]; ka=dt.counts.size(); va=vb=0;
  cb=gt.counts[0]; kb=gt.counts.size(); a=b=1; i=u=0; ct=1;
  while( ct>0 ) {
    c=min(ca,cb); if(va||vb) { u+=c; if(va&&vb) i+=c; } ct=0;
    ca-=c; if(!ca && a<ka) { ca=dt.counts[a++]; va=!va; } ct+=ca;
    cb-=c; if(!cb && b<kb) { cb=gt.counts[b++]; vb=!vb; } ct+=cb;
  }
  if(i==0) return 0; if(iscrowd) u=dt.area();
  return double(i)/double(u);
}

void RLE::iou( RLES &dt, RLES &gt, byte *iscrowd, double *o ) {
  siz m=dt.size(), n=gt.size(), g, d;
  BB db=new double[4*m], gb=new double[4*n];
  for( d=0; d<m; d++ ) dt[d].toBbox(db+d*4);
  for( g=0; g<n; g++ ) gt[g].toBbox(gb+g*4);
  RLE::iou( db, gb, m, n, iscrowd, o );
  for( g=0; g<n; g++ ) for( d=0; d<m; d++ ) if(o[g*m+d]>0) {
    o[g*m+d]=RLE::iou(dt[d],gt[g],iscrowd?iscrowd[g]:0); }
  delete [] db; delete [] gb;
}

void RLE::iou( BB dt, BB gt, siz m, siz n, byte *iscrowd, double *o ) {
  double h, w, i, u, ga, da; siz g, d; bool crowd;
  for( g=0; g<n; g++ ) {
    BB G=gt+g*4; ga=G[2]*G[3]; crowd=iscrowd!=NULL && iscrowd[g];
    for( d=0; d<m; d++ ) {
      BB D=dt+d*4; da=D[2]*D[3]; o[g*m+d]=0;
      w=min(D[2]+D[0],G[2]+G[0])-max(D[0],G[0]); if(w<=0) continue;
      h=min(D[3]+D[1],G[3]+G[1])-max(D[1],G[1]); if(h<=0) continue;
      i=w*h; u = crowd ? da : da+ga-i; o[g*m+d]=i/u;
    }
  }
}

uint RLE::area() const {
  uint a=0; siz i, k=counts.size();
  for(i=1; i<k; i+=2) a+=counts[i]; return a;
}

void RLE::toBbox( BB bbox ) const {
  uint hu=uint(h); siz k=counts.size(); k=siz(k/2)*2;
  if(k==0) { bbox[0]=bbox[1]=bbox[2]=bbox[3]=0; return; }
  vector<uint> cc, y, x; cc=counts; x.resize(k); y.resize(k);
  for(siz i=1; i<k; i++) cc[i]+=cc[i-1];
  for(siz i=1; i<k; i+=2) cc[i]--;
  for(siz i=0; i<k; i++) { y[i]=cc[i]%hu; x[i]=(cc[i]-y[i])/hu; }
  bbox[0] = *min_element(x.begin(),x.end());
  bbox[1] = *min_element(y.begin(),y.end());
  bbox[2] = *max_element(x.begin(),x.end()) - bbox[0] + 1;
  bbox[3] = *max_element(y.begin(),y.end()) - bbox[1] + 1;
}

void RLE::frBbox( const BB bbox, siz h0, siz w0 ) {
  double xs=bbox[0], xe=bbox[0]+bbox[2];
  double ys=bbox[1], ye=bbox[1]+bbox[3];
  double x[4] = {xs,xs,xe,xe}, y[4]={ys,ye,ye,ys};
  frPoly( x, y, 4, h0, w0 );
}

void RLE::frPoly( double *x0, double *y0, siz k, siz h0, siz w0 ) {
  h=h0; w=w0; counts.clear(); siz i; double scale=5;
  // upsample and get discrete points densely along entire boundary
  vector<int> x1(k), y1(k), x, y;
  for(i=0; i<k; i++) x1[i]=int(scale*x0[i]+.5);
  for(i=0; i<k; i++) y1[i]=int(scale*y0[i]+.5);
  if( x1[k-1]!=x1[0] || y1[k-1]!=y1[0] ) {
    x1.push_back(x1[0]); y1.push_back(y1[0]); k++; }
  for( i=0; i<k-1; i++ ) {
    int xs=x1[i], xe=x1[i+1], ys=y1[i], ye=y1[i+1], dx, dy, t;
    dx=abs(xe-xs); dy=abs(ys-ye); bool flip; double s;
    if(!dx&&!dy) { x.push_back(xs); y.push_back(ys); continue; }
    flip = (dx>=dy && xs>xe) || (dx<dy && ys>ye);
    if(flip) { t=xs; xs=xe; xe=t; t=ys; ys=ye; ye=t; }
    s = dx>=dy ? double(ye-ys)/dx : double(xe-xs)/dy;
    if(dx>=dy) for( int j=0; j<=dx; j++ ) {
      t=flip?dx-j:j; x.push_back(t+xs); y.push_back(int(ys+s*t+.5));
    } else for( int j=0; j<=dy; j++ ) {
      t=flip?dy-j:j; y.push_back(t+ys); x.push_back(int(xs+s*t+.5));
    }
  }
  // get points along y-boundary and downsample
  x1.clear(); y1.clear(); k=x.size(); double xd, yd;
  for( i=1; i<k; i++ ) if(x[i]!=x[i-1]) {
    xd=double(x[i]<x[i-1]?x[i]:x[i]-1); xd=(xd+.5)/scale-.5;
    if( floor(xd)!=xd || xd<0 || xd>w-1 ) continue;
    yd=double(y[i]<y[i-1]?y[i]:y[i-1]); yd=(yd+.5)/scale-.5;
    if(yd<0) yd=0; else if(yd>h) yd=h; yd=ceil(yd);
    x1.push_back(int(xd)); y1.push_back(int(yd));
  }
  // compute rle encoding given y-boundary points
  k=x1.size(); x.resize(k);
  for( i=0; i<k; i++ ) x[i]=x1[i]*int(h)+y1[i];
  sort(x.begin(),x.end()); int p=0; x.push_back(int(h*w)); k++;
  for( i=0; i<k; i++ ) { counts.push_back(uint(x[i]-p)); p=x[i]; }
  vector<uint> c(counts); counts.clear(); counts.push_back(c[0]); i=1;
  while(i<k) if(c[i]>0) counts.push_back(c[i++]); else {
    i++; if(i<k) counts.back()+=c[i++]; }
}

void RLE::toString( string &s ) const {
  // Similar to LEB128 but using 6 bits/char and ascii chars 48-111.
  siz i, n=counts.size(); s.clear(); long x; bool more;
  for( i=0; i<n; i++ ) {
    x=long(counts[i]); if(i>2) x-=long(counts[i-2]); more=true;
    while( more ) {
      char c=x & 0x1f; x >>= 5; more=(c & 0x10) ? x!=-1 : x!=0;
      if(more) c |= 0x20; c+=48; s.push_back(c);
    }
  }
}

void RLE::frString( const string &s, siz h0, siz w0 ) {
  h=h0; w=w0; counts.clear(); siz i=0, k=0, m; long x; bool more;
  while( s[k] ) {
    x=0; m=0; more=true;
    while( more ) {
      char c=s[k]-48; x |= (c & 0x1f) << 5*m;
      more = c & 0x20; k++; m++;
      if(!more && (c & 0x10)) x |= -1 << 5*m;
    }
    if(i>2) x+=long(counts[i-2]); i++; counts.push_back(uint(x));
  }
}
