/**************************************************************************
* Microsoft COCO Toolbox.      Version 1.0
* Data, paper, and tutorials available at:  http://mscoco.org/
* Code written by Piotr Dollar and Tsung-Yi Lin, 2015.
* Licensed under the Simplified BSD License [see coco/license.txt]
**************************************************************************/
#include "mex.h"
#include "maskApi.h"
#include <string.h>

void checkType( const mxArray *M, mxClassID id ) {
  if(mxGetClassID(M)!=id) mexErrMsgTxt("Invalid type.");
}

void toMxArray( const RLES &Rs, mxArray *&M ) {
  const char *fs[] = {"size", "counts"};
  siz n=Rs.size(); M=mxCreateStructMatrix(1,n,2,fs);
  for( siz i=0; i<n; i++ ) {
    mxArray *S=mxCreateNumericMatrix(1,2,mxDOUBLE_CLASS,mxREAL);
    mxSetFieldByNumber(M,i,0,S); double *s=mxGetPr(S);
    s[0]=Rs[i].h; s[1]=Rs[i].w; std::string c; Rs[i].toString(c);
    mxSetFieldByNumber(M,i,1,mxCreateString(c.c_str()));
  }
}

void frMxArray( RLES &Rs, const mxArray *M ) {
  const char *fs[] = {"size", "counts"}; siz i, j, n, k, O[2];
  const char *err="Invalid RLE struct array.";
  n=mxGetNumberOfElements(M); Rs.resize(n); if(n==0) return;
  if(!mxIsStruct(M) || mxGetNumberOfFields(M)!=2) mexErrMsgTxt(err);
  for( i=0; i<2; i++ ) { O[i]=2; for( j=0; j<2; j++ ) {
    if(!strcmp(mxGetFieldNameByNumber(M,j),fs[i])) O[i]=j; }}
  for( i=0; i<2; i++ ) if(O[i]>1) mexErrMsgTxt(err);
  for( i=0; i<n; i++ ) {
    mxArray *S=mxGetFieldByNumber(M,i,O[0]); checkType(S,mxDOUBLE_CLASS);
    double *s=mxGetPr(S); Rs[i].h=siz(s[0]); Rs[i].w=siz(s[1]);
    if(Rs[i].h!=Rs[0].h||Rs[i].w!=Rs[0].w) mexErrMsgTxt(err);
    mxArray *C=mxGetFieldByNumber(M,i,O[1]); void *c=mxGetData(C);
    k=mxGetNumberOfElements(C); Rs[i].counts.resize(k);
    if( mxGetClassID(C)==mxDOUBLE_CLASS )
      for(j=0; j<k; j++) Rs[i].counts[j]=uint(((double*)c)[j]);
    else if( mxGetClassID(C)==mxUINT32_CLASS )
      for(j=0; j<k; j++) Rs[i].counts[j]=((uint*)c)[j];
    else if( mxGetClassID(C)==mxCHAR_CLASS ) {
      char *c=new char[k+1]; mxGetString(C,c,k+1);
      Rs[i].frString(c,Rs[i].h,Rs[i].w); delete [] c;
    }
    else mexErrMsgTxt(err);
  }
}

void mexFunction( int nl, mxArray *pl[], int nr, const mxArray *pr[] )
{
  char action[1024];
  mxGetString(pr[0],action,1024); nr--; pr++;
  
  if(!strcmp(action,"encode")) {
    checkType(pr[0],mxUINT8_CLASS);
    const mwSize *ds=mxGetDimensions(pr[0]), a=ds[0]*ds[1];
    siz n=(mxGetNumberOfDimensions(pr[0])==2) ? 1 : ds[2];
    byte *mask = (byte*) mxGetData(pr[0]); RLES Rs(n);
    for(siz i=0; i<n; i++) Rs[i].encode(mask+a*i,ds[0],ds[1]);
    toMxArray(Rs,pl[0]);
    
  } else if(!strcmp(action,"decode")) {
    RLES Rs; frMxArray(Rs,pr[0]); siz n=Rs.size();
    mwSize ds[3]; ds[0]=n?Rs[0].h:0; ds[1]=n?Rs[0].w:0; ds[2]=n;
    pl[0]=mxCreateNumericArray(3,ds,mxUINT8_CLASS,mxREAL);
    byte *mask=(byte*) mxGetPr(pl[0]); siz a=ds[0]*ds[1];
    for(siz i=0; i<n; i++) Rs[i].decode(mask+a*i);
    
  } else if(!strcmp(action,"merge")) {
    RLES Rs, R(1); frMxArray(Rs,pr[0]);
    bool intersect = (nr>=2) ? (mxGetScalar(pr[1])>0) : false;
    R[0].merge(Rs,intersect); toMxArray(R,pl[0]);
    
  } else if(!strcmp(action,"area")) {
    RLES Rs; frMxArray(Rs,pr[0]); siz n=Rs.size();
    pl[0]=mxCreateNumericMatrix(1,n,mxUINT32_CLASS,mxREAL);
    uint *a=(uint*) mxGetPr(pl[0]);
    for(siz i=0; i<n; i++) a[i]=Rs[i].area();
    
  } else if(!strcmp(action,"iou")) {
    if(nr>2) checkType(pr[2],mxUINT8_CLASS);
    byte *iscrowd = nr>2 ? (byte*) mxGetPr(pr[2]) : NULL;
    if(mxIsStruct(pr[0]) || mxIsStruct(pr[1])) {
      RLES dt, gt; frMxArray(dt,pr[0]); frMxArray(gt,pr[1]);
      siz m=dt.size(), n=gt.size();
      pl[0]=mxCreateNumericMatrix(m,n,mxDOUBLE_CLASS,mxREAL);
      double *o=mxGetPr(pl[0]); RLE::iou(dt,gt,iscrowd,o);
    } else {
      checkType(pr[0],mxDOUBLE_CLASS); checkType(pr[1],mxDOUBLE_CLASS);
      double *dt=mxGetPr(pr[0]); double *gt=mxGetPr(pr[1]);
      siz m=mxGetN(pr[0]), n=mxGetN(pr[1]);
      pl[0]=mxCreateNumericMatrix(m,n,mxDOUBLE_CLASS,mxREAL);
      double *o=mxGetPr(pl[0]); RLE::iou(dt,gt,m,n,iscrowd,o);
    }
    
  } else if(!strcmp(action,"toBbox")) {
    RLES Rs; frMxArray(Rs,pr[0]); siz n=Rs.size();
    pl[0]=mxCreateNumericMatrix(4,n,mxDOUBLE_CLASS,mxREAL);
    double *B=mxGetPr(pl[0]);
    for(siz i=0; i<n; i++) Rs[i].toBbox(B+4*i);
    
  } else if(!strcmp(action,"frBbox")) {
    checkType(pr[0],mxDOUBLE_CLASS);
    double *B=mxGetPr(pr[0]); siz n=mxGetN(pr[0]);
    siz h=siz(mxGetScalar(pr[1])), w=siz(mxGetScalar(pr[2]));
    RLES Rs(n); for(siz i=0; i<n; i++) Rs[i].frBbox(B+4*i,h,w);
    toMxArray(Rs,pl[0]);
    
  } else if(!strcmp(action,"frPoly")) {
    checkType(pr[0],mxCELL_CLASS);
    siz h=siz(mxGetScalar(pr[1])), w=siz(mxGetScalar(pr[2]));
    siz n=mxGetNumberOfElements(pr[0]); RLES Rs(n), R(1);
    for(siz i=0; i<n; i++) {
      mxArray *M=mxGetCell(pr[0],i); checkType(M,mxDOUBLE_CLASS);
      siz k=mxGetNumberOfElements(M)/2; double *xy=mxGetPr(M);
      double *x=new double[k], *y=new double[k];
      for(siz j=0; j<k; j++) { x[j]=xy[j*2]; y[j]=xy[j*2+1]; }
      Rs[i].frPoly(x,y,k,h,w); delete [] x; delete [] y;
    }
    R[0].merge(Rs,0); toMxArray(R,pl[0]);
    
  } else mexErrMsgTxt("Invalid action.");
}
