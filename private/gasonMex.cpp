/**************************************************************************
* Microsoft COCO Toolbox.      Version 0.90
* Data, paper, and tutorials available at:  http://mscoco.org/
* Code written by Piotr Dollar and Tsung-Yi Lin, 2014.
* Licensed under the Simplified BSD License [see private/bsd.txt]
**************************************************************************/
#include "gason.h"
#include "mex.h"

mxArray* toMatlab( const JsonValue &o, bool flatten ) {
  int k, m, n; mxArray *M; const char **names;
  switch( o.getTag() ) {
    case JSON_NUMBER:
      return mxCreateDoubleScalar(o.toNumber());
    case JSON_STRING:
      return mxCreateString(o.toString());
    case JSON_ARRAY: {
      if (!o.toNode()) return mxCreateCellMatrix(1,0);
      JsonValue o0=o.toNode()->value; JsonTag tag=o0.getTag();
      for(auto i:o) flatten=flatten && i->value.getTag()==tag;
      n=0; for(auto i:o) n++;
      if( flatten && tag==JSON_NUMBER ) {
        M = mxCreateDoubleMatrix(1,n,mxREAL); double *p=mxGetPr(M);
        k=0; for(auto i:o) p[k++]=i->value.toNumber(); return M;
      } else {
        M = mxCreateCellMatrix(1,n);
        k=0; for(auto i:o) mxSetCell(M,k++,toMatlab(i->value,flatten));
        return M;
      }
    }
    case JSON_OBJECT:
      if(!o.toNode()) return mxCreateStructMatrix(1,0,0,NULL);
      n=0; for(auto i:o) n++; names=new const char*[n];
      k=0; for(auto i:o) names[k++]=i->key;
      M = mxCreateStructMatrix(1,1,n,names); k=0;
      for(auto i:o) mxSetFieldByNumber(M,0,k++,toMatlab(i->value,flatten));
      delete [] names; return M;
    case JSON_TRUE:
      return mxCreateDoubleScalar(1);
    case JSON_FALSE:
      return mxCreateDoubleScalar(0);
    case JSON_NULL:
      return mxCreateDoubleMatrix(0,0,mxREAL);
  }
}

// json = mexFunction( jsonstring, [flatten] )
void mexFunction( int nl, mxArray *pl[], int nr, const mxArray *pr[] )
{
  // parse inputs
  if( nr<1 && nr>2 ) mexErrMsgTxt("One or two inputs expected.");
  if( nl>1 ) mexErrMsgTxt("One output expected.");
  char *str = mxArrayToString(pr[0]);
  bool flatten = (nr>1) ? mxGetScalar(pr[1])>0 : 0;
  
  // run gason parser
  char *endptr; JsonValue value; JsonAllocator allocator;
  int status = jsonParse(str, &endptr, &value, allocator);
  if( status != JSON_OK) mexErrMsgTxt(jsonStrError(status));
  
  // convert to matlab struct and free str
  pl[0] = toMatlab( value, flatten );
  mxFree(str);
}
