/**************************************************************************
* Microsoft COCO Toolbox.      Version 0.90
* Data, paper, and tutorials available at:  http://mscoco.org/
* Code written by Piotr Dollar and Tsung-Yi Lin, 2014.
* Licensed under the Simplified BSD License [see private/bsd.txt]
**************************************************************************/
#include "gason.h"
#include "mex.h"
#include "string.h"

bool isRegularObjArray( const JsonValue &a ) {
  // check if all JSON_OBJECTs in JSON_ARRAY have the same fields
  JsonValue o=a.toNode()->value; int k, m, n; const char **keys;
  n=0; for(auto j:o) n++; keys=new const char*[n];
  k=0; for(auto j:o) keys[k++]=j->key;
  for( auto i:a ) {
    m=0; for(auto j:i->value) m++; if(m!=n) return false; k=0;
    for(auto j:i->value) if(strcmp(j->key,keys[k++])) return false;
  }
  delete [] keys; return true;
}

mxArray* toMatlab( const JsonValue &o, bool flatten ) {
  // convert JsonValue to Matlab mxArray
  int k, m, n; mxArray *M; const char **keys;
  switch( o.getTag() ) {
    case JSON_NUMBER:
      return mxCreateDoubleScalar(o.toNumber());
    case JSON_STRING:
      return mxCreateString(o.toString());
    case JSON_ARRAY: {
      if(!o.toNode()) return mxCreateCellMatrix(1,0);
      JsonValue o0=o.toNode()->value; JsonTag tag=o0.getTag();
      n=0; for(auto i:o) n++; bool isRegular=flatten;
      for(auto i:o) isRegular=isRegular && i->value.getTag()==tag;
      if( isRegular && tag==JSON_OBJECT && isRegularObjArray(o) ) {
        m=0; for(auto j:o0) m++; keys=new const char*[m];
        k=0; for(auto j:o0) keys[k++]=j->key;
        M = mxCreateStructMatrix(1,n,m,keys);
        k=0; for(auto i:o) { m=0; for(auto j:i->value)
          mxSetFieldByNumber(M,k,m++,toMatlab(j->value,flatten)); k++; }
        delete [] keys; return M;
      } else if( isRegular && tag==JSON_NUMBER ) {
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
      n=0; for(auto i:o) n++; keys=new const char*[n];
      k=0; for(auto i:o) keys[k++]=i->key;
      M = mxCreateStructMatrix(1,1,n,keys); k=0;
      for(auto i:o) mxSetFieldByNumber(M,0,k++,toMatlab(i->value,flatten));
      delete [] keys; return M;
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
  // get inputs
  if( nr<1 && nr>2 ) mexErrMsgTxt("One or two inputs expected.");
  if( nl>1 ) mexErrMsgTxt("One output expected.");
  char *str = mxArrayToString(pr[0]);
  bool flatten = (nr>1) ? mxGetScalar(pr[1])>0 : 1;
  
  // run gason parser
  char *endptr; JsonValue value; JsonAllocator allocator;
  int status = jsonParse(str, &endptr, &value, allocator);
  if( status != JSON_OK) mexErrMsgTxt(jsonStrError(status));
  
  // convert to matlab struct and free str
  pl[0] = toMatlab( value, flatten );
  mxFree(str);
}
