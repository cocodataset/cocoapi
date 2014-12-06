/*******************************************************************************
* Microsoft COCO Toolbox.      Version 0.90
* Data, paper, and tutorials available at:  http://mscoco.org/
* Code written by Piotr Dollar and Tsung-Yi Lin, 2014.
* Licensed under the Simplified BSD License [see private/bsd.txt]
*******************************************************************************/
#include "gason.h"
#include "mex.h"

mxArray* toMatlab( const JsonValue &o, bool flat )
{
  switch( o.getTag() ) {
    case JSON_NUMBER:
      return mxCreateDoubleScalar(o.toNumber());
    case JSON_STRING:
      return mxCreateString(o.toString());
    case JSON_ARRAY: {
      if (!o.toNode()) return mxCreateCellMatrix(1,0);
      int n=0; for(auto i:o) n++; mxArray *m; int k=0;
      int reg=1; for(auto i:o) reg=reg&&i->value.getTag()==JSON_NUMBER;
      if( flat && reg ) {
        // if all elements have type JSON_NUMBER use regular array
        m = mxCreateDoubleMatrix(1,n,mxREAL); double *mp=mxGetPr(m);
        for(auto i:o) mp[k++]=i->value.toNumber();
      } else {
        m = mxCreateCellMatrix(1,n);
        for(auto i:o) mxSetCell(m,k++,toMatlab(i->value,flat));
      }
      return m;
    }
    case JSON_OBJECT: {
      if(!o.toNode()) return mxCreateStructMatrix(1,0,0,NULL);
      int n=0; for(auto i:o) n++;
      const char **names = new const char*[n];
      int k=0; for(auto i:o) names[k++]=i->key;
      mxArray *m = mxCreateStructMatrix(1,1,n,names); k=0;
      for(auto i:o) mxSetFieldByNumber(m,0,k++,toMatlab(i->value,flat));
      delete [] names; return m;
    }
    case JSON_TRUE:
      return mxCreateDoubleScalar(1);
    case JSON_FALSE:
      return mxCreateDoubleScalar(0);
    case JSON_NULL:
      return mxCreateDoubleMatrix(0,0,mxREAL);
  }
  return NULL;
}

// json = mexFunction( jsonstring, [flatten] )
void mexFunction( int nl, mxArray *pl[], int nr, const mxArray *pr[] )
{
  // parse inputs
  if( nr<1 && nr>2 ) mexErrMsgTxt("One or two inputs expected.");
  if( nl>1 ) mexErrMsgTxt("One output expected.");
  char *str = mxArrayToString(pr[0]);
  bool regular = (nr>1) ? mxGetScalar(pr[1])>0 : 0;
  
  // run gason parser
  char *endptr; JsonValue value; JsonAllocator allocator;
  int status = jsonParse(str, &endptr, &value, allocator);
  if( status != JSON_OK) mexErrMsgTxt(jsonStrError(status));
  
  // convert to matlab struct and free str
  pl[0] = toMatlab( value, regular );
  mxFree(str);
}
