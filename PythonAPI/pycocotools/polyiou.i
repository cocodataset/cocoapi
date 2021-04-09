%module polyiou
%include "std_vector.i"

namespace std {
    %template(VectorDouble) vector<double>;
};

%{
#define SWIG_FILE_WITH_INIT
#include<cstdio>
#include<iostream>
#include<algorithm>
#include <vector>

#include "polyiou.h"
%}

%include "polyiou.h"

