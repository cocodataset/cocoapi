function json = gason( varargin )
% Parse JSON string and return corresponding JSON object.
%
% This parser is based on Gason written and maintained by Ivan Vashchaev:
%                 https://github.com/vivkin/gason
% Gason is a "lightweight and fast JSON parser for C++". Please see the
% above link for license information and additional details about Gason.
%
% This function simply calls the C++ parser and converts the output into an
% appropriate Matlab structure. As the parsing is performed in mex the
% resulting parser is blazingly fast. Large JSON structs (100MB+) take only
% a few seconds to parse (compared to hours for pure Matlab parsers).
%
% Gason require C++11 to compile (for GCC this requires version 4.7 or
% later). The following command compiles the parser (may require tweaking):
%   mex('CXXFLAGS="\$CXXFLAGS -std=c++11"','private/gason.cpp', ...
%     'private/gasonMex.cpp','-output',['private/gasonMex.' mexext]);
% Note the use of the "-std=c++11" flag. Precompiled binaries for Linux and
% Mac are included. Please do not contact us for help with compiling.
%
% Note that by default all JSON arrays are stored as Matlab cell arrays. If
% flatten=1 than JSON arrays which contain only numbers are stored as
% regular arrays. This is much faster and can use considerably less memory.
%
% USAGE
%  json = gason( string, [flatten] )
%
% INPUTS
%  string     - JSON string to be parsed
%  flatten    - [0] if 1 flatten arrays of numbers to regular arrays
%
% OUTPUTS
%  json       - parsed JSON object
%
% EXAMPLE
%  s = '{"first":"piotr","last":"dollar"}';
%  json = gason( s );
%
% See also
%
% Microsoft COCO Toolbox.      Version 0.90
% Data, paper, and tutorials available at:  http://mscoco.org/
% Code written by Piotr Dollar and Tsung-Yi Lin, 2014.
% Licensed under the Simplified BSD License [see private/bsd.txt]

json = gasonMex( varargin{:} );
