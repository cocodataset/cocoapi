/**************************************************************************
* Microsoft COCO Toolbox.      Version 1.0
* Data, paper, and tutorials available at:  http://mscoco.org/
* Code written by Piotr Dollar and Tsung-Yi Lin, 2015.
* Licensed under the Simplified BSD License [see coco/license.txt]
**************************************************************************/
#pragma once
#include <vector>
#include <string>
class RLE;
typedef unsigned int uint;
typedef unsigned long siz;
typedef unsigned char byte;
typedef std::vector<RLE> RLES;
typedef double* BB;

class RLE {
public:
  // Basic constructor.
  RLE() { w=h=0; }
  
  // Encode binary mask using RLE.
  void encode( const byte *mask, siz h, siz w );
  
  // Decode binary mask encoded via RLE.
  void decode( byte *mask ) const;
  
  // Compute union or intersection of two encoded masks.
  void merge( const RLE &A, const RLE &B, bool intersect );
  
  // Compute union or intersection of multiple encoded masks.
  void merge( const RLES &Rs, bool intersect );
  
  // Compute intersection over union between two masks.
  static double iou( RLE &dt, RLE &gt, byte iscrowd );
  
  // Compute intersection over union between two sets of masks.
  static void iou( RLES &dt, RLES &gt, byte *iscrowd, double *o );
  
  // Compute intersection over union between two sets of bounding boxes.
  static void iou( BB dt, BB gt, siz m, siz n, byte *iscrowd, double *o );
  
  // Compute area of encoded mask.
  uint area() const;
  
  // Get bounding box surrounding encoded mask.
  void toBbox( BB bbox ) const;
  
  // Convert bounding box to encoded mask.
  void frBbox( const BB bbox, siz h, siz w );
  
  // Convert polygon to encoded mask.
  void frPoly( double *x, double *y, siz k, siz h, siz w );
  
  // Get compressed string representation of encoded mask.
  void toString( std::string &s ) const;
  
  // Convert from compressed string representation of encoded mask.
  void frString( const std::string &s, siz h, siz w );
  
  // Data structure for (column-wise) run length encoding.
  std::vector<uint> counts;
  siz h, w;
};
