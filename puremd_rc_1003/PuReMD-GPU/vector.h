/*----------------------------------------------------------------------
  PuReMD-GPU - Reax Force Field Simulator
      
  Copyright (2014) Purdue University
  Sudhir Kylasa, skylasa@purdue.edu
  Hasan Metin Aktulga, haktulga@cs.purdue.edu
  Ananth Y Grama, ayg@cs.purdue.edu
 
  This program is free software; you can redistribute it and/or
  modify it under the terms of the GNU General Public License as
  published by the Free Software Foundation; either version 2 of 
  the License, or (at your option) any later version.
               
  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  
  See the GNU General Public License for more details:
  <http://www.gnu.org/licenses/>.
  ----------------------------------------------------------------------*/

#ifndef __VECTOR_H_
#define __VECTOR_H_

#include "mytypes.h"
#include "random.h"

int  Vector_isZero( real*, int );
void Vector_MakeZero( real*, int );
void Vector_Copy( real*, real*, int );
//void Vector_Scale( real*, real, real*, int );
//void Vector_Sum( real*, real, real*, real, real*, int );
//void Vector_Add( real*, real, real*, int );
void Vector_Print( FILE*, char*, real*, int );
real Norm( real*, int );

HOST_DEVICE inline real Dot( real*, real*, int );

void rvec_Sum( rvec, rvec, rvec );
real rvec_ScaledDot( real, rvec, real, rvec );
void rvec_Multiply( rvec, rvec, rvec );
void rvec_Divide( rvec, rvec, rvec );
void rvec_iDivide( rvec, rvec, ivec );
void rvec_Invert( rvec, rvec );
void rvec_OuterProduct( rtensor, rvec, rvec );
int  rvec_isZero( rvec );

HOST_DEVICE inline real rvec_Dot( rvec, rvec );
HOST_DEVICE inline void rvec_Scale( rvec, real, rvec );
HOST_DEVICE inline real rvec_Norm_Sqr( rvec );
HOST_DEVICE inline void rvec_Random( rvec );
HOST_DEVICE inline void rvec_MakeZero( rvec );
HOST_DEVICE inline void rvec_Add( rvec, rvec );
HOST_DEVICE inline void rvec_Copy( rvec, rvec );
HOST_DEVICE inline void rvec_Cross( rvec, rvec, rvec );
HOST_DEVICE inline void rvec_ScaledAdd( rvec, real, rvec );
HOST_DEVICE inline void rvec_ScaledSum( rvec, real, rvec, real, rvec );
HOST_DEVICE inline void rvec_iMultiply( rvec, ivec, rvec );
HOST_DEVICE inline real rvec_Norm( rvec );

void rtensor_MakeZero( rtensor );
void rtensor_Multiply( rtensor, rtensor, rtensor );
void rtensor_MatVec( rvec, rtensor, rvec );
void rtensor_Scale( rtensor, real, rtensor );
void rtensor_Add( rtensor, rtensor );
void rtensor_ScaledAdd( rtensor, real, rtensor );
void rtensor_Sum( rtensor, rtensor, rtensor );
void rtensor_ScaledSum( rtensor, real, rtensor, real, rtensor );
void rtensor_Scale( rtensor, real, rtensor );
void rtensor_Copy( rtensor, rtensor );
void rtensor_Identity( rtensor );
void rtensor_Transpose( rtensor, rtensor );
real rtensor_Det( rtensor );
real rtensor_Trace( rtensor );

void Print_rTensor(FILE*,rtensor);

int  ivec_isZero( ivec );
int  ivec_isEqual( ivec, ivec );
void ivec_MakeZero( ivec );
void ivec_rScale( ivec, real, rvec );


HOST_DEVICE inline void ivec_Copy( ivec, ivec );
HOST_DEVICE inline void ivec_Scale( ivec, real, ivec );
HOST_DEVICE inline void ivec_Sum( ivec, ivec, ivec );

/*
 * Code which is common to multiple HOST and DEVICE
 *
 */

HOST_DEVICE inline real Dot( real* v1, real* v2, int k ) 
{
  real ret = 0;
  
  for( --k; k>=0; --k )
    ret +=  v1[k] * v2[k];

  return ret;
}




/////////////////////////////
//rvec functions
/////////////////////////////

HOST_DEVICE inline void rvec_MakeZero( rvec v ) 
{
  v[0] = v[1] = v[2] = ZERO;
}

HOST_DEVICE inline void rvec_Add( rvec ret, rvec v ) 
{
  ret[0] += v[0];
  ret[1] += v[1]; 
  ret[2] += v[2];
}

HOST_DEVICE inline void rvec_Copy( rvec dest, rvec src )
{
  dest[0] = src[0], dest[1] = src[1], dest[2] = src[2];
}

HOST_DEVICE inline void rvec_Cross( rvec ret, rvec v1, rvec v2 )
{
  ret[0] = v1[1] * v2[2] - v1[2] * v2[1];
  ret[1] = v1[2] * v2[0] - v1[0] * v2[2];
  ret[2] = v1[0] * v2[1] - v1[1] * v2[0];
}

HOST_DEVICE inline void rvec_ScaledAdd( rvec ret, real c, rvec v ) 
{
  ret[0] += c * v[0], ret[1] += c * v[1], ret[2] += c * v[2];
}

HOST_DEVICE inline void rvec_ScaledSum( rvec ret, real c1, rvec v1 ,real c2, rvec v2 )
{
  ret[0] = c1 * v1[0] + c2 * v2[0]; 
  ret[1] = c1 * v1[1] + c2 * v2[1];
  ret[2] = c1 * v1[2] + c2 * v2[2];
}

HOST_DEVICE inline void rvec_Random( rvec v ) 
{
  v[0] = Random(2.0)-1.0;
  v[1] = Random(2.0)-1.0;
  v[2] = Random(2.0)-1.0;
}

HOST_DEVICE inline real rvec_Norm_Sqr( rvec v ) 
{
  return SQR(v[0]) + SQR(v[1]) + SQR(v[2]);
}

HOST_DEVICE inline void rvec_Scale( rvec ret, real c, rvec v ) 
{
  ret[0] = c * v[0], ret[1] = c * v[1], ret[2] = c * v[2];
}

HOST_DEVICE inline real rvec_Dot( rvec v1, rvec v2 )
{
  return v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2];
}

HOST_DEVICE inline void rvec_iMultiply( rvec r, ivec v1, rvec v2 )
{
  r[0] = v1[0] * v2[0];
  r[1] = v1[1] * v2[1];
  r[2] = v1[2] * v2[2];
}

HOST_DEVICE inline real rvec_Norm( rvec v ) 
{
  return SQRT( SQR(v[0]) + SQR(v[1]) + SQR(v[2]) );
}



/////////////////
//ivec functions
/////////////////


HOST_DEVICE inline void ivec_Copy( ivec dest , ivec src )
{
	dest[0] = src[0], dest[1] = src[1], dest[2] = src[2];
}

HOST_DEVICE inline void ivec_Scale( ivec dest, real C, ivec src )
{
  dest[0] = C * src[0];
  dest[1] = C * src[1];
  dest[2] = C * src[2];
}

HOST_DEVICE inline void ivec_Sum( ivec dest, ivec v1, ivec v2 )
{
  dest[0] = v1[0] + v2[0];
  dest[1] = v1[1] + v2[1];
  dest[2] = v1[2] + v2[2];
}



/////////////////
//vector functions
/////////////////
HOST_DEVICE inline void Vector_Sum( real* dest, real c, real* v, real d, real* y, int k ) 
{
	for (k--; k>=0; k--)
   	dest[k] = c * v[k] + d * y[k];
}

HOST_DEVICE inline void Vector_Scale( real* dest, real c, real* v, int k ) 
{
	for (k--; k>=0; k--)
   	dest[k] = c * v[k];
}

HOST_DEVICE inline void Vector_Add( real* dest, real c, real* v, int k ) 
{
	for (k--; k>=0; k--)
   	dest[k] += c * v[k];
}

#endif
