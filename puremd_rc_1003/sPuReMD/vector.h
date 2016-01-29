/*----------------------------------------------------------------------
  SerialReax - Reax Force Field Simulator

  Copyright (2010) Purdue University
  Hasan Metin Aktulga, haktulga@cs.purdue.edu
  Joseph Fogarty, jcfogart@mail.usf.edu
  Sagar Pandit, pandit@usf.edu
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

int Vector_isZero( const real * const, const unsigned int );
void Vector_MakeZero( real * const, const unsigned int );
void Vector_Copy( real * const, const real * const, const unsigned int );
void Vector_Scale( real * const, const real, const real * const, const unsigned int );
void Vector_Sum( real * const, const real, const real * const, const real, const real * const, const unsigned int );
void Vector_Add( real * const, const real, const real * const, const unsigned int );
void Vector_Add2( real * const, const real * const, const unsigned int );
void Vector_Print( FILE * const, const char * const, const real * const, const unsigned int );
real Dot( const real * const, const real * const, const unsigned int );
real Norm( const real * const, const unsigned int );

void rvec_Copy( rvec, const rvec );
void rvec_Scale( rvec, const real, const rvec );
void rvec_Add( rvec, const rvec );
void rvec_ScaledAdd( rvec, const real, const rvec );
void rvec_Sum( rvec, const rvec, const rvec );
void rvec_ScaledSum( rvec, const real, const rvec, const real, const rvec );
real rvec_Dot( const rvec, const rvec );
real rvec_ScaledDot( const real, const rvec, const real, const rvec );
void rvec_Multiply( rvec, const rvec, const rvec );
void rvec_iMultiply( rvec, const ivec, const rvec );
void rvec_Divide( rvec, const rvec, const rvec );
void rvec_iDivide( rvec, const rvec, const ivec );
void rvec_Invert( rvec, const rvec );
void rvec_Cross( rvec, const rvec, const rvec );
void rvec_OuterProduct( rtensor, const rvec, const rvec );
real rvec_Norm_Sqr( const rvec );
real rvec_Norm( const rvec );
int rvec_isZero( const rvec );
void rvec_MakeZero( rvec );
void rvec_Random( rvec );

void rtensor_MakeZero( rtensor );
void rtensor_Multiply( rtensor, rtensor, rtensor );
void rtensor_MatVec( rvec, rtensor, const rvec );
void rtensor_Scale( rtensor, const real, rtensor );
void rtensor_Add( rtensor, rtensor );
void rtensor_ScaledAdd( rtensor, const real, rtensor );
void rtensor_Sum( rtensor, rtensor, rtensor );
void rtensor_ScaledSum( rtensor, const real, rtensor, const real, rtensor );
void rtensor_Scale( rtensor, const real, rtensor );
void rtensor_Copy( rtensor, rtensor );
void rtensor_Identity( rtensor );
void rtensor_Transpose( rtensor, rtensor );
real rtensor_Det( rtensor );
real rtensor_Trace( rtensor );

void Print_rTensor(FILE * const, rtensor);

int ivec_isZero( const ivec );
int ivec_isEqual( const ivec, const ivec );
void ivec_MakeZero( ivec );
void ivec_Copy( ivec, const ivec );
void ivec_Scale( ivec, const real, const ivec );
void ivec_rScale( ivec, const real, const rvec );
void ivec_Sum( ivec, const ivec, const ivec );

#endif
