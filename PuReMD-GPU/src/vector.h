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


/* global to make OpenMP shared (Vector_isZero) */
unsigned int ret;
/* global to make OpenMP shared (Dot, Norm) */
real ret2;


#ifdef __cplusplus
extern "C"  {
#endif

int Vector_isZero( const real * const, const unsigned int );
void Vector_MakeZero( real * const, const unsigned int );
void Vector_Copy( real * const, const real * const, const unsigned int );
void Vector_Print( FILE * const, const char * const, const real * const, const unsigned int );
real Norm( const real * const, const unsigned int );

void rvec_Sum( rvec, const rvec, const rvec );
real rvec_ScaledDot( const real, const rvec, const real, const rvec );
void rvec_Multiply( rvec, const rvec, const rvec );
void rvec_Divide( rvec, const rvec, const rvec );
void rvec_iDivide( rvec, const rvec, const ivec );
void rvec_Invert( rvec, const rvec );
void rvec_OuterProduct( rtensor, const rvec, const rvec );
int rvec_isZero( const rvec );

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
void ivec_rScale( ivec, const real, const rvec );


static inline HOST_DEVICE real Dot( const real * const v1, const real * const v2, const unsigned int k )
{
    unsigned int i;

    #pragma omp master
    {
        ret2 = ZERO;
    }

    #pragma omp barrier


    #pragma omp for reduction(+: ret2) schedule(static)
    for ( i = 0; i < k; ++i )
    {
        ret2 += v1[i] * v2[i];
    }

    return ret2;
}


static inline HOST_DEVICE void rvec_MakeZero( rvec v )
{
    v[0] = ZERO;
    v[1] = ZERO;
    v[2] = ZERO;
}


static inline HOST_DEVICE void rvec_Add( rvec ret, const rvec v )
{
    ret[0] += v[0];
    ret[1] += v[1];
    ret[2] += v[2];
}


static inline HOST_DEVICE void rvec_Copy( rvec dest, const rvec src )
{
    dest[0] = src[0];
    dest[1] = src[1];
    dest[2] = src[2];
}


static inline HOST_DEVICE void rvec_Cross( rvec ret, const rvec v1, const rvec v2 )
{
    ret[0] = v1[1] * v2[2] - v1[2] * v2[1];
    ret[1] = v1[2] * v2[0] - v1[0] * v2[2];
    ret[2] = v1[0] * v2[1] - v1[1] * v2[0];
}


static inline HOST_DEVICE void rvec_ScaledAdd( rvec ret, const real c, const rvec v )
{
    ret[0] += c * v[0];
    ret[1] += c * v[1];
    ret[2] += c * v[2];
}


static inline HOST_DEVICE void rvec_ScaledSum( rvec ret, const real c1, const rvec v1,
        const real c2, const rvec v2 )
{
    ret[0] = c1 * v1[0] + c2 * v2[0];
    ret[1] = c1 * v1[1] + c2 * v2[1];
    ret[2] = c1 * v1[2] + c2 * v2[2];
}


static inline HOST_DEVICE void rvec_Random( rvec v )
{
    v[0] = Random(2.0) - 1.0;
    v[1] = Random(2.0) - 1.0;
    v[2] = Random(2.0) - 1.0;
}


static inline HOST_DEVICE real rvec_Norm_Sqr( const rvec v )
{
    return SQR(v[0]) + SQR(v[1]) + SQR(v[2]);
}


static inline HOST_DEVICE void rvec_Scale( rvec ret, const real c, const rvec v )
{
    ret[0] = c * v[0];
    ret[1] = c * v[1];
    ret[2] = c * v[2];
}


static inline HOST_DEVICE real rvec_Dot( const rvec v1, const rvec v2 )
{
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
}


static inline HOST_DEVICE void rvec_iMultiply( rvec r, const ivec v1, const rvec v2 )
{
    r[0] = v1[0] * v2[0];
    r[1] = v1[1] * v2[1];
    r[2] = v1[2] * v2[2];
}


static inline HOST_DEVICE real rvec_Norm( const rvec v )
{
    return SQRT( SQR(v[0]) + SQR(v[1]) + SQR(v[2]) );
}


static inline HOST_DEVICE void ivec_Copy( ivec dest , const ivec src )
{
    dest[0] = src[0];
    dest[1] = src[1];
    dest[2] = src[2];
}


static inline HOST_DEVICE void ivec_Scale( ivec dest, const real C, const ivec src )
{
    dest[0] = C * src[0];
    dest[1] = C * src[1];
    dest[2] = C * src[2];
}


static inline HOST_DEVICE void ivec_Sum( ivec dest, const ivec v1, const ivec v2 )
{
    dest[0] = v1[0] + v2[0];
    dest[1] = v1[1] + v2[1];
    dest[2] = v1[2] + v2[2];
}


static inline HOST_DEVICE void Vector_Sum( real * const dest, const real c,
        const real * const v, const real d, const real * const y,
        const unsigned int k )
{
    unsigned int i;

    #pragma omp for schedule(static)
    for ( i = 0; i < k; ++i )
    {
        dest[i] = c * v[i] + d * y[i];
    }
}


static inline HOST_DEVICE void Vector_Scale( real * const dest, const real c,
        const real * const v, const unsigned int k )
{
    unsigned int i;

    #pragma omp for schedule(static)
    for ( i = 0; i < k; ++i )
    {
        dest[i] = c * v[i];
    }
}


static inline HOST_DEVICE void Vector_Add( real * const dest, const real c,
        const real * const v, const unsigned int k )
{
    unsigned int i;

    #pragma omp for schedule(static)
    for ( i = 0; i < k; ++i )
    {
        dest[i] += c * v[i];
    }
}

#ifdef __cplusplus
}
#endif


#endif
