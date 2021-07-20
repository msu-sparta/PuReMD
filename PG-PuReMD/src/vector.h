/*----------------------------------------------------------------------
  PuReMD - Purdue ReaxFF Molecular Dynamics Program

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

#include "reax_types.h"

#include <assert.h>

#include "random.h"

#ifdef __cplusplus
extern "C"  {
#endif

#if defined(LAMMPS_REAX) || defined(PURE_REAX)

/* check if all entries of a dense vector are sufficiently close to zero
 *
 * inputs:
 *  v: dense vector
 *  k: number of entries in v
 * output: TRUE if all entries are sufficiently close to zero, FALSE otherwise
 */
CUDA_HOST_DEVICE static inline int Vector_isZero( real const * const v, int k )
{
    int i, ret;

    assert( k >= 0 );

    ret = TRUE;

    for ( i = 0; i < k; ++i )
    {
        if ( FABS( v[k] ) > ALMOST_ZERO )
        {
            ret = FALSE;
            break;
        }
    }

    return ret;
}


/* sets all entries of a dense vector to zero
 *
 * inputs:
 *  v: dense vector
 *  k: number of entries in v
 * output: v with entries set to zero
 */
CUDA_HOST_DEVICE static inline void Vector_MakeZero( real * const v, int k )
{
    int i;

    assert( k >= 0 );

    for ( i = 0; i < k; ++i )
    {
        v[i] = 0;
    }
}


/* copy the entries from one vector to another
 *
 * inputs:
 *  v: dense vector to copy
 *  k: number of entries in v
 * output:
 *  dest: vector copied into
 */
CUDA_HOST_DEVICE static inline void Vector_Copy( real * const dest, real const * const v, int k )
{
    int i;

    assert( k >= 0 );

    for ( i = 0; i < k; ++i )
    {
        dest[i] = v[i];
    }
}


/* copy the entries from one vector to another
 *
 * inputs:
 *  v: dense vector to copy
 *  k: number of entries in v
 * output:
 *  dest: vector copied into
 */
CUDA_HOST_DEVICE static inline void Vector_Copy_rvec2( rvec2 * const dest, rvec2 const * const v, int k )
{
    int i;

    assert( k >= 0 );

    for ( i = 0; i < k; ++i )
    {
        dest[i][0] = v[i][0];
        dest[i][1] = v[i][1];
    }
}


CUDA_HOST_DEVICE static inline void Vector_Copy_From_rvec2( real * const dst, rvec2 const * const src,
        int index, int k )
{
    int i;

    assert( k >= 0 );
    assert( index >= 0 && index <= 1 );

    for ( i = 0; i < k; ++i )
    {
        dst[i] = src[i][index];
    }
}


CUDA_HOST_DEVICE static inline void Vector_Copy_To_rvec2( rvec2 * const dst, real const * const src,
        int index, int k )
{
    int i;

    assert( k >= 0 );
    assert( index >= 0 && index <= 1 );

    for ( i = 0; i < k; ++i )
    {
        dst[i][index] = src[i];
    }
}


/* scales the entries of a dense vector by a constant
 *
 * inputs:
 *  c: scaling constant
 *  v: dense vector whose entries to scale
 *  k: number of entries in v
 * output:
 *  dest: with entries scaled
 */
CUDA_HOST_DEVICE static inline void Vector_Scale( real * const dest, real c,
        real const * const v, int k )
{
    int i;

    assert( k >= 0 );

    for ( i = 0; i < k; ++i )
    {
        dest[i] = c * v[i];
    }
}


/* computed the scaled sum of two dense vector and store
 * the result in a third vector (SAXPY operation in BLAS)
 *
 * inputs:
 *  c, d: scaling constants
 *  v, y: dense vector whose entries to scale
 *  k: number of entries in the vectors
 * output:
 *  dest: vector containing the scaled sum
 */
CUDA_HOST_DEVICE static inline void Vector_Sum( real * const dest, real c,
        real const * const v, real d, real const * const y, int k )
{
    int i;

    assert( k >= 0 );

    for ( i = 0; i < k; ++i )
    {
        dest[i] = c * v[i] + d * y[i];
    }
}


/* computed the scaled sum of two dense vector and store
 * the result in a third vector (SAXPY operation in BLAS)
 *
 * inputs:
 *  c, d: scaling constants
 *  v, y: dense vector whose entries to scale
 *  k: number of entries in the vectors
 * output:
 *  dest: vector containing the scaled sum
 */
CUDA_HOST_DEVICE static inline void Vector_Sum_rvec2( rvec2 * const dest, real c0, real c1,
        rvec2 const * const v, real d0, real d1, rvec2 const * const y, int k )
{
    int i;

    assert( k >= 0 );

    for ( i = 0; i < k; ++i )
    {
        dest[i][0] = c0 * v[i][0] + d0 * y[i][0];
        dest[i][1] = c1 * v[i][1] + d1 * y[i][1];
    }
}


/* add the scaled sum of a dense vector to another vector
 * and store in-place
 *
 * inputs:
 *  c: scaling constant
 *  v: dense vector whose entries to scale
 *  k: number of entries in the vectors
 * output:
 *  dest: vector to accumulate with the scaled sum
 */
CUDA_HOST_DEVICE static inline void Vector_Add( real * const dest, real c,
        real const * const v, int k )
{
    int i;

    assert( k >= 0 );

    for ( i = 0; i < k; ++i )
    {
        dest[i] += c * v[i];
    }
}


/* add the scaled sum of a dense vector to another vector
 * and store in-place
 *
 * inputs:
 *  c: scaling constant
 *  v: dense vector whose entries to scale
 *  k: number of entries in the vectors
 * output:
 *  dest: vector to accumulate with the scaled sum
 */
CUDA_HOST_DEVICE static inline void Vector_Add_rvec2( rvec2 * const dest, real c0, real c1,
        rvec2 const * const v, int k )
{
    int i;

    assert( k >= 0 );

    for ( i = 0; i < k; ++i )
    {
        dest[i][0] += c0 * v[i][0];
        dest[i][1] += c1 * v[i][1];
    }
}


/* compute the local portions of the inner product of two dense vectors
 *
 * inputs:
 *  workspace: storage container for workspace structures
 *  v1, v2: dense vectors
 *  k: number of entries in the vectors
 * output:
 *  dot: inner product of the two vector
 */
CUDA_HOST_DEVICE static inline real Dot_local( real const * const v1,
        real const * const v2, int k )
{
    int i;
    real sum;

    assert( k >= 0 );

    sum = 0.0;

    for ( i = 0; i < k; ++i )
    {
        sum += v1[i] * v2[i];
    }

    return sum;
}


/* compute the local portions of the inner product of two dense vectors
 *
 * inputs:
 *  workspace: storage container for workspace structures
 *  v1, v2: dense vectors
 *  k: number of entries in the vectors
 * output:
 *  dot: inner product of the two vectors
 */
CUDA_HOST_DEVICE static inline void Dot_local_rvec2( rvec2 const * const v1,
        rvec2 const * const v2, int k, real * const sum1, real * const sum2 )
{
    int i;

    assert( k >= 0 );

    *sum1 = 0.0;
    *sum2 = 0.0;

    for ( i = 0; i < k; ++i )
    {
        *sum1 += v1[i][0] * v2[i][0];
        *sum2 += v1[i][1] * v2[i][1];
    }
}


CUDA_HOST_DEVICE static inline void Vector_Print( FILE * const fout,
        char const * const vname, real const * const v, int k )
{
    int i;

    assert( k >= 0 );

    fprintf( fout, "%s:", vname );
    for ( i = 0; i < k; ++i )
    {
        fprintf( fout, "%8.3f\n", v[i] );
    }
    fprintf( fout, "\n" );
}


CUDA_HOST_DEVICE static inline void rvec_Copy( rvec dest, const rvec src )
{
    int i;

    for ( i = 0; i < 3; ++i )
    {
        dest[i] = src[i];
    }
}


CUDA_HOST_DEVICE static inline void rvec_Scale( rvec ret, real c, const rvec v )
{
    int i;

    for ( i = 0; i < 3; ++i )
    {
        ret[i] = c * v[i];
    }
}


CUDA_HOST_DEVICE static inline void rvec_Add( rvec ret, const rvec v )
{
    int i;

    for ( i = 0; i < 3; ++i )
    {
        ret[i] += v[i];
    }
}


CUDA_HOST_DEVICE static inline void rvec_ScaledAdd( rvec ret, real c, const rvec v )
{
    int i;

    for ( i = 0; i < 3; ++i )
    {
        ret[i] += c * v[i];
    }
}


CUDA_HOST_DEVICE static inline void rvec_Sum( rvec ret, const rvec v1, const rvec v2 )
{
    int i;

    for ( i = 0; i < 3; ++i )
    {
        ret[i] = v1[i] + v2[i];
    }
}


CUDA_HOST_DEVICE static inline void rvec_ScaledSum( rvec ret, real c1, const rvec v1,
        real c2, const rvec v2 )
{
    int i;

    for ( i = 0; i < 3; ++i )
    {
        ret[i] = c1 * v1[i] + c2 * v2[i];
    }
}


CUDA_HOST_DEVICE static inline real rvec_Dot( const rvec v1, const rvec v2 )
{
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
}


CUDA_HOST_DEVICE static inline real rvec_ScaledDot( real c1, const rvec v1,
        real c2, const rvec v2 )
{
    return (c1 * c2) * (v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]);
}


CUDA_HOST_DEVICE static inline void rvec_Multiply( rvec r, const rvec v1, const rvec v2 )
{
    int i;

    for ( i = 0; i < 3; ++i )
    {
        r[i] = v1[i] * v2[i];
    }
}


CUDA_HOST_DEVICE static inline void rvec_iMultiply( rvec r, const ivec v1, const rvec v2 )
{
    int i;

    for ( i = 0; i < 3; ++i )
    {
        r[i] = v1[i] * v2[i];
    }
}


CUDA_HOST_DEVICE static inline void rvec_Divide( rvec r, const rvec v1, const rvec v2 )
{
    int i;

    for ( i = 0; i < 3; ++i )
    {
        r[i] = v1[i] / v2[i];
    }
}


CUDA_HOST_DEVICE static inline void rvec_iDivide( rvec r, const rvec v1, const ivec v2 )
{
    int i;

    for ( i = 0; i < 3; ++i )
    {
        r[i] = v1[i] / v2[i];
    }
}


CUDA_HOST_DEVICE static inline void rvec_Invert( rvec r, const rvec v )
{
    int i;

    for ( i = 0; i < 3; ++i )
    {
        r[i] = 1.0 / v[i];
    }
}


CUDA_HOST_DEVICE static inline void rvec_Cross( rvec ret,
        const rvec v1, const rvec v2 )
{
    int i;

    for ( i = 0; i < 3; ++i )
    {
        ret[i] = v1[(i + 1) % 3] * v2[(i + 2) % 3]
            - v1[(i + 2) % 3] * v2[(i + 1) % 3];
    }
}


CUDA_HOST_DEVICE static inline void rvec_OuterProduct( rtensor r,
        const rvec v1, const rvec v2 )
{
    int i, j;

    for ( i = 0; i < 3; ++i )
    {
        for ( j = 0; j < 3; ++j )
        {
            r[i][j] = v1[i] * v2[j];
        }
    }
}


CUDA_HOST_DEVICE static inline real rvec_Norm_Sqr( rvec v )
{
    return SQR(v[0]) + SQR(v[1]) + SQR(v[2]);
}


CUDA_HOST_DEVICE static inline real rvec_Norm( rvec v )
{
    return SQRT( SQR(v[0]) + SQR(v[1]) + SQR(v[2]) );
}


CUDA_HOST_DEVICE static inline int rvec_isZero( rvec v )
{
    if ( FABS(v[0]) > ALMOST_ZERO ||
            FABS(v[1]) > ALMOST_ZERO ||
            FABS(v[2]) > ALMOST_ZERO )
    {
        return FALSE;
    }

    return TRUE;
}


CUDA_HOST_DEVICE static inline void rvec_MakeZero( rvec v )
{
    int i;

    for ( i = 0; i < 3; ++i )
    {
        v[i] = 0.0;
    }
}


#if defined(PURE_REAX)
static inline void rvec_Random( rvec v )
{
    int i;

    for ( i = 0; i < 3; ++i )
    {
        v[i] = Random( 2.0 ) - 1.0;
    }
}
#endif


CUDA_HOST_DEVICE static inline void rtensor_Multiply( rtensor ret,
        const rtensor m1, const rtensor m2 )
{
    int i, j, k;
    rtensor temp;

    // check if the result matrix is the same as one of m1, m2.
    // if so, we cannot modify the contents of m1 or m2, so
    // we have to use a temp matrix.
    if ( ret == m1 || ret == m2 )
    {
        for ( i = 0; i < 3; ++i )
        {
            for ( j = 0; j < 3; ++j )
            {
                temp[i][j] = 0;

                for ( k = 0; k < 3; ++k )
                {
                    temp[i][j] += m1[i][k] * m2[k][j];
                }
            }
        }

        for ( i = 0; i < 3; ++i )
        {
            for ( j = 0; j < 3; ++j )
            {
                ret[i][j] = temp[i][j];
            }
        }
    }
    else
    {
        for ( i = 0; i < 3; ++i )
        {
            for ( j = 0; j < 3; ++j )
            {
                ret[i][j] = 0;

                for ( k = 0; k < 3; ++k )
                {
                    ret[i][j] += m1[i][k] * m2[k][j];
                }
            }
        }
    }
}


CUDA_HOST_DEVICE static inline void rtensor_MatVec( rvec ret,
        const rtensor m, const rvec v )
{
    int i;
    rvec temp;

    // if ret is the same vector as v, we cannot modify the
    // contents of v until all computation is finished.
    if ( ret == v )
    {
        for ( i = 0; i < 3; ++i )
        {
            temp[i] = m[i][0] * v[0] + m[i][1] * v[1] + m[i][2] * v[2];
        }

        for ( i = 0; i < 3; ++i )
        {
            ret[i] = temp[i];
        }
    }
    else
    {
        for ( i = 0; i < 3; ++i )
        {
            ret[i] = m[i][0] * v[0] + m[i][1] * v[1] + m[i][2] * v[2];
        }
    }
}


CUDA_HOST_DEVICE static inline void rtensor_Scale( rtensor ret,
        real c, const rtensor m )
{
    int i, j;

    for ( i = 0; i < 3; ++i )
    {
        for ( j = 0; j < 3; ++j )
        {
            ret[i][j] = c * m[i][j];
        }
    }
}


CUDA_HOST_DEVICE static inline void rtensor_Add( rtensor ret,
        const rtensor t )
{
    int i, j;

    for ( i = 0; i < 3; ++i )
    {
        for ( j = 0; j < 3; ++j )
        {
            ret[i][j] += t[i][j];
        }
    }
}


CUDA_HOST_DEVICE static inline void rtensor_ScaledAdd( rtensor ret,
        real c, const rtensor t )
{
    int i, j;

    for ( i = 0; i < 3; ++i )
    {
        for ( j = 0; j < 3; ++j )
        {
            ret[i][j] += c * t[i][j];
        }
    }
}


CUDA_HOST_DEVICE static inline void rtensor_Sum( rtensor ret,
        const rtensor t1, const rtensor t2 )
{
    int i, j;

    for ( i = 0; i < 3; ++i )
    {
        for ( j = 0; j < 3; ++j )
        {
            ret[i][j] = t1[i][j] + t2[i][j];
        }
    }
}


CUDA_HOST_DEVICE static inline void rtensor_ScaledSum( rtensor ret,
        real c1, const rtensor t1, real c2, const rtensor t2 )
{
    int i, j;

    for ( i = 0; i < 3; ++i )
    {
        for ( j = 0; j < 3; ++j )
        {
            ret[i][j] = c1 * t1[i][j] + c2 * t2[i][j];
        }
    }
}


CUDA_HOST_DEVICE static inline void rtensor_Copy( rtensor ret,
        const rtensor t )
{
    int i, j;

    for ( i = 0; i < 3; ++i )
    {
        for ( j = 0; j < 3; ++j )
        {
            ret[i][j] = t[i][j];
        }
    }
}


CUDA_HOST_DEVICE static inline void rtensor_Identity( rtensor t )
{
    t[0][0] = 1.0;
    t[0][1] = 0.0;
    t[0][2] = 0.0;
    t[1][0] = 0.0;
    t[1][1] = 1.0;
    t[1][2] = 0.0;
    t[2][0] = 0.0;
    t[2][1] = 0.0;
    t[2][2] = 1.0;
}


CUDA_HOST_DEVICE static inline void rtensor_MakeZero( rtensor t )
{
    int i, j;

    for ( i = 0; i < 3; ++i )
    {
        for ( j = 0; j < 3; ++j )
        {
            t[i][j] = 0.0;
        }
    }
}


CUDA_HOST_DEVICE static inline void rtensor_Transpose( rtensor ret,
        const rtensor t )
{
    int i, j;

    for ( i = 0; i < 3; ++i )
    {
        for ( j = 0; j < 3; ++j )
        {
            ret[i][j] = t[j][i];
        }
    }
}


CUDA_HOST_DEVICE static inline real rtensor_Det( const rtensor t )
{
    return t[0][0] * (t[1][1] * t[2][2] - t[1][2] * t[2][1] )
            + t[0][1] * (t[1][2] * t[2][0] - t[1][0] * t[2][2] )
            + t[0][2] * (t[1][0] * t[2][1] - t[1][1] * t[2][0] );
}


CUDA_HOST_DEVICE static inline real rtensor_Trace( rtensor t )
{
    return t[0][0] + t[1][1] + t[2][2];
}


CUDA_HOST_DEVICE static inline void Print_rTensor( FILE * const fp,
        const rtensor t )
{
    int i, j;

    for ( i = 0; i < 3; i++ )
    {
        fprintf( fp, "[" );

        for ( j = 0; j < 3; j++ )
        {
            fprintf( fp, "%8.3f,\t", t[i][j] );
        }

        fprintf( fp, "]\n" );
    }
}


CUDA_HOST_DEVICE static inline void ivec_MakeZero( ivec v )
{
    int i;

    for ( i = 0; i < 3; ++i )
    {
        v[i] = 0;
    }
}


CUDA_HOST_DEVICE static inline void ivec_Copy( ivec dest, const ivec src )
{
    int i;

    for ( i = 0; i < 3; ++i )
    {
        dest[i] = src[i];
    }
}


CUDA_HOST_DEVICE static inline void ivec_Scale( ivec dest, real C, const ivec src )
{
    int i;

    for ( i = 0; i < 3; ++i )
    {
        dest[i] = (int)(C * src[i]);
    }
}


CUDA_HOST_DEVICE static inline void ivec_rScale( ivec dest, real C, const rvec src )
{
    int i;

    for ( i = 0; i < 3; ++i )
    {
        dest[i] = (int)(C * src[i]);
    }
}


CUDA_HOST_DEVICE static inline int ivec_isZero( const ivec v )
{
    int i, ret;

    ret = TRUE;
    
    for ( i = 0; i < 3; ++i )
    {
        if ( v[i] != 0 )
        {
            ret = FALSE;
            break;
        }
    }

    return ret;
}


CUDA_HOST_DEVICE static inline int ivec_isEqual( const ivec v1, const ivec v2 )
{
    int i, ret;

    ret = TRUE;
    
    for ( i = 0; i < 3; ++i )
    {
        if ( v1[i] != v2[i] )
        {
            ret = FALSE;
            break;
        }
    }

    return ret;
}


CUDA_HOST_DEVICE static inline void ivec_Sum( ivec dest, const ivec v1, const ivec v2 )
{
    int i;

    for ( i = 0; i < 3; ++i )
    {
        dest[i] = v1[i] + v2[i];
    }
}


CUDA_HOST_DEVICE static inline void ivec_ScaledSum( ivec dest,
        int k1, const ivec v1, int k2, const ivec v2 )
{
    int i;

    for ( i = 0; i < 3; ++i )
    {
        dest[i] = k1 * v1[i] + k2 * v2[i];
    }
}


CUDA_HOST_DEVICE static inline void ivec_Add( ivec dest, const ivec v )
{
    int i;

    for ( i = 0; i < 3; ++i )
    {
        dest[i] += v[i];
    }
}


CUDA_HOST_DEVICE static inline void ivec_ScaledAdd( ivec dest,
        int k, const ivec v )
{
    int i;

    for ( i = 0; i < 3; ++i )
    {
        dest[i] += k * v[i];
    }
}



CUDA_HOST_DEVICE static inline void ivec_Max( ivec res,
        const ivec v1, const ivec v2 )
{
    int i;

    for ( i = 0; i < 3; ++i )
    {
        res[i] = MAX( v1[i], v2[i] );
    }
}


CUDA_HOST_DEVICE static inline void ivec_Max3( ivec res,
        const ivec v1, const ivec v2, const ivec v3 )
{
    int i;

    for ( i = 0; i < 3; ++i )
    {
        res[i] = MAX3( v1[i], v2[i], v3[i] );
    }
}
#endif


#ifdef __cplusplus
}
#endif


#endif
