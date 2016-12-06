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

#ifndef __CUDA_UTILS_H_
#define __CUDA_UTILS_H_

#include "mytypes.h"

#include <stdlib.h>
#include <stdio.h>

#define IDX2C(i,j,ld) (((j)*(ld))+(i))


#ifdef __cplusplus
extern "C"  {
#endif

static __inline__ void modify( cublasHandle_t handle, float *m, int ldm, int n, int p, int q, float alpha, float beta )
{
    cublasSscal( handle, n - p, &alpha, &m[IDX2C(p, q, ldm)], ldm );
    cublasSscal( handle, ldm - p, &beta, &m[IDX2C(p, q, ldm)], 1 );
}

void cuda_malloc( void **, int , int , int );
void cuda_free( void *, int );
void cuda_memset( void *, int , size_t , int );
void copy_host_device( void *, void *, int , enum cudaMemcpyKind, int );
void copy_device( void *, void *, int , int );

void compute_blocks( int *, int *, int );
void compute_nearest_pow_2( int blocks, int *result );

void print_device_mem_usage( );

#define cusparseCheckError(cusparseStatus) __cusparseCheckError (cusparseStatus, __FILE__, __LINE__)
static inline void __cusparseCheckError( cusparseStatus_t cusparseStatus, const char *file, const int line )
{
    if ( cusparseStatus != CUSPARSE_STATUS_SUCCESS )
    {
        fprintf (stderr, "failed .. %s:%d -- error code %d \n", __FILE__, __LINE__, cusparseStatus);
        exit (-1);
    }
    return;
}


#define cublasCheckError(cublasStatus) __cublasCheckError (cublasStatus, __FILE__, __LINE__)
static inline void __cublasCheckError( cublasStatus_t cublasStatus, const char *file, const int line )
{
    if ( cublasStatus != CUBLAS_STATUS_SUCCESS )
    {
        fprintf( stderr, "failed .. %s:%d -- error code %d \n", __FILE__, __LINE__, cublasStatus );
        exit( -1 );
    }
    return;
}


#define cudaCheckError() __cudaCheckError( __FILE__, __LINE__ )
static inline void __cudaCheckError( const char *file, const int line )
{
    cudaError err = cudaGetLastError( );
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "Failed .. %s:%d -- gpu erro code %d\n", file, line, err );
        exit( -1 );
    }

    // More careful checking. However, this will affect performance.
    /*
    err = cudaDeviceSynchronize( );
    if( cudaSuccess != err )
    {
        exit( -1 );
    }
    */

    return;
}

#ifdef __cplusplus
}
#endif


#endif
