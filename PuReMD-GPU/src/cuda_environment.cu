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

#include "cuda_environment.h"

#include "cuda_utils.h"


void Setup_Cuda_Environment( int rank, int nprocs, int gpus_per_node )
{

    int deviceCount = 0;
    cudaError_t flag;
    cublasHandle_t cublasHandle;
    cusparseHandle_t cusparseHandle;
    cusparseMatDescr_t matdescriptor;
    
    flag = cudaGetDeviceCount( &deviceCount );

    if ( flag != cudaSuccess || deviceCount < 1 )
    {
        fprintf( stderr, "ERROR: no CUDA capable device(s) found. Terminating...\n" );
        exit( 1 );
    }

    //Calculate the # of GPUs per processor
    //and assign the GPU for each process
    //TODO: handle condition where # CPU procs > # GPUs
    cudaSetDevice( rank % deviceCount );

#if defined(__CUDA_DEBUG__)
    fprintf( stderr, "p:%d is using GPU: %d \n", rank, rank % deviceCount );
#endif

    //CHANGE ORIGINAL
    //cudaDeviceSetLimit( cudaLimitStackSize, 8192 );
    //cudaDeviceSetCacheConfig( cudaFuncCachePreferL1 );
    //cudaCheckError( );

    cublasCheckError( cublasCreate(&cublasHandle) );

    cusparseCheckError( cusparseCreate(&cusparseHandle) );
    cusparseCheckError( cusparseCreateMatDescr(&matdescriptor) );
    cusparseSetMatType( matdescriptor, CUSPARSE_MATRIX_TYPE_GENERAL );
    cusparseSetMatIndexBase( matdescriptor, CUSPARSE_INDEX_BASE_ZERO );

}


void Cleanup_Cuda_Environment( )
{
    cudaDeviceReset( );
    cudaDeviceSynchronize( );
}
