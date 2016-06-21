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

#include "reduction.h"
#include "vector.h"
#include "mytypes.h"


GLOBAL void Cuda_reduction(const real *input, real *per_block_results, const size_t n)
{
    extern __shared__ real sdata[];
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    real x = 0;

    if(i < n)
    {
        x = input[i];
    }
    sdata[threadIdx.x] = x;
    __syncthreads();

    for(int offset = blockDim.x / 2; offset > 0; offset >>= 1)
    {
        if(threadIdx.x < offset)
        {
            sdata[threadIdx.x] += sdata[threadIdx.x + offset];
        }

        __syncthreads();
    }

    if(threadIdx.x == 0)
    {
        per_block_results[blockIdx.x] = sdata[0];
    }
}

GLOBAL void Cuda_Norm (const real *input, real *per_block_results, const size_t n, int pass)
{
    extern __shared__ real sdata[];
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    real x = 0;

    if(i < n)
    {
        if (pass == INITIAL)
            x = SQR (input[i]);
        else 
            x = input[i];
    }
    sdata[threadIdx.x] = x;
    __syncthreads();

    for(int offset = blockDim.x / 2; offset > 0; offset >>= 1)
    {
        if(threadIdx.x < offset)
        {
            sdata[threadIdx.x] += sdata[threadIdx.x + offset];
        }

        __syncthreads();
    }

    if(threadIdx.x == 0)
    {
        if (pass == INITIAL)
            per_block_results[blockIdx.x] = sdata[0];
        else
            per_block_results[blockIdx.x] = SQRT (sdata[0]);
    }
}

GLOBAL void Cuda_Dot (const real *a, const real *b, real *per_block_results, const size_t n )
{
    extern __shared__ real sdata[];
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    real x = 0;

    if(i < n)
    {
        x = a[i] * b[i];
    }
    sdata[threadIdx.x] = x;
    __syncthreads();

    for(int offset = blockDim.x / 2; offset > 0; offset >>= 1)
    {
        if(threadIdx.x < offset)
        {
            sdata[threadIdx.x] += sdata[threadIdx.x + offset];
        }

        __syncthreads();
    }

    if(threadIdx.x == 0)
    {
        per_block_results[blockIdx.x] = sdata[0];
    }
}

GLOBAL void Cuda_matrix_col_reduction(const real *input, real *per_block_results, const size_t n)
{
    extern __shared__ real sdata[];
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    real x = 0;

    if(i < n)
    {
        x = input[i * n + i];
    }
    sdata[threadIdx.x] = x;
    __syncthreads();

    for(int offset = blockDim.x / 2; offset > 0; offset >>= 1)
    {
        if(threadIdx.x < offset)
        {
            sdata[threadIdx.x] += sdata[threadIdx.x + offset];
        }

        __syncthreads();
    }

    if(threadIdx.x == 0) 
    {
        per_block_results[blockIdx.x] = sdata[0];
    }
}






GLOBAL void Cuda_reduction(const int *input, int *per_block_results, const size_t n)
{
    extern __shared__ int sh_input[];
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    real x = 0;

    if(i < n)
    {
        x = input[i];
    }
    sh_input[threadIdx.x] = x;
    __syncthreads();

    for(int offset = blockDim.x / 2; offset > 0; offset >>= 1)
    {
        if(threadIdx.x < offset)
        {
            sh_input[threadIdx.x] += sh_input[threadIdx.x + offset];
        }

        __syncthreads();
    }

    if(threadIdx.x == 0)
    {
        per_block_results[blockIdx.x] = sh_input[0];
    }
}


GLOBAL void Cuda_reduction_rvec (rvec *input, rvec *results, size_t n)
{
    extern __shared__ rvec svec_data[];
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    rvec x;

    rvec_MakeZero (x);

    if(i < n)
    {
        rvec_Copy (x, input[i]);
    }

    rvec_Copy (svec_data[threadIdx.x], x);
    __syncthreads();

    for(int offset = blockDim.x / 2; offset > 0; offset >>= 1)
    {
        if(threadIdx.x < offset)
        {
            rvec_Add (svec_data[threadIdx.x], svec_data[threadIdx.x + offset]);
        }

        __syncthreads();
    }

    if(threadIdx.x == 0)
    {
        //rvec_Copy (results[blockIdx.x], svec_data[0]);
        rvec_Add (results[blockIdx.x], svec_data[0]);
    }
}

//////////////////////////////////////////////////
//vector functions
//////////////////////////////////////////////////

GLOBAL void Cuda_Vector_Sum( real* dest, real c, real* v, real d, real* y, int k ) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= k) return;

    dest[i] = c * v[i] + d * y[i];
}

GLOBAL void Cuda_Vector_Scale( real* dest, real c, real* v, int k ) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= k) return;

    dest[i] = c * v[i];
}

GLOBAL void Cuda_Vector_Add( real* dest, real c, real* v, int k )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= k) return;

    dest[i] += c * v[i];
}
