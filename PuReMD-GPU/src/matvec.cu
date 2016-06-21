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


#include "matvec.h"

//one thread per row
GLOBAL void Cuda_Matvec (sparse_matrix H, real *vec, real *results, int rows)
{
    real results_row = 0;
    int col;
    real val;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= rows) return;

    for (int c = H.start[i]; c < H.end[i]; c++)
    {
        col = H.entries [c].j;
        val = H.entries[c].val;

        results_row += val * vec [col];
    }

    results [i] = results_row;
}

//32 thread warp per matrix row.
//invoked as follows
// <<< system->N, 32 >>>
GLOBAL void Cuda_Matvec_csr (sparse_matrix H, real *vec, real *results, int num_rows)
{
    extern __shared__ real vals [];
    int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    int warp_id = thread_id / 32;
    int lane = thread_id & (32 - 1);

    int row_start;
    int row_end;

    // one warp per row
    //int row = warp_id;
    int row = warp_id;
    //if (row < num_rows)
    {
        vals[threadIdx.x] = 0;

        if (row < num_rows) {
            row_start = H.start[row];
            row_end = H.end[row];

            // compute running sum per thread
            for(int jj = row_start + lane; jj < row_end; jj += 32)
                vals[threadIdx.x] += H.entries[jj].val * vec [ H.entries[jj].j ];
            //vals[threadIdx.x] += H.val[jj] * vec [ H.j[jj] ];
        }

        __syncthreads ();

        // parallel reduction in shared memory
        //SIMD instructions with a WARP are synchronous -- so we do not need to synch here
        if (lane < 16) vals[threadIdx.x] += vals[threadIdx.x + 16]; __syncthreads();
        if (lane < 8) vals[threadIdx.x] += vals[threadIdx.x + 8]; __syncthreads ();
        if (lane < 4) vals[threadIdx.x] += vals[threadIdx.x + 4]; __syncthreads ();
        if (lane < 2) vals[threadIdx.x] += vals[threadIdx.x + 2]; __syncthreads ();
        if (lane < 1) vals[threadIdx.x] += vals[threadIdx.x + 1]; __syncthreads ();

        // first thread writes the result
        if (lane == 0 && row < num_rows)
            results[row] = vals[threadIdx.x];
    }
}
