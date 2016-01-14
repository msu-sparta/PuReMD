

#include "matvec.h"
#include "cuda_shuffle.h"

//one thread per row
CUDA_GLOBAL void k_matvec (sparse_matrix H, real *vec, real *results, int rows)
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
//CUDA_GLOBAL void __launch_bounds__(384, 16) k_matvec_csr(sparse_matrix H, real *vec, real *results, int num_rows)
CUDA_GLOBAL void k_matvec_csr(sparse_matrix H, real *vec, real *results, int num_rows)
{
#if defined(__SM_35__)
	real vals;
#else
	extern __shared__ real vals [];
#endif
	int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
	int warp_id = thread_id / MATVEC_KER_THREADS_PER_ROW;
	int lane = thread_id & ( MATVEC_KER_THREADS_PER_ROW - 1);

	int row_start;
	int row_end;

	// one warp per row
	//int row = warp_id;
	int row = warp_id;
	//if (row < num_rows)
	{
#if defined(__SM_35__)
		vals = 0;
#else
		vals[threadIdx.x] = 0;
#endif
		
		if (row < num_rows) {
			row_start = H.start[row];
			row_end = H.end[row];

			// compute running sum per thread
			for(int jj = row_start + lane; jj < row_end; jj += MATVEC_KER_THREADS_PER_ROW)
#if defined(__SM_35__)
				vals += H.entries[jj].val * vec [ H.entries[jj].j ];
		}
#else
				vals[threadIdx.x] += H.entries[jj].val * vec [ H.entries[jj].j ];
		}
		__syncthreads ();
#endif

		// parallel reduction in shared memory
		//SIMD instructions with a WARP are synchronous -- so we do not need to synch here
#if defined(__SM_35__)
        for (int x = MATVEC_KER_THREADS_PER_ROW >> 1; x >= 1; x/=2)
                vals += shfl( vals, x );

	if (lane == 0 && row < num_rows)
		results[row] = vals;
#else
		if (lane < 16) vals[threadIdx.x] += vals[threadIdx.x + 16]; __syncthreads();
		if (lane < 8) vals[threadIdx.x] += vals[threadIdx.x + 8]; __syncthreads ();
		if (lane < 4) vals[threadIdx.x] += vals[threadIdx.x + 4]; __syncthreads ();
		if (lane < 2) vals[threadIdx.x] += vals[threadIdx.x + 2]; __syncthreads ();
		if (lane < 1) vals[threadIdx.x] += vals[threadIdx.x + 1]; __syncthreads ();

		// first thread writes the result
		if (lane == 0 && row < num_rows)
			results[row] = vals[threadIdx.x];
#endif
	}
}
